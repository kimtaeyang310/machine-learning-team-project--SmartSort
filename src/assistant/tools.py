"""Ollama 비서가 요청할 수 있는 SmartSort 작업과 입력 검사를 담당합니다."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.agent.decision import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    REVIEW_REQUIRED_CLASSES,
)
from src.joint import EXTENSIONS, MODEL_DIR
from src.predict import (
    PREDICT_DIR,
    predict_folder,
    save_prediction_artifacts,
)


MAX_IMAGES = 60
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "predictions"

ALLOWED_FRUITS = {"banana", "mandarin", "tomato"}
ALLOWED_ACTIONS = {
    "analyze_folder",
    "summarize_recent",
    "list_unknown",
    "explain_model",
}
RIPENESS_LABELS = {"unripe", "ripe", "overripe"}


def _summarize_saved_timing(rows):
    """CSV의 처리 시간을 읽어 잘못된 값은 건너뛰고 속도 지표를 계산합니다."""
    elapsed_times = []

    for row in rows:
        try:
            elapsed = float(row.get("total_ms") or 0)
        except (TypeError, ValueError):
            continue

        if elapsed >= 0:
            elapsed_times.append(elapsed)

    processing_ms = sum(elapsed_times)
    completed_count = sum(
        row.get("status") in {"ok", "unknown"}
        for row in rows
    )

    performance = {
        "mean_ms": float(np.mean(elapsed_times)) if elapsed_times else None,
        "p95_ms": (
            float(np.percentile(elapsed_times, 95))
            if elapsed_times else None
        ),
        "fps": (
            completed_count * 1000 / processing_ms
            if processing_ms else None
        ),
        "processing_ms": processing_ms,
    }
    return performance


def _normalize_fruit(fruit: str | None) -> str | None:
    """None 또는 'all'은 전체 과일, 이름을 지정하면 해당 과일을 뜻합니다."""
    if fruit is None:
        return None

    if not isinstance(fruit, str):
        raise ValueError("fruit은 과일 이름 문자열이어야 합니다.")

    fruit = fruit.strip().lower()

    if fruit in {"", "all"}:
        return None

    if fruit not in ALLOWED_FRUITS:
        raise ValueError(
            "지원하지 않는 과일입니다. banana, mandarin, tomato 중에서 선택하세요."
        )

    return fruit


def _resolve_folder(selected_folder: str | Path | None) -> Path:
    """UI 선택 폴더를 사용하고, 미선택 시 기본 예측 폴더를 사용합니다."""
    folder = Path(selected_folder) if selected_folder else PREDICT_DIR
    folder = folder.expanduser().resolve()

    if not folder.is_dir():
        raise ValueError(f"폴더를 찾을 수 없습니다: {folder}")

    return folder


def _image_paths(folder: Path) -> list[Path]:
    """폴더 바로 아래에서 지원하는 이미지 파일을 찾습니다."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def _new_csv_path() -> Path:
    """기존 파일을 덮어쓰지 않는 새 CSV 경로를 만듭니다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = OUTPUT_DIR / f"{timestamp}.csv"
    suffix = 1

    while target.exists():
        target = OUTPUT_DIR / f"{timestamp}_{suffix}.csv"
        suffix += 1

    return target


def _read_json(value: Any) -> dict[str, Any]:
    """CSV 안의 JSON 문자열을 딕셔너리로 읽습니다."""
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value:
        return {}

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _retake_items(
    rows: list[dict[str, Any]],
    fruit_filter: str | None = None,
) -> list[dict[str, Any]]:
    """재촬영 판정을 받은 이미지의 표시 정보를 모읍니다."""
    items = []

    for row in rows:
        decision = row.get("agent_decision")

        # 새 예측 결과는 agent_decision을, CSV 결과는 JSON 열을 사용합니다.
        if not isinstance(decision, dict):
            decision = _read_json(row.get("agent_decision_json"))

        if decision.get("action") != "retake":
            continue

        fruit = row.get("fruit")
        if fruit_filter is not None and fruit != fruit_filter:
            continue

        image_path = row.get("image_path") or row.get("image") or ""

        items.append({
            "image": Path(str(image_path)).name,
            "fruit": fruit,
            "ripeness": row.get("ripeness"),
            "confidence": decision.get("confidence"),
            "reason": decision.get("reason"),
        })

    return items


def _candidate_fruit(row: dict[str, Any]) -> str | None:
    """예측 결과에서 후보 과일을 가져옵니다."""
    identity = row.get("fruit_identity")

    if isinstance(identity, str):
        identity = _read_json(identity)

    if isinstance(identity, dict):
        candidate = identity.get("candidate_fruit")
        if candidate:
            return str(candidate)

    candidate_class = row.get("candidate_class")
    if isinstance(candidate_class, str) and "_" in candidate_class:
        return candidate_class.split("_", 1)[0]

    return None


def _json_safe(value: Any) -> Any:
    """numpy 값, Path 등 JSON으로 바로 표현하기 어려운 값을 변환합니다."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    return value


def _summarize_prediction_counts(rows, fruit_filter):
    """메모리 예측과 CSV 행에 같은 집계 규칙을 적용합니다."""
    counts_by_fruit = {
        name: Counter({label: 0 for label in RIPENESS_LABELS})
        for name in ALLOWED_FRUITS
    }

    all_unknown_rows = []
    unknown_by_candidate = Counter()
    failure_count = 0

    for row in rows:
        status = row.get("status")

        if status == "ok":
            predicted_fruit = row.get("fruit")
            ripeness = row.get("ripeness")

            if (
                predicted_fruit in counts_by_fruit
                and ripeness in RIPENESS_LABELS
            ):
                counts_by_fruit[predicted_fruit][ripeness] += 1

        elif status == "unknown":
            all_unknown_rows.append(row)
            candidate = _candidate_fruit(row)
            if candidate in ALLOWED_FRUITS:
                unknown_by_candidate[candidate] += 1

        elif status == "failed":
            failure_count += 1

    if fruit_filter is None:
        ripeness_counts = {
            name: dict(counts_by_fruit[name])
            for name in sorted(ALLOWED_FRUITS)
        }
        unknown_candidate_count = sum(unknown_by_candidate.values())
    else:
        ripeness_counts = dict(counts_by_fruit[fruit_filter])
        unknown_candidate_count = unknown_by_candidate[fruit_filter]
    return {"ripeness_counts": ripeness_counts, "unknown_count": len(all_unknown_rows),
            "unknown_candidate_count": unknown_candidate_count, "failure_count": failure_count}


def analyze_folder(
    fruit: str | None = None,
    *,
    selected_folder: str | Path | None = None,
) -> dict[str, Any]:
    """폴더를 예측하고 집계, 재촬영 권고, 속도, CSV 경로를 반환합니다."""
    try:
        fruit_filter = _normalize_fruit(fruit)
        folder = _resolve_folder(selected_folder)
        paths = _image_paths(folder)
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}

    if not paths:
        return {
            "ok": False,
            "error": "선택한 폴더에 지원되는 이미지가 없습니다.",
        }

    if len(paths) > MAX_IMAGES:
        return {
            "ok": False,
            "error": (
                f"사진이 {len(paths)}장 있어 한 번에 처리할 수 있는 "
                f"{MAX_IMAGES}장을 초과합니다. 사진 수를 {MAX_IMAGES}장 이하로 "
                "줄여서 다시 시도해 주세요."
            ),
            "image_count": len(paths),
            "maximum_images": MAX_IMAGES,
        }

    try:
        # 기존 main.py의 예측 흐름을 재사용합니다.
        results, timing = predict_folder(folder)
    except (OSError, RuntimeError, ValueError) as error:
        return {
            "ok": False,
            "error": f"예측을 완료하지 못했습니다: {error}",
        }

    # 각 이미지 결과에서 agent_decision.action == "retake"인 항목을 모읍니다.
    retake_images = _retake_items(results, fruit_filter)

    counts = _summarize_prediction_counts(results, fruit_filter)

    try:
        csv_path, manifest_path = save_prediction_artifacts(
            results,
            _new_csv_path(),
            source_folder=folder,
        )
    except (OSError, ValueError, RuntimeError) as error:
        return {
            "ok": False,
            "error": f"예측은 끝났지만 결과 파일을 저장하지 못했습니다: {error}",
        }

    return _json_safe({
        "ok": True,
        "action": "analyze_folder",
        "fruit_filter": fruit_filter or "all",
        "image_count": len(results),
        **counts,
        "retake_count": len(retake_images),
        "retake_images": retake_images,
        "performance": {
            "mean_ms": timing.get("mean_ms"),
            "p95_ms": timing.get("p95_ms"),
            "fps": timing.get("fps"),
            "processing_ms": timing.get("processing_ms"),
        },
        "csv_path": csv_path,
        "manifest_path": manifest_path,
    })


def _latest_csv() -> Path | None:
    """예측 결과 폴더에서 가장 최근 CSV를 찾습니다."""
    if not OUTPUT_DIR.is_dir():
        return None

    files = list(OUTPUT_DIR.glob("*.csv"))
    if not files:
        return None

    return max(files, key=lambda path: path.stat().st_mtime)


def _read_latest_rows() -> tuple[Path | None, list[dict[str, str]]]:
    path = _latest_csv()
    if path is None:
        return None, []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return path, list(csv.DictReader(file))


def _validate_fruit_filter(fruit: str | None) -> str | None:
    return _normalize_fruit(fruit)


def summarize_recent(fruit: str | None = None) -> dict[str, Any]:
    """가장 최근 예측 결과와 재촬영 권고를 요약합니다."""
    try:
        fruit_filter = _validate_fruit_filter(fruit)
    except ValueError as error:
        return {"ok": False, "error": str(error)}

    try:
        csv_path, rows = _read_latest_rows()
    except (OSError, csv.Error) as error:
        return {"ok": False, "error": f"최근 결과를 읽지 못했습니다: {error}"}

    if csv_path is None:
        return {"ok": False, "error": "저장된 예측 결과가 아직 없습니다."}

    retake_images = _retake_items(rows, fruit_filter)

    counts = _summarize_prediction_counts(rows, fruit_filter)

    performance = _summarize_saved_timing(rows)

    return _json_safe({
        "ok": True,
        "csv_path": csv_path,
        "fruit_filter": fruit_filter or "all",
        "image_count": len(rows),
        **counts,
        "retake_count": len(retake_images),
        "retake_images": retake_images,
        "performance": performance,
    })


def list_unknown(fruit: str | None = None) -> dict[str, Any]:
    """최근 결과에서 모르는 과일 판정 이미지 목록을 반환합니다."""
    try:
        fruit_filter = _validate_fruit_filter(fruit)
    except ValueError as error:
        return {"ok": False, "error": str(error)}

    try:
        csv_path, rows = _read_latest_rows()
    except (OSError, csv.Error) as error:
        return {"ok": False, "error": f"최근 결과를 읽지 못했습니다: {error}"}

    if csv_path is None:
        return {"ok": False, "error": "저장된 예측 결과가 아직 없습니다."}

    all_unknown_rows = [
        row for row in rows
        if row.get("status") == "unknown"
    ]

    unknown_images = []

    for row in all_unknown_rows:
        identity = _read_json(row.get("fruit_identity_json"))
        candidate = identity.get("candidate_fruit")

        if not candidate:
            candidate_class = row.get("candidate_class", "")
            if "_" in candidate_class:
                candidate = candidate_class.split("_", 1)[0]

        if fruit_filter is not None and candidate != fruit_filter:
            continue

        unknown_images.append({
            "image": Path(row.get("image_path", "")).name,
            "candidate_fruit": candidate,
            "candidate_class": row.get("candidate_class"),
            "rejection_score": row.get("rejection_score"),
            "rejection_threshold": row.get("rejection_threshold"),
        })

    return {
        "ok": True,
        "csv_path": str(csv_path),
        "fruit_filter": fruit_filter or "all",
        "unknown_total_count": len(all_unknown_rows),
        "unknown_count": len(unknown_images),
        "images": unknown_images,
    }


def explain_model() -> dict[str, Any]:
    """현재 모델과 모르는 과일·에이전트 판정 기준을 반환합니다."""
    bundle_path = MODEL_DIR / "bundle.json"
    policy_path = MODEL_DIR / "policy.joblib"

    if not bundle_path.is_file() or not policy_path.is_file():
        return {
            "ok": False,
            "error": "현재 모델 또는 판정 기준 파일을 찾을 수 없습니다.",
        }

    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        policy = joblib.load(policy_path)
    except Exception as error:
        return {
            "ok": False,
            "error": f"모델 정보를 읽지 못했습니다: {error}",
        }

    return _json_safe({
        "ok": True,
        "model_name": bundle.get("model_name"),
        "image_size": bundle.get("imgsz"),
        "unknown_detection_method": policy.get("method"),
        "unknown_detection_thresholds": policy.get("thresholds"),
        "agent_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "review_required_classes": [
            list(item) for item in sorted(REVIEW_REQUIRED_CLASSES)
        ],
        "note": (
            "모르는 과일 거리 기준과 에이전트 확률 기준은 서로 다릅니다. "
            "모델 확률은 정답을 보장하지 않습니다."
        ),
    })


def execute_smartfarm_task(
    action: str,
    fruit: str | None = None,
    *,
    selected_folder: str | Path | None = None,
    all_fruits: bool = False,
) -> dict[str, Any]:
    """허용된 작업과 과일 범위를 검사한 뒤 실행합니다."""
    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        return {
            "ok": False,
            "error": "지원하지 않는 작업입니다.",
            "allowed_actions": sorted(ALLOWED_ACTIONS),
        }

    if not isinstance(all_fruits, bool):
        return {
            "ok": False,
            "error": "전체 과일 분석 여부는 True 또는 False여야 합니다.",
        }

    # 먼저 정규화해야 'all'이나 빈 문자열도 동일하게 검사할 수 있습니다.
    try:
        fruit_filter = _normalize_fruit(fruit)
    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
        }

    if action == "analyze_folder":
        if fruit_filter is None and not all_fruits:
            return {
                "ok": False,
                "error": (
                    "분석할 과일을 지정하거나 "
                    "전체 과일 분석을 명시해 주세요."
                ),
            }

        if fruit_filter is not None and all_fruits:
            return {
                "ok": False,
                "error": (
                    "특정 과일 분석과 전체 과일 분석을 "
                    "동시에 요청할 수 없습니다."
                ),
            }

        return analyze_folder(
            fruit_filter,
            selected_folder=selected_folder,
        )

    if action == "summarize_recent":
        return summarize_recent(fruit_filter)

    if action == "list_unknown":
        return list_unknown(fruit_filter)

    return explain_model()