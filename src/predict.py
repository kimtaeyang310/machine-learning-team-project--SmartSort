#6. predict : 새 사진에 같은 처리 과정을 적용해 예측

"""폴더의 이미지를 순차 예측하고 처리 시간을 측정.

실행:
    python src/predict.py
또는
    python -m src.predict

학습 때 저장한 전처리 설정과 특징 순서를 그대로 사용합니다.
"""

from pathlib import Path
import csv
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from time import perf_counter


# ============================================================
# 1. 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# VS Code에서 파일을 직접 실행해도 src 패키지를 찾도록 합니다.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. 라이브러리
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.model_bundle import MODEL_PATH, FRUITS, validate_bundle, select_model
from src.agent import decide_next_action
from src.agent.decision import DEFAULT_CONFIDENCE_THRESHOLD, REVIEW_REQUIRED_CLASSES
from src.fruit_classifier import FruitClassifier, MODEL_PATH as FRUIT_MODEL_PATH


# ============================================================
# 3. 경로 설정
# ============================================================

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREDICT_DIR = PROJECT_ROOT / "data" / "predict_list"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================
# 4. 모델 불러오기
# ============================================================

def load_model(model_path):
    """학습할 때 저장한 모델과 설정을 불러옵니다."""

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}\n"
            "먼저 train.py를 실행하세요."
        )

    bundle = joblib.load(model_path)

    validate_bundle(bundle)
    # 한 장씩 처리할 때 RF의 작업 분배 비용을 피합니다. 저장 모델은 변경하지 않습니다.
    entries = bundle["models"].values() if "models" in bundle else [bundle]
    for entry in entries:
        configure_prediction_jobs(entry["model"])
        if getattr(entry["model"], "input_kind", None) == "image":
            entry["model"].load()
    return bundle


def configure_prediction_jobs(model, n_jobs=1):
    """Random Forest에만 적용하며 학습 설정과 다른 모델은 건드리지 않습니다."""
    if isinstance(model, RandomForestClassifier):
        model.set_params(n_jobs=n_jobs)
    elif hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            configure_prediction_jobs(step, n_jobs)


# ============================================================
# 5. 이미지 경로 확인
# ============================================================

def resolve_image_path(image_path):
    """입력된 이미지 경로를 실제 파일 경로로 변환합니다."""

    path = Path(image_path)

    # 상대 경로라면 프로젝트 최상위 폴더를 기준으로 합니다.
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {path}"
        )

    return path


# ============================================================
# 6. 이미지 → 특징 벡터
# ============================================================

def extract_image_features(image_path, bundle, timings=None):
    """학습 때와 동일한 전처리와 특징 추출을 수행합니다."""

    feature_names = bundle["feature_names"]
    preprocess_config = dict(bundle["preprocess_config"])
    # 특징 추출은 HSV·마스크만 사용하므로 표시용 이미지 두 개는 만들지 않습니다.
    preprocess_config.setdefault("include_preview", False)

    if not feature_names:
        raise ValueError("저장된 특징 이름이 없습니다.")

    # --------------------------------------------------------
    # 1) 이미지 전처리
    # --------------------------------------------------------

    processed = preprocess(
        image_path,
        timings=timings,
        **preprocess_config,
    )

    # --------------------------------------------------------
    # 2) 특징 추출
    # --------------------------------------------------------

    features = extract_features(processed)

    if not isinstance(features, dict) or not features:
        raise ValueError(
            "extract_features()가 올바른 특징 딕셔너리를 반환하지 않았습니다."
        )

    # 학습 때와 특징 종류가 같은지 확인
    if set(features) != set(feature_names):
        raise ValueError(
            "새 이미지의 특징 이름이 학습 당시와 다릅니다."
        )

    # --------------------------------------------------------
    # 3) 학습 때와 똑같은 순서로 정렬
    # --------------------------------------------------------

    values = np.asarray(
        [features[name] for name in feature_names],
        dtype=np.float64,
    )

    if values.shape != (len(feature_names),):
        raise ValueError(
            "특징 벡터의 크기가 학습 당시와 다릅니다."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "특징에 NaN 또는 무한대가 포함되어 있습니다."
        )

    # DataFrame으로 만들어 학습 때의 X와 같은 형태로 모델에 전달
    X = pd.DataFrame(
        [values],
        columns=feature_names,
    )

    return X


# ============================================================
# 7. 숙도 예측
# ============================================================

def predict_loaded(image_path, bundle, model):
    """모델 재로딩 없이 한 장 처리. 파일 읽기부터 확률 계산까지 측정합니다."""
    started = perf_counter()
    timings = {}
    X = ([image_path] if getattr(model, "input_kind", None) == "image"
         else extract_image_features(image_path, bundle, timings=timings))
    prepared = perf_counter()
    inverse = {value: name for name, value in bundle["label_mapping"].items()}
    probabilities = None
    if hasattr(model, "predict_proba"):
        # RF·LR·KNN·DT는 확률의 최댓값으로 예측합니다.
        # predict()까지 호출하면 같은 확률 계산이 중복됩니다.
        values = np.asarray(model.predict_proba(X))[0]
        classes = np.asarray(model.classes_)
        if values.shape != classes.shape or not np.isfinite(values).all():
            raise ValueError("모델의 예측 확률 형식이 올바르지 않습니다.")
        if not set(classes).issubset(inverse):
            raise ValueError("모델 확률에 알 수 없는 라벨이 있습니다.")
        label = int(classes[np.argmax(values)])
        probabilities = {
            inverse[int(c)]: float(v)
            for c, v in zip(classes, values)
        }
    else:
        # 현재 SVM은 확률을 지원하지 않으므로 기존 predict() 사용.
        label = int(np.asarray(model.predict(X))[0])
    if label not in inverse:
        raise ValueError(f"알 수 없는 예측 라벨: {label}")
    finished = perf_counter()
    return {"image_path": str(image_path), "ripeness": inverse[label],
            "probabilities": probabilities,
            "preprocessing_ms": timings,
            "feature_ms": (prepared - started) * 1000,
            "prediction_ms": (finished - prepared) * 1000,
            "total_ms": (finished - started) * 1000}


def predict(image_path, model_path=MODEL_PATH, *, fruit=None, fruit_mode="image",
            fruit_model_path=FRUIT_MODEL_PATH):
    """기본은 이미지 기반 판별. filename 모드는 기존 명시적 종류 입력을 지원합니다."""
    path = resolve_image_path(image_path)
    if fruit_mode not in {"image", "filename"}:
        raise ValueError("fruit_mode: image 또는 filename")
    if fruit_mode == "image":
        identity = FruitClassifier(fruit_model_path).classify(path)
        if not identity["accepted"]:
            print("모르는 과일입니다. 숙도 판정을 생략합니다.")
            return "unknown"
        fruit = identity["fruit"]
    elif fruit is None:
        fruit = infer_fruit(path)
    bundle = load_model(model_path)
    model, name = select_model(bundle, fruit)
    result = predict_loaded(path, bundle, model)
    print(f"{fruit} / {name}: {result['ripeness']} ({result['total_ms']:.2f} ms)")
    return result["ripeness"]


def infer_fruit(image_path):
    """파일명의 과일 토큰 또는 기존 바나나 데이터셋 형식만 사용합니다."""
    stem = Path(image_path).stem.lower()
    found = {fruit for fruit in FRUITS
             if re.search(r"(?:^|[^a-z])" + fruit + r"(?:$|[^a-z])", stem)}
    # 기존 바나나 데이터셋 이름. 숙도 정답에는 사용하지 않습니다.
    if re.fullmatch(r"(?:overripe|ripe|semi_ripe)_\d+_(?:jpg|jpeg|png)\.rf\.[a-f0-9]+(?:_copy\d+)?", stem):
        found.add("banana")
    if len(found) != 1:
        raise ValueError("파일명에서 과일 종류를 구분할 수 없습니다. banana / mandarin / tomato 중 하나를 넣으세요.")
    return found.pop()


def predict_folder(folder_path=PREDICT_DIR, model_path=MODEL_PATH, *, fruit_mode="image",
                   fruit_model_path=FRUIT_MODEL_PATH):
    """폴더 바로 아래 이미지 전체를 처리합니다. 실패 파일도 시간과 이유를 기록합니다."""
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"예측 폴더가 없습니다: {folder}")
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"지원하는 이미지가 없습니다: {folder}")
    if fruit_mode not in {"image", "filename"}:
        raise ValueError("fruit_mode: image 또는 filename")
    load_started = perf_counter()
    detector = FruitClassifier(fruit_model_path) if fruit_mode == "image" else None
    bundle = load_model(model_path)
    load_ms = (perf_counter() - load_started) * 1000
    print(f"\n폴더: {folder} / 사진: {len(paths)}장")
    print("이미지로 과일 종류와 모름 여부를 판별합니다." if detector else
          "호환 모드: 파일명으로 과일을 구분합니다. 모르는 과일 판별은 수행하지 않습니다.")
    print(f"모델 로딩: {load_ms:.2f} ms (이미지 처리 시간과 별도)")
    results = []
    batch_started = perf_counter()
    for index, path in enumerate(paths, 1):
        started = perf_counter()
        identity = None
        identity_ms = None
        try:
            identity_started = perf_counter()
            identity = detector.classify(path) if detector else None
            identity_ms = (perf_counter() - identity_started) * 1000
            if identity is not None and not identity["accepted"]:
                result = {"image_path": str(path), "status": "unknown", "fruit": "unknown",
                          "error": "", "ripeness": None, "probabilities": None,
                          "fruit_identity": identity, "fruit_identity_ms": identity_ms,
                          "agent_decision": {"action": "human_review", "reason": "모르는 과일입니다. 숙도 판정을 생략합니다.",
                                             "suggested_label": None, "confidence": None,
                                             "requires_confirmation": True}}
            else:
                fruit = identity["fruit"] if identity else infer_fruit(path)
                model, name = select_model(bundle, fruit)
                result = predict_loaded(path, bundle, model)
                result.update(status="ok", error="", fruit=fruit, model=name,
                              fruit_identity=identity, fruit_identity_ms=identity_ms)
                result["agent_decision"] = decide_next_action(result)
            result["total_ms"] = (perf_counter() - started) * 1000
        except Exception as error:
            result = {"image_path": str(path), "status": "failed", "error": str(error),
                      "fruit_identity": identity, "fruit_identity_ms": identity_ms,
                      "total_ms": (perf_counter() - started) * 1000}
        results.append(result)
        if result["status"] == "ok":
            probability = result["probabilities"]
            confidence = f" / 확률 {probability[result['ripeness']]:.1%}" if probability else ""
            print(f"[{index}/{len(paths)}] {path.name} → {result['fruit']} / {result['model']} / {result['ripeness']}{confidence}"
                  f" | 준비 {result['feature_ms']:.2f} + 예측 {result['prediction_ms']:.2f}"
                  f" | 전체 {result['total_ms']:.2f} ms")
            decision = result["agent_decision"]
            print(f"    에이전트: {decision['action']} — {decision['reason']}")
        elif result["status"] == "unknown":
            print(f"[{index}/{len(paths)}] {path.name} → 모르는 과일입니다. 숙도 판정 생략"
                  f" | {result['total_ms']:.2f} ms")
        else:
            print(f"[{index}/{len(paths)}] {path.name} → 실패 ({result['total_ms']:.2f} ms): {result['error']}")
    batch_ms = (perf_counter() - batch_started) * 1000
    successful = [r for r in results if r["status"] == "ok"]
    unknown_count = sum(r["status"] == "unknown" for r in results)
    failed_count = sum(r["status"] == "failed" for r in results)
    times = np.asarray([r["total_ms"] for r in successful])
    total_processing_ms = sum(r["total_ms"] for r in results)
    summary = {"count": len(results), "success_count": len(successful),
               "failure_count": failed_count, "unknown_count": unknown_count,
               "fruit_mode": fruit_mode, "model_load_ms": load_ms,
               "processing_ms": total_processing_ms, "batch_ms": batch_ms,
               "mean_ms": float(times.mean()) if len(times) else None,
               "p95_ms": float(np.percentile(times, 95)) if len(times) else None,
               "fps": (len(successful) + unknown_count) * 1000 / total_processing_ms if total_processing_ms else 0.0}
    stages = ("read_ms", "denoise_ms", "resize_ms", "manual_mask_ms", "kmeans_ms", "mask_other_ms", "finish_ms")
    measured = [r for r in successful if r.get("preprocessing_ms")]
    summary["preprocessing_mean_ms"] = {
        stage: float(np.mean([r["preprocessing_ms"][stage] for r in measured]))
        for stage in stages
    } if measured else {}
    if measured:
        labels = ("읽기", "노이즈 제거", "크기 조정", "수동 마스크 확인", "K-means", "기타 영역 분리", "마무리")
        print("\n전처리 단계별 평균(ms): " + " / ".join(
            f"{label} {summary['preprocessing_mean_ms'][stage]:.2f}"
            for label, stage in zip(labels, stages)))
    print(f"\n완료: 숙도 판정 {summary['success_count']} / 모르는 과일 {unknown_count} / 실패 {summary['failure_count']}")
    print(f"전체 순차 실행: {batch_ms / 1000:.3f}초 (결과 출력 포함, 모델 로딩 제외)")
    if len(times):
        print(f"성공 사진 평균: {summary['mean_ms']:.2f} ms / P95: {summary['p95_ms']:.2f} ms")
    print(f"처리 FPS: {summary['fps']:.2f} (정상 처리·모름 판정 장수 / 실패 포함 처리 시간, 결과 출력 제외)")
    goal = "판정 보류: 처리 실패 있음" if summary['failure_count'] else ("달성" if summary['fps'] >= 25 else "미달성")
    print(f"25 FPS 목표: {goal}")
    print("첫 이미지 포함 측정이며 워밍업은 하지 않습니다. 준비 시간은 이미지 읽기·전처리·영역 분리·특징 구성을 포함합니다.")
    return results, summary

def save_prediction_artifacts(results, output_csv, *, model_path=MODEL_PATH,
                              source_folder=PREDICT_DIR):
    """예측·에이전트 결과 CSV와 실행 환경 manifest를 저장합니다."""
    output_csv = Path(output_csv)
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_path", "status", "fruit", "model", "ripeness",
        "probabilities_json", "agent_decision_json", "error",
        "preprocessing_ms_json", "feature_ms", "prediction_ms", "total_ms",
        "fruit_identity_json", "fruit_identity_ms",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "image_path": result.get("image_path", ""),
                "status": result.get("status", ""),
                "fruit": result.get("fruit", ""),
                "model": result.get("model", ""),
                "ripeness": result.get("ripeness", ""),
                "probabilities_json": json.dumps(
                    result.get("probabilities"), ensure_ascii=False, sort_keys=True
                ),
                "agent_decision_json": json.dumps(
                    result.get("agent_decision"), ensure_ascii=False, sort_keys=True
                ),
                "error": result.get("error", ""),
                "preprocessing_ms_json": json.dumps(
                    result.get("preprocessing_ms"), ensure_ascii=False, sort_keys=True
                ),
                "feature_ms": result.get("feature_ms", ""),
                "prediction_ms": result.get("prediction_ms", ""),
                "total_ms": result.get("total_ms", ""),
                "fruit_identity_json": json.dumps(result.get("fruit_identity"), ensure_ascii=False, sort_keys=True),
                "fruit_identity_ms": result.get("fruit_identity_ms", ""),
            })

    model_path = Path(model_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_folder": str(Path(source_folder).resolve()),
        "model_bundle_path": str(model_path.resolve()),
        "model_bundle_sha256": (
            hashlib.sha256(model_path.read_bytes()).hexdigest()
            if model_path.is_file() else None
        ),
        "models_used": sorted({
            f"{row.get('fruit')}: {row.get('model')}"
            for row in results if row.get("status") == "ok"
        }),
        "agent_policy": {
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "human_review_classes": sorted(
                [fruit, ripeness] for fruit, ripeness in REVIEW_REQUIRED_CLASSES
            ),
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "result_count": len(results),
        "fruit_identity_models": sorted({row["fruit_identity"]["model_sha256"] for row in results
                                          if row.get("fruit_identity")}),
        "unknown_count": sum(row.get("status") == "unknown" for row in results),
        "failure_count": sum(row.get("status") == "failed" for row in results),
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_csv, manifest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="폴더 전체 숙도 예측 및 시간 측정")
    parser.add_argument("folder", nargs="?", default=str(PREDICT_DIR), help="기본: data/predict_list")
    parser.add_argument("--results-csv", type=Path,
                        default=PROJECT_ROOT / "outputs" / "predictions" /
                        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv"),
                        help="예측·에이전트 결과 CSV와 실행 manifest 저장 경로 (기본 outputs/predictions)")
    parser.add_argument("--fruit-mode", choices=("image", "filename"), default="image",
                        help="기본 image. filename은 모름 판별 없는 기존 호환 모드")
    parser.add_argument("--fruit-model", type=Path, default=FRUIT_MODEL_PATH)
    args = parser.parse_args()
    try:
        results, _ = predict_folder(args.folder, fruit_mode=args.fruit_mode, fruit_model_path=args.fruit_model)
        if args.results_csv:
            csv_path, manifest_path = save_prediction_artifacts(
                results, args.results_csv, source_folder=args.folder
            )
            print(f"결과 CSV 저장: {csv_path}")
            print(f"실행 manifest 저장: {manifest_path}")
    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n예측 중단:\n{error}")
        raise SystemExit(1)
