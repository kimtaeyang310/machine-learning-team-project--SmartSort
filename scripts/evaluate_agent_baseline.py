"""고정된 에이전트 평가 CSV를 읽어 같은 지표 보고서를 다시 생성합니다."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "docs" / "agent_evaluation" / "baseline_predictions.csv"
LABELS = ("unripe", "ripe", "overripe")
LABELED_FIELDS = {
    "fruit", "true_ripeness", "predicted_ripeness", "confidence",
    "agent_action", "group_id", "split",
}
PREDICTION_FIELDS = {
    "image_path", "status", "fruit", "ripeness",
    "probabilities_json", "agent_decision_json",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Excel에서 저장한 UTF-8 BOM CSV와 일반 UTF-8 CSV를 읽습니다."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def normalize_predictions(
    predictions: list[dict[str, str]], labels_path: Path
) -> list[dict[str, str]]:
    """Join exported predictions to independently verified labels by filename."""
    labels_by_name: dict[str, dict[str, str]] = {}
    for label in read_csv(labels_path):
        # Prefer an explicit name; otherwise use the basename from the capture path.
        name = (label.get("image_name") or Path(label.get("image_path", "")).name).strip()
        if not name:
            continue
        if name in labels_by_name:
            raise ValueError(f"Duplicate image_name in label CSV: {name}")
        # 모델 출력으로 추정한 라벨이 평가 정답에 섞이지 않도록 검수 완료분만 사용합니다.
        if label.get("ground_truth_status", "").lower() in {"confirmed", "verified"}:
            labels_by_name[name] = label

    normalized = []
    for prediction in predictions:
        # Failed image processing has no model decision to compare with a ground truth.
        if prediction.get("status") != "ok":
            continue
        image_name = Path(prediction["image_path"]).name
        label = labels_by_name.get(image_name)
        if label is None:
            continue

        decision = json.loads(prediction.get("agent_decision_json") or "{}") or {}
        probabilities = json.loads(prediction.get("probabilities_json") or "{}") or {}
        # Confidence is the score for the predicted class, not the largest score by assumption.
        confidence = decision.get("confidence")
        if confidence is None:
            confidence = probabilities.get(prediction.get("ripeness", ""), "")

        normalized.append({
            "image_name": image_name,
            "image_path": prediction["image_path"],
            "fruit": prediction.get("fruit", ""),
            "true_ripeness": label.get("ground_truth", ""),
            "predicted_ripeness": prediction.get("ripeness", ""),
            "confidence": str(confidence),
            "agent_action": decision.get("action", ""),
            "group_id": label.get("group_id", ""),
            "split": label.get("split", ""),
        })

    if not normalized:
        raise ValueError("No verified predictions matched labels by image_name.")
    return normalized


def load_rows(path: Path, labels_path: Path | None = None) -> list[dict[str, str]]:
    """Load a labeled snapshot or join a prediction export to verified labels."""
    rows = read_csv(path)
    if rows and LABELED_FIELDS.issubset(rows[0]):
        return rows
    if not rows or not PREDICTION_FIELDS.issubset(rows[0]) or labels_path is None:
        raise ValueError(
            "Provide a labeled evaluation CSV, or a predict.py result CSV "
            "together with --labels-csv."
        )
    return normalize_predictions(rows, labels_path)


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "N/A"


def confidence_text(value: str) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def evaluate(rows: list[dict[str, str]]) -> str:
    """모델 정확도와 에이전트의 자동 분류/검토 분기 결과를 따로 요약합니다."""
    correct = sum(row["true_ripeness"] == row["predicted_ripeness"] for row in rows)
    actions = Counter(row["agent_action"] for row in rows)
    auto = [row for row in rows if row["agent_action"] == "auto_classify"]
    # Coverage measures how often automation is proposed; its error rate measures risk within that set.
    auto_errors = [row for row in auto if row["true_ripeness"] != row["predicted_ripeness"]]
    # 비자동 라우팅은 검토 대상이라는 뜻일 뿐, 사람이 실제 수정했다는 뜻은 아닙니다.
    routed_errors = [
        row for row in rows
        if row["agent_action"] != "auto_classify"
        and row["true_ripeness"] != row["predicted_ripeness"]
    ]
    splits = Counter(row["split"] for row in rows)
    lines = [
        "# 재현 가능한 에이전트 기준선 평가",
        "",
        f"- 평가 이미지: {len(rows)}장 / 분할: {dict(splits)}",
        f"- 모델 예측 정확도: {correct}/{len(rows)} ({ratio(correct, len(rows))})",
        f"- 행동 분포: {dict(actions)}",
        f"- 자동 분류 커버리지: {len(auto)}/{len(rows)} ({ratio(len(auto), len(rows))})",
        f"- 자동 분류 오분류: {len(auto_errors)}/{len(auto)} ({ratio(len(auto_errors), len(auto))})",
        f"- 비자동 행동으로 라우팅된 모델 오분류: {len(routed_errors)}건",
        "",
        "## 과일·숙도별 모델 지표",
        "",
        "| 과일 | 숙도 | 표본 수 | 정밀도 | 재현율 |",
        "|---|---|---:|---:|---:|",
    ]
    for fruit in sorted({row["fruit"] for row in rows}):
        fruit_rows = [row for row in rows if row["fruit"] == fruit]
        for label in LABELS:
            tp = sum(
                row["true_ripeness"] == label
                and row["predicted_ripeness"] == label
                for row in fruit_rows
            )
            support = sum(row["true_ripeness"] == label for row in fruit_rows)
            predicted = sum(row["predicted_ripeness"] == label for row in fruit_rows)
            lines.append(
                f"| {fruit} | {label} | {support} | "
                f"{ratio(tp, predicted)} | {ratio(tp, support)} |"
            )

    lines.extend([
        "",
        "## 혼동 행렬 (실제 행 -> 예측 열)",
        "",
        "| 과일 | 실제 숙도 | 미숙 예측 | 적숙 예측 | 과숙 예측 |",
        "|---|---|---:|---:|---:|",
    ])
    for fruit in sorted({row["fruit"] for row in rows}):
        fruit_rows = [row for row in rows if row["fruit"] == fruit]
        for actual in LABELS:
            counts = [
                sum(
                    row["true_ripeness"] == actual
                    and row["predicted_ripeness"] == predicted
                    for row in fruit_rows
                )
                for predicted in LABELS
            ]
            lines.append(
                f"| {fruit} | {actual} | "
                f"{counts[0]} | {counts[1]} | {counts[2]} |"
            )

    lines.extend([
        "",
        "## 자동 분류 오분류 이미지",
        "",
        "| 파일 | 실제 | 예측 | 확률 | 과일 개체 그룹 |",
        "|---|---|---|---:|---|",
    ])
    for row in auto_errors:
        lines.append(
            f"| {row['image_name']} | {row['true_ripeness']} | {row['predicted_ripeness']} "
            f"| {confidence_text(row['confidence'])} | {row['group_id']} |"
        )
    lines.extend([
        "",
        "## 비자동 행동으로 라우팅된 오분류",
        "",
        "| 파일 | 실제 | 예측 | 행동 | 확률 |",
        "|---|---|---|---|---:|",
    ])
    for row in routed_errors:
        lines.append(
            f"| {row['image_name']} | {row['true_ripeness']} | {row['predicted_ripeness']} "
            f"| {row['agent_action']} | {confidence_text(row['confidence'])} |"
        )
    lines.extend([
        "",
        "## 표본 독립성",
        "",
        "| 과일 | 이미지 수 | 서로 다른 개체 그룹 수 |",
        "|---|---:|---:|",
    ])
    for fruit in sorted({row["fruit"] for row in rows}):
        fruit_rows = [row for row in rows if row["fruit"] == fruit]
        group_count = len({row["group_id"] for row in fruit_rows})
        lines.append(f"| {fruit} | {len(fruit_rows)} | {group_count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--labels-csv", type=Path,
        help="verified labels for a predict.py --results-csv export",
    )
    parser.add_argument(
        "--output", type=Path, help="optional UTF-8 Markdown report path"
    )
    args = parser.parse_args()
    report = evaluate(load_rows(args.csv_path, args.labels_csv))
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report saved: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
