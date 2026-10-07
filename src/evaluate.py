"""저장한 숙도 모델의 전체·과일별 Test 성능 평가.
실행: python src/evaluate.py 또는 python -m src.evaluate
"""

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

from src.preprocessing import preprocess
from src.feature_extraction import extract_features

MODEL_PATH = PROJECT_ROOT / "models" / "ripeness_model.joblib"
TRAIN_CSV = PROJECT_ROOT / "outputs" / "training" / "train_samples.csv"
TEST_CSV = PROJECT_ROOT / "outputs" / "training" / "test_samples.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def save_csv(table, path, index=False):
    table.to_csv(path, index=index, encoding="utf-8-sig")


def load_inputs(model_path, train_csv, test_csv):
    """학습 당시 저장한 모델과 분할 목록을 검사합니다."""
    for path in (model_path, train_csv, test_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"필요한 파일이 없습니다: {path}\ntrain.py를 먼저 완료하세요."
            )
    bundle = joblib.load(model_path)
    required = {"model", "feature_names", "label_mapping", "preprocess_config"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("train.py에서 저장한 모델 형식과 다릅니다.")

    train = pd.read_csv(train_csv, encoding="utf-8-sig", keep_default_na=False)
    test = pd.read_csv(test_csv, encoding="utf-8-sig", keep_default_na=False)
    columns = {"image_path", "fruit", "group_id", "split", "ripeness", "ripeness_label"}
    for name, frame in (("train", train), ("test", test)):
        if not columns.issubset(frame.columns):
            raise ValueError(f"{name}_samples.csv에 필수 열이 없습니다.")
        if frame.empty:
            raise ValueError(f"{name}_samples.csv에 이미지가 없습니다.")
        if not frame["split"].eq(name).all():
            raise ValueError(f"{name} 목록에 다른 split이 섞였습니다.")
        if frame["image_path"].duplicated().any():
            raise ValueError(f"{name} 목록에 중복 이미지가 있습니다.")
        if frame[["image_path", "fruit", "group_id"]].eq("").any().any():
            raise ValueError(f"{name} 목록에 비어 있는 식별 정보가 있습니다.")
        expected = frame["ripeness"].map(bundle["label_mapping"])
        actual = pd.to_numeric(frame["ripeness_label"], errors="raise")
        if expected.isna().any() or not np.array_equal(expected, actual):
            raise ValueError(f"{name} 목록의 숙도 이름과 숫자 라벨이 다릅니다.")
        frame["ripeness_label"] = actual.astype(int)
        count = bundle.get(f"{name}_count")
        if count is not None and count != len(frame):
            raise ValueError("모델과 분할 목록의 사진 수가 다릅니다. 같은 학습 실행의 파일을 사용하세요.")

    for column in ("group_id", "image_path"):
        if set(train[column]) & set(test[column]):
            raise ValueError(f"Train/Test 사이에 {column} 중복이 있습니다.")
    return bundle, test


def extract_test_features(test, bundle, output_dir):
    """Test 특징을 추출합니다. 실패가 있으면 기록 후 평가를 중단합니다."""
    names = bundle["feature_names"]
    if not names or len(names) != len(set(names)):
        raise ValueError("모델의 특징 이름이 비어 있거나 중복되었습니다.")
    rows, failures = [], []
    raw_root = RAW_DIR.resolve()
    for index, sample in enumerate(test.to_dict("records"), start=1):
        try:
            path = (raw_root / sample["image_path"]).resolve()
            path.relative_to(raw_root)
            processed = preprocess(path, **bundle["preprocess_config"])
            features = extract_features(processed)
            if not isinstance(features, dict) or set(features) != set(names):
                raise ValueError("학습 당시와 특징 이름이 다릅니다.")
            values = np.asarray([features[name] for name in names], dtype=float)
            if values.shape != (len(names),) or not np.isfinite(values).all():
                raise ValueError("특징에 배열, NaN 또는 무한대가 있습니다.")
            rows.append(values)
        except Exception as error:
            failures.append({
                "image_path": sample["image_path"],
                "fruit": sample["fruit"],
                "ripeness": sample["ripeness"],
                "error": str(error),
            })
        if index % 100 == 0 or index == len(test):
            print(f"Test 특징 추출: {index}/{len(test)}장")

    if failures:
        path = output_dir / "failed_samples.csv"
        save_csv(pd.DataFrame(failures), path)
        raise RuntimeError(
            f"Test {len(test)}장 중 {len(failures)}장 처리 실패.\n"
            f"실패 목록: {path}\n"
            "일부 이미지만으로 점수를 계산하지 않았습니다. 원인을 확인하세요."
        )
    return pd.DataFrame(rows, columns=names)


def score_subset(frame, labels, label_names):
    """숙도 3개를 고정한 Macro 지표와 클래스별 결과를 계산합니다."""
    truth = frame["ripeness_label"].to_numpy()
    predicted = frame["predicted_label"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="macro", zero_division=0
    )
    missing = [name for label, name in zip(labels, label_names) if label not in set(truth)]
    metrics = {
        "n_images": len(frame),
        "n_groups": frame["group_id"].nunique(),
        "accuracy": accuracy_score(truth, predicted),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "missing_true_classes": ", ".join(missing),
    }
    report = classification_report(
        truth, predicted, labels=labels, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    matrix = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=labels),
        index=label_names, columns=label_names,
    )
    matrix.index.name = "actual / predicted"
    return metrics, pd.DataFrame(report).T, matrix


def evaluate(
    model_path=MODEL_PATH, train_csv=TRAIN_CSV,
    test_csv=TEST_CSV, output_root=OUTPUT_DIR,
):
    bundle, test = load_inputs(model_path, train_csv, test_csv)
    # 매번 별도 폴더에 저장해 이전 성공 결과와 실패 기록을 구분합니다.
    run_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    save_csv(test, run_dir / "test_samples.csv")
    print(f"평가 모델: {bundle.get('model_name', 'unknown')}")
    print(f"평가 대상: {len(test)}장 / 과일: {', '.join(sorted(test['fruit'].unique()))}")

    X = extract_test_features(test, bundle, run_dir)
    # 저장된 Pipeline이 학습 때의 스케일러를 그대로 적용합니다.
    predictions = np.asarray(bundle["model"].predict(X))
    label_mapping = bundle["label_mapping"]
    labels = sorted(label_mapping.values())
    inverse = {value: name for name, value in label_mapping.items()}
    names = [inverse[label] for label in labels]
    if predictions.shape != (len(test),) or not set(predictions).issubset(labels):
        raise ValueError("모델 예측 결과의 형식 또는 라벨이 올바르지 않습니다.")

    result = test.copy()
    result["predicted_label"] = predictions.astype(int)
    result["predicted_ripeness"] = result["predicted_label"].map(inverse)
    result["correct"] = result["ripeness_label"] == result["predicted_label"]
    save_csv(result, run_dir / "predictions.csv")
    save_csv(result.loc[~result["correct"]], run_dir / "misclassified.csv")
    save_csv(X, run_dir / "test_features.csv")

    summaries = []
    subsets = [("overall", result)] + list(result.groupby("fruit", sort=True))
    for number, (scope, frame) in enumerate(subsets):
        metrics, report, matrix = score_subset(frame, labels, names)
        summaries.append({"scope": scope, **metrics})
        folder = run_dir / f"subset_{number}"
        folder.mkdir()
        save_csv(report, folder / "classification_report.csv", index=True)
        save_csv(matrix, folder / "confusion_matrix.csv", index=True)
        print(f"\n[{scope}] Accuracy={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1']:.4f}")
        print(matrix.to_string())
        if metrics["missing_true_classes"]:
            print(f"Test에 없는 숙도: {metrics['missing_true_classes']}")
        print(f"상세 결과: {folder}")

    summary = pd.DataFrame(summaries)
    save_csv(summary, run_dir / "metrics.csv")
    print(f"\n평가 완료: {run_dir}")
    return summary, run_dir


if __name__ == "__main__":
    try:
        evaluate()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n평가 중단:\n{error}")
        raise SystemExit(1)
