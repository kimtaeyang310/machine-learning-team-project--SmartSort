"""Compare tomato SVM class weighting and train-fold-only image augmentation.

Run from the project root:
    python -m src.compare_tomato_augmentation

The script keeps the saved Test split untouched. It compares three conditions
using identical StratifiedGroupKFold splits, then fits each condition on the
full tomato Train set and reports its one-time Test result.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import load_dataset, split_dataset, RIPENESS_LABELS
from src.feature_extraction import extract_features
from src.preprocessing import preprocess
from src.train import PREPROCESS_CONFIG, SEED, build_feature_table, get_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tomato_augmentation_comparison"
N_SPLITS = 3
CONDITIONS = (
    ("baseline_balanced", True, False),
    ("augmentation_balanced", True, True),
    ("augmentation_unweighted", False, True),
)


def image_path(sample):
    path = (RAW_DIR / sample["image_path"]).resolve()
    path.relative_to(RAW_DIR.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def augment_image(image, rng):
    """Apply mild geometry and illumination changes; preserve tomato hue."""
    height, width = image.shape[:2]
    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.92, 1.08))
    tx = float(rng.uniform(-0.04, 0.04) * width)
    ty = float(rng.uniform(-0.04, 0.04) * height)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    result = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)
    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-8.0, 8.0))
    return cv2.convertScaleAbs(result, alpha=alpha, beta=beta)


def features_for_path(path):
    processed = preprocess(path, **PREPROCESS_CONFIG, save_failed=False, include_preview=False)
    return extract_features(processed)


def augmented_training_features(samples, directory, seed, base_features=None):
    """Balance only the supplied training subset; no validation/test rows enter."""
    counts = Counter(sample["ripeness_label"] for sample in samples)
    if set(counts) != set(RIPENESS_LABELS.values()):
        raise ValueError(f"증강할 Train fold에 세 숙도가 모두 있어야 합니다: {counts}")
    target = max(counts.values())
    rng = np.random.default_rng(seed)
    if base_features is None:
        features = [features_for_path(image_path(sample)) for sample in samples]
        base_features = pd.DataFrame(features)
    result = base_features.reset_index(drop=True).copy()
    labels = [int(sample["ripeness_label"]) for sample in samples]

    by_label = {
        label: [sample for sample in samples if sample["ripeness_label"] == label]
        for label in counts
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for label, class_samples in by_label.items():
        needed = target - len(class_samples)
        for index in range(needed):
            sample = class_samples[int(rng.integers(len(class_samples)))]
            image = cv2.imread(str(image_path(sample)), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"이미지를 읽을 수 없습니다: {sample['image_path']}")
            generated = augment_image(image, rng)
            generated_path = directory / f"class{label}_aug{index:04d}.png"
            if not cv2.imwrite(str(generated_path), generated):
                raise OSError(f"증강 이미지를 저장할 수 없습니다: {generated_path}")
            generated_features = features_for_path(generated_path)
            result.loc[len(result)] = generated_features
            labels.append(int(label))

    return result, np.asarray(labels, dtype=np.int64), counts


def tomato_splits(samples):
    y = np.asarray([sample["ripeness_label"] for sample in samples], dtype=np.int64)
    groups = np.asarray([sample["group_id"] for sample in samples])
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(samples)), y, groups))
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError(f"{fold}번 fold에서 group_id가 겹칩니다.")
        if set(y[train_idx]) != set(RIPENESS_LABELS.values()) or set(y[valid_idx]) != set(RIPENESS_LABELS.values()):
            raise ValueError(f"{fold}번 fold에 세 숙도가 모두 포함되지 않습니다. 그룹 수를 확인하세요.")
    return y, groups, splits


def set_class_weight(model, balanced):
    configured = clone(model)
    configured.set_params(classifier__class_weight="balanced" if balanced else None)
    return configured


def main():
    all_samples = load_dataset()
    train_all, test_all = split_dataset(all_samples)
    train = [sample for sample in train_all if sample["fruit"] == "tomato"]
    test = [sample for sample in test_all if sample["fruit"] == "tomato"]
    if not train or not test:
        raise ValueError("토마토 Train/Test 데이터가 모두 필요합니다.")

    y, groups, splits = tomato_splits(train)
    print(f"토마토 Train {len(train)}장 / {len(set(groups))}그룹, Test {len(test)}장")
    print("Train 클래스 수:", dict(Counter(y)))
    print("Test 클래스 수:", dict(Counter(s["ripeness_label"] for s in test)))
    print("모든 증강은 각 CV fold의 학습 부분에만 생성됩니다. Test는 원본 그대로 둡니다.")

    X = build_feature_table(train)
    X_test = build_feature_table(test, feature_names=X.columns)
    y_test = np.asarray([sample["ripeness_label"] for sample in test], dtype=np.int64)
    template = get_models()["svm"]
    rows = []
    test_reports = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix=".tmp_tomato_aug_", dir=output_dir) as temporary:
        temporary = Path(temporary)
        for condition, balanced, use_augmentation in CONDITIONS:
            fold_scores = []
            for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
                if use_augmentation:
                    X_fit, y_fit, raw_counts = augmented_training_features(
                        [train[i] for i in fit_idx],
                        temporary / condition / f"fold_{fold}",
                        seed=SEED + fold,
                        base_features=X.iloc[fit_idx],
                    )
                else:
                    X_fit = X.iloc[fit_idx].reset_index(drop=True)
                    y_fit = y[fit_idx]
                    raw_counts = Counter(y_fit)
                model = set_class_weight(template, balanced)
                model.fit(X_fit, y_fit)
                prediction = model.predict(X.iloc[valid_idx])
                score = f1_score(y[valid_idx], prediction, labels=[0, 1, 2],
                                 average="macro", zero_division=0)
                fold_scores.append(float(score))
                rows.append({
                    "condition": condition,
                    "fold": fold,
                    "train_count_before_augmentation": len(fit_idx),
                    "train_class_counts_before_augmentation": str(dict(raw_counts)),
                    "validation_count": len(valid_idx),
                    "macro_f1": float(score),
                })
                print(f"{condition} fold {fold}: Macro F1={score:.4f}")

            if use_augmentation:
                X_fit, y_fit, raw_counts = augmented_training_features(
                    train, temporary / condition / "full_train", seed=SEED,
                    base_features=X,
                )
            else:
                X_fit, y_fit = X.reset_index(drop=True), y
                raw_counts = Counter(y)
            final_model = set_class_weight(template, balanced).fit(X_fit, y_fit)
            prediction = final_model.predict(X_test)
            test_macro_f1 = f1_score(y_test, prediction, labels=[0, 1, 2],
                                     average="macro", zero_division=0)
            test_reports.append({
                "condition": condition,
                "test_count": len(test),
                "test_accuracy": float(accuracy_score(y_test, prediction)),
                "test_macro_f1": float(test_macro_f1),
                "class_report": classification_report(
                    y_test, prediction, labels=[0, 1, 2],
                    target_names=list(RIPENESS_LABELS), output_dict=True,
                    zero_division=0,
                ),
            })
            print(f"{condition} untouched Test: Macro F1={test_macro_f1:.4f}")

    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(output_dir / "fold_scores.csv", index=False, encoding="utf-8-sig")
    summary = fold_frame.groupby("condition", sort=False).macro_f1.agg(["mean", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "cv_macro_f1_mean", "std": "cv_macro_f1_std"})
    summary["test_count"] = len(test)
    test_metrics = pd.DataFrame([
        {key: report[key] for key in ("condition", "test_count", "test_accuracy", "test_macro_f1")}
        for report in test_reports
    ])
    summary = summary.merge(test_metrics, on=["condition", "test_count"], how="left")
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.json_normalize(test_reports, sep=".").to_json(
        output_dir / "test_class_reports.json", orient="records", force_ascii=False, indent=2,
    )
    print("\n비교 요약")
    print(summary.to_string(index=False))
    print(f"\n결과 저장: {output_dir}")
    print("Test 표본이 적으므로 CV 일관성과 숙도별 결과를 함께 해석하세요.")
    return summary


if __name__ == "__main__":
    main()
