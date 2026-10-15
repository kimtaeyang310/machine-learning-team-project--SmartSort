"""SmartSort: 이미지 특징으로 숙도 분류 모델 학습·비교·저장.

실행 방법: 프로젝트 최상위 폴더에서
    python -m src.train
또는 파일 직접 실행:
    python src/train.py

CSV의 수동 split을 유지하며 Train 내부에서만 모델을 비교합니다.
Test가 없어도 학습할 수 있습니다. 최종 평가는 evaluate.py의 역할입니다.
완성된 extract_features()가 반환하는 숫자 특징만 학습에 사용합니다.
"""

from pathlib import Path
import sys

# 직접 실행할 때도 src 패키지를 찾도록 프로젝트 경로를 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.dataset import (
    RIPENESS_LABELS,
    load_dataset,
    print_summary,
    split_dataset,
)
from src.preprocessing import preprocess
from src.feature_cache import FeatureCache
from src.performance_history import record_training, sample_id
from src.model_bundle import FRUITS, MODEL_PATH, TRAINING_DIR


MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = TRAINING_DIR

SEED = 42
CV_SPLITS = 3

# predict.py에서도 같은 설정으로 전처리해야 합니다.
PREPROCESS_CONFIG = {
    "size": 320,
    "k": 3,
    "seed": SEED,
    "min_saturation": 40,
    "denoise": True,
}

LABEL_IDS = sorted(RIPENESS_LABELS.values())


def check_labels(y, name):
    """숙도 3개 클래스가 모두 있는지 확인합니다."""
    missing = set(LABEL_IDS) - set(np.unique(y))

    if missing:
        names = [
            name
            for name, label in RIPENESS_LABELS.items()
            if label in missing
        ]

        raise ValueError(
            f"{name}: 없는 숙도 클래스 = {names}. "
            "클래스별 개체/원본 수와 그룹 분할을 확인하세요."
        )


def build_feature_table(samples, feature_names=None):
    """이미지마다 특징을 추출하고 동일한 열 순서로 정렬합니다."""
    from src.feature_extraction import extract_features

    rows = []
    cache = FeatureCache(PREPROCESS_CONFIG)

    expected_names = (
        list(feature_names) if feature_names is not None else None
    )

    for index, sample in enumerate(samples, start=1):
        try:
            # 1. 이미지 전처리
            features = cache.get(sample["path"], preprocess, extract_features)

            # 3. 반환 형식 검사
            if not isinstance(features, dict) or not features:
                raise ValueError(
                    "extract_features()는 비어 있지 않은 "
                    "{특징 이름: 숫자} 딕셔너리를 반환해야 합니다."
                )

            if not all(
                isinstance(key, str) for key in features
            ):
                raise ValueError("특징 이름은 문자열이어야 합니다.")

            # 첫 이미지의 특징 이름을 기준으로 순서를 고정합니다.
            if expected_names is None:
                expected_names = sorted(features)

            if set(features) != set(expected_names):
                raise ValueError(
                    "이미지마다 특징 이름이나 개수가 다릅니다."
                )

            values = np.asarray(
                [features[name] for name in expected_names],
                dtype=np.float64,
            )

            if values.shape != (len(expected_names),):
                raise ValueError(
                    "각 특징은 배열이 아닌 숫자 하나여야 합니다."
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    "특징에 NaN 또는 무한대가 포함되어 있습니다."
                )

            rows.append(values)

        except Exception as error:
            # 실패 이미지를 임의로 제외하지 않고 원인을 표시합니다.
            raise RuntimeError(
                f"특징 추출 실패: {sample['image_path']}\n{error}"
            ) from error

        if index % 100 == 0 or index == len(samples):
            print(f"특징 추출: {index}/{len(samples)}장")

    if not rows:
        raise ValueError("특징을 추출할 이미지가 없습니다.")

    cache.report()
    return pd.DataFrame(rows, columns=expected_names)


def get_models():
    """동일한 그룹 교차검증으로 비교할 숙도 분류 모델 5개를 만듭니다.

    KNN과 DT 설정은 초기 비교용이며 최적화된 값이 아닙니다.
    스케일러는 Pipeline 안에서 각 fold의 학습 데이터에만 맞춥니다.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),

        "svm": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]),

        "random_forest": Pipeline([
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="uniform",
                ),
            ),
        ]),

        "decision_tree": Pipeline([
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),
    }


def make_cv_splits(X, y, groups):
    """같은 group_id를 분리하지 않는 교차검증을 구성합니다."""
    if len(np.unique(groups)) < CV_SPLITS:
        raise ValueError(
            f"교차검증에는 학습 그룹이 {CV_SPLITS}개 이상 필요합니다."
        )

    cv = StratifiedGroupKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    splits = list(cv.split(X, y, groups))

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        # 그룹을 유지하면 모든 클래스가 각 구간에 들어간다는
        # 보장이 없으므로 실제 분할 결과를 검사합니다.
        check_labels(y[train_idx], f"{fold}번 교차검증 학습")
        check_labels(y[valid_idx], f"{fold}번 교차검증 검증")

        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise ValueError("학습·검증 그룹이 중복되었습니다.")

    return splits


def save_split(samples, split_name):
    """나중에 동일한 데이터로 평가하도록 분할 목록을 저장합니다."""
    columns = [
        "image_path",
        "fruit",
        "fruit_id",
        "date",
        "source",
        "group_id",
        "split",
        "ripeness",
        "ripeness_label",
    ]

    # samples가 비어 있어도 열 이름은 저장합니다.
    pd.DataFrame(
        [
            {key: sample[key] for key in columns}
            for sample in samples
        ],
        columns=columns,
    ).to_csv(
        OUTPUT_DIR / f"{split_name}_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )


def fit_fruit_models(X_train, train_samples):
    """과일별 데이터만으로 5개 모델을 비교합니다. Test는 받지 않습니다."""
    fruits = np.asarray([s["fruit"] for s in train_samples])
    if set(fruits) != set(FRUITS):
        raise ValueError(f"학습 데이터에 필요한 과일: {FRUITS}; 현재: {sorted(set(fruits))}")
    prepared = {}
    # 모든 과일의 분할을 먼저 검사하고, 실패하면 저장 전에 중단합니다.
    for fruit in FRUITS:
        indices = np.flatnonzero(fruits == fruit)
        samples = [train_samples[i] for i in indices]
        X = X_train.iloc[indices].reset_index(drop=True)
        y = np.asarray([s["ripeness_label"] for s in samples], dtype=np.int64)
        groups = np.asarray([s["group_id"] for s in samples])
        check_labels(y, fruit)
        try:
            splits = make_cv_splits(X, y, groups)
        except ValueError as error:
            raise ValueError(f"{fruit} 교차검증 구성 실패: {error}") from error
        if any(len(a) < 5 for a, _ in splits):
            raise ValueError(f"{fruit}: KNN 비교에는 각 fold의 학습 사진이 5장 이상 필요합니다.")
        prepared[fruit] = (X, y, samples, splits)

    scorer = make_scorer(f1_score, labels=LABEL_IDS, average="macro", zero_division=0)
    bundles, comparisons = {}, {}
    for fruit, (X, y, samples, splits) in prepared.items():
        models = get_models()
        rows = []
        for name, model in models.items():
            print(f"\n[{fruit}] {name} 교차검증 중...", flush=True)
            scores = cross_val_score(model, X, y, cv=splits, scoring=scorer,
                                     n_jobs=1, error_score="raise")
            rows.append({
                "model": name,
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std()),
                **{f"fold_{i}_macro_f1": float(v) for i, v in enumerate(scores, 1)},
            })
        comparison = pd.DataFrame(rows).sort_values(
            "cv_macro_f1_mean", ascending=False, kind="stable"
        ).reset_index(drop=True)
        name = comparison.loc[0, "model"]
        best = clone(models[name]).fit(X, y)
        bundles[fruit] = {
            "model": best, "model_name": name, "fruit": fruit,
            "train_count": len(samples),
            "cv_macro_f1": float(comparison.loc[0, "cv_macro_f1_mean"]),
        }
        comparisons[fruit] = comparison
        print(f"\n[{fruit}] 모델 비교\n{comparison.to_string(index=False)}")
        print(f"[{fruit}] 선택한 모델: {name}", flush=True)
    return bundles, comparisons


def train():
    """과일별 모델을 학습하고 하나의 모델 묶음으로 저장합니다."""
    samples = load_dataset()
    train_samples, test_samples = split_dataset(samples)
    print_summary(train_samples, "학습")
    if test_samples:
        print_summary(test_samples, "최종 테스트")
    if {s["group_id"] for s in train_samples} & {s["group_id"] for s in test_samples}:
        raise ValueError("학습·테스트에 같은 그룹이 포함되었습니다.")
    if {s["fruit"] for s in test_samples} - set(FRUITS):
        raise ValueError("Test에 지원하지 않는 과일이 있습니다.")

    X_train = build_feature_table(train_samples)
    models, comparisons = fit_fruit_models(X_train, train_samples)
    bundle = {
        "schema_version": 2,
        "models": models,
        "feature_names": X_train.columns.tolist(),
        "label_mapping": dict(RIPENESS_LABELS),
        "preprocess_config": dict(PREPROCESS_CONFIG),
        "seed": SEED,
        "cv_grouping": "dataset.group_id (직접 촬영: 개체+날짜)",
        "train_count": len(train_samples), "test_count": len(test_samples),
        "train_dataset_id": sample_id(pd.DataFrame(train_samples)),
        "test_dataset_id": sample_id(pd.DataFrame(test_samples)) if test_samples else None,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_split(train_samples, "train")
    save_split(test_samples, "test")
    X_train.to_csv(OUTPUT_DIR / "train_features.csv", index=False, encoding="utf-8-sig")
    combined = []
    for fruit, comparison in comparisons.items():
        fruit_samples = [s for s in train_samples if s["fruit"] == fruit]
        root = OUTPUT_DIR / fruit
        root.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(root / "model_comparison.csv", index=False, encoding="utf-8-sig")
        record_training(comparison, fruit_samples, root)
        combined.append(comparison.assign(fruit=fruit, selected=comparison["model"].eq(models[fruit]["model_name"])))
    comparison = pd.concat(combined, ignore_index=True)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    # 기존 통합 모델은 보존하고 과일별 모델 묶음을 별도 경로에 저장합니다.
    temporary = MODEL_PATH.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(MODEL_PATH)
    print(f"\n과일별 모델 저장: {MODEL_PATH}")
    print(f"비교 결과: {OUTPUT_DIR}")
    print("최종 Test 평가는 python -m src.evaluate 로 실행하세요." if test_samples
          else "Test 0장: 현재 점수는 학습 내부 교차검증 결과입니다.")
    return models, comparison

# main.py 와 연결 

def main():
    """학습 프로그램의 시작점"""
    train()


if __name__ == "__main__":
    main()
