#4. train.py : 추출한 특징으로 모델 학습

"""SmartSort: 이미지 특징으로 숙도 분류 모델 학습·비교·저장.

실행 방법: 프로젝트 최상위 폴더에서
    python -m src.train

feature_extraction.py 연결 규격:
    extract_features(preprocessed) -> dict[str, float]

예시 반환값:
    {
        "h_mean": 30.5,
        "s_mean": 120.0,
        "v_mean": 180.0,
        "green_ratio": 0.2,
        "yellow_ratio": 0.7,
        "brown_ratio": 0.1,
    }

히스토그램처럼 여러 숫자가 나오는 특징은
h_hist_0, h_hist_1처럼 각각 별도의 항목으로 반환하세요.
"""

from pathlib import Path

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset import (
    RIPENESS_LABELS,
    load_dataset,
    print_summary,
    split_dataset,
)
from src.preprocessing import preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"

SEED = 42
TEST_SIZE = 0.2
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
    # 특징 추출 파일이 완성된 뒤 이 함수와 연결됩니다.
    from src.feature_extraction import extract_features

    rows = []
    expected_names = (
        list(feature_names) if feature_names is not None else None
    )

    for index, sample in enumerate(samples, start=1):
        try:
            processed = preprocess(
                sample["path"],
                **PREPROCESS_CONFIG,
            )
            features = extract_features(processed)

            if not isinstance(features, dict) or not features:
                raise ValueError(
                    "extract_features()는 비어 있지 않은 "
                    "{특징 이름: 숫자} 딕셔너리를 반환해야 합니다."
                )

            if not all(
                isinstance(key, str) for key in features
            ):
                raise ValueError("특징 이름은 문자열이어야 합니다.")

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
            # 실패 이미지를 조용히 제외하면 평가 대상이 바뀌므로 중단합니다.
            raise RuntimeError(
                f"특징 추출 실패: {sample['image_path']}\n{error}"
            ) from error

        if index % 100 == 0 or index == len(samples):
            print(f"특징 추출: {index}/{len(samples)}장")

    if not rows:
        raise ValueError("특징을 추출할 이미지가 없습니다.")

    return pd.DataFrame(rows, columns=expected_names)


def get_models():
    """비교할 기본 모델을 만듭니다."""
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
    }


def make_cv_splits(X, y, groups):
    """같은 개체/원본을 분리하지 않는 교차검증을 구성합니다."""
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
        # 그룹 제약 때문에 모든 클래스가 각 구간에 들어간다는
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
        "group_id",
        "ripeness",
        "ripeness_label",
    ]

    pd.DataFrame(
        [{key: sample[key] for key in columns} for sample in samples]
    ).to_csv(
        OUTPUT_DIR / f"{split_name}_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )


def train():
    """데이터 준비부터 모델 저장까지 실행합니다."""
    samples = load_dataset()

    # 기존 dataset.py의 과일별 그룹 분할을 그대로 사용합니다.
    train_samples, test_samples = split_dataset(
        samples,
        test_size=TEST_SIZE,
        seed=SEED,
    )

    print_summary(train_samples, "학습")
    print_summary(test_samples, "최종 테스트")

    train_groups = {s["group_id"] for s in train_samples}
    test_groups = {s["group_id"] for s in test_samples}

    if train_groups & test_groups:
        raise ValueError("학습·테스트에 같은 그룹이 포함되었습니다.")

    y_train = np.asarray(
        [s["ripeness_label"] for s in train_samples],
        dtype=np.int64,
    )
    y_test = np.asarray(
        [s["ripeness_label"] for s in test_samples],
        dtype=np.int64,
    )
    groups = np.asarray(
        [s["group_id"] for s in train_samples]
    )

    check_labels(y_train, "전체 학습 데이터")
    check_labels(y_test, "최종 테스트 데이터")

    print("\n학습 데이터 특징 추출")
    X_train = build_feature_table(train_samples)

    cv_splits = make_cv_splits(X_train, y_train, groups)

    # 항상 숙도 3개 클래스를 동일하게 반영합니다.
    scorer = make_scorer(
        f1_score,
        labels=LABEL_IDS,
        average="macro",
        zero_division=0,
    )

    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\n{name} 교차검증 중...")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scorer,
            n_jobs=1,
            error_score="raise",
        )

        results.append({
            "model": name,
            "cv_macro_f1_mean": float(scores.mean()),
            "cv_macro_f1_std": float(scores.std()),
            **{
                f"fold_{i}_macro_f1": float(score)
                for i, score in enumerate(scores, start=1)
            },
        })

        print(
            f"Macro F1: {scores.mean():.4f} "
            f"(표준편차 {scores.std():.4f})"
        )

    comparison = pd.DataFrame(results).sort_values(
        "cv_macro_f1_mean",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    best_name = comparison.loc[0, "model"]
    best_model = clone(models[best_name])

    # 최종 테스트는 모델 선택에 사용하지 않습니다.
    # 선택한 모델만 학습 데이터 전체로 다시 학습합니다.
    best_model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_split(train_samples, "train")
    save_split(test_samples, "test")

    # 행 순서는 train_samples.csv와 동일합니다.
    X_train.to_csv(
        OUTPUT_DIR / "train_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_path = MODEL_DIR / "ripeness_model.joblib"

    # 스케일러와 분류기를 함께 저장합니다.
    # 예측할 때 feature_names 순서로 특징을 정렬해야 합니다.
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "feature_names": X_train.columns.tolist(),
            "label_mapping": dict(RIPENESS_LABELS),
            "preprocess_config": dict(PREPROCESS_CONFIG),
            "seed": SEED,
            "cv_macro_f1": float(
                comparison.loc[0, "cv_macro_f1_mean"]
            ),
        },
        model_path,
    )

    print("\n모델 비교 결과")
    print(comparison.to_string(index=False))
    print(f"\n선택한 모델: {best_name}")
    print(f"저장 위치: {model_path}")
    print(f"분할·비교 결과: {OUTPUT_DIR}")
    print(
        "\n최종 테스트 성능은 evaluate.py에서 "
        "test_samples.csv의 이미지로 측정하세요."
    )

    return best_model, comparison


if __name__ == "__main__":
    train()