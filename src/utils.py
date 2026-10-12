#7. utils : 여러 파일에서 공통으로 쓰는 작은 기능


"""SmartSort - 여러 파일에서 공통으로 사용하는 유틸리티."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. 저장된 모델 불러오기 및 형식 검사
# ============================================================

def load_model_bundle(model_path):
    """학습된 모델과 예측에 필요한 설정을 불러옵니다."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}\n"
            "먼저 train.py를 실행하세요."
        )

    bundle = joblib.load(model_path)

    required = {
        "model",
        "feature_names",
        "label_mapping",
        "preprocess_config",
    }

    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(
            "저장된 모델 형식이 올바르지 않습니다."
        )

    return bundle


# ============================================================
# 2. 이미지 경로 확인
# ============================================================

def resolve_image_path(image_path, project_root):
    """이미지 경로를 확인하고 절대 경로로 변환합니다."""

    project_root = Path(project_root).resolve()
    path = Path(image_path)

    if not path.is_absolute():
        path = project_root / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {path}"
        )

    return path


# ============================================================
# 3. 특징 데이터 검증 및 순서 정렬
# ============================================================

def validate_features(features, feature_names):
    """특징 이름, 순서, 데이터 형태와 숫자 값을 검증합니다."""

    feature_names = list(feature_names)

    if not feature_names:
        raise ValueError("특징 이름이 비어 있습니다.")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError("특징 이름에 중복이 있습니다.")

    if not isinstance(features, dict) or not features:
        raise ValueError(
            "특징은 비어 있지 않은 딕셔너리여야 합니다."
        )

    if set(features) != set(feature_names):
        raise ValueError(
            "특징 이름이 학습 당시와 다릅니다."
        )

    values = np.asarray(
        [features[name] for name in feature_names],
        dtype=np.float64,
    )

    if values.shape != (len(feature_names),):
        raise ValueError(
            "각 특징은 숫자 하나여야 합니다."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "특징에 NaN 또는 무한대가 포함되어 있습니다."
        )

    return pd.DataFrame(
        [values],
        columns=feature_names,
    )
