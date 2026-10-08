#6. predict : 새 사진에 같은 처리 과정을 적용해 예측

"""새로운 이미지 1장의 숙도를 예측.

실행:
    python src/predict.py "data/raw/새사진.jpg"
또는
    python -m src.predict "data/raw/새사진.jpg"

학습 때 저장한 전처리 설정과 특징 순서를 그대로 사용합니다.
"""

from pathlib import Path
import sys


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

from src.preprocessing import preprocess
from src.feature_extraction import extract_features


# ============================================================
# 3. 경로 설정
# ============================================================

MODEL_PATH = PROJECT_ROOT / "models" / "ripeness_model.joblib"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


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

    required = {
        "model",
        "feature_names",
        "label_mapping",
        "preprocess_config",
    }

    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError(
            "저장된 모델 형식이 train.py에서 저장한 형식과 다릅니다."
        )

    return bundle


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

def extract_image_features(image_path, bundle):
    """학습 때와 동일한 전처리와 특징 추출을 수행합니다."""

    feature_names = bundle["feature_names"]
    preprocess_config = bundle["preprocess_config"]

    if not feature_names:
        raise ValueError("저장된 특징 이름이 없습니다.")

    # --------------------------------------------------------
    # 1) 이미지 전처리
    # --------------------------------------------------------

    processed = preprocess(
        image_path,
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

def predict(image_path, model_path=MODEL_PATH):
    """새 이미지 한 장의 숙도를 예측합니다."""

    # --------------------------------------------------------
    # 모델 및 설정 불러오기
    # --------------------------------------------------------

    bundle = load_model(model_path)

    model = bundle["model"]
    label_mapping = bundle["label_mapping"]

    # 숫자 라벨 → 숙도 이름
    inverse_mapping = {
        value: name
        for name, value in label_mapping.items()
    }

    labels = sorted(label_mapping.values())
    label_names = [
        inverse_mapping[label]
        for label in labels
    ]

    # --------------------------------------------------------
    # 이미지 경로 확인
    # --------------------------------------------------------

    image_path = resolve_image_path(image_path)

    # --------------------------------------------------------
    # 특징 추출
    # --------------------------------------------------------

    X = extract_image_features(
        image_path,
        bundle,
    )

    # --------------------------------------------------------
    # 숙도 예측
    # --------------------------------------------------------

    predicted_label = int(
        np.asarray(model.predict(X))[0]
    )

    if predicted_label not in inverse_mapping:
        raise ValueError(
            f"모델이 알 수 없는 라벨을 예측했습니다: "
            f"{predicted_label}"
        )

    predicted_name = inverse_mapping[predicted_label]

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("SmartSort 숙도 예측")
    print("=" * 50)

    print(f"이미지 : {image_path.name}")
    print(f"모델   : {bundle.get('model_name', 'unknown')}")
    print(f"예측   : {predicted_name}")

    # --------------------------------------------------------
    # 예측 확률
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

        # 모델이 실제로 가진 class 순서를 사용합니다.
        if hasattr(model, "classes_"):
            model_classes = model.classes_
        elif hasattr(model, "named_steps"):
            # Pipeline의 마지막 분류기에서 classes_ 가져오기
            classifier = model.named_steps.get("classifier")

            if classifier is not None and hasattr(
                classifier,
                "classes_",
            ):
                model_classes = classifier.classes_
            else:
                model_classes = labels
        else:
            model_classes = labels

        print("\n예측 확률")

        probability_dict = {}

        for class_label, probability in zip(
            model_classes,
            probabilities,
        ):
            class_label = int(class_label)

            if class_label in inverse_mapping:
                name = inverse_mapping[class_label]
                probability_dict[name] = float(probability)

        # 항상 unripe → ripe → overripe 순서로 출력
        for name in label_names:
            probability = probability_dict.get(
                name,
                0.0,
            )

            print(
                f"  {name:<9}: "
                f"{probability * 100:6.2f}%"
            )

    else:
        print(
            "\n이 모델은 predict_proba()를 지원하지 않아 "
            "예측 확률을 표시할 수 없습니다."
        )

    print("=" * 50)

    return predicted_name


# ============================================================
# 8. 프로그램 실행
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "사용법:\n"
            '  python src/predict.py "이미지 경로"\n\n'
            "예:\n"
            '  python src/predict.py "data/raw/banana.jpg"'
        )
        raise SystemExit(1)

    image_path = sys.argv[1]

    try:
        predict(image_path)

    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n예측 중단:\n{error}")
        raise SystemExit(1)