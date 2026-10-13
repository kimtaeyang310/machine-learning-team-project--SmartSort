"""과일별 숙도 모델의 공통 경로와 명시적 모델 선택."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRUITS = ("banana", "mandarin", "tomato")
MODEL_PATH = PROJECT_ROOT / "models" / "ripeness_by_fruit.joblib"
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training" / "by_fruit"


def validate_bundle(bundle):
    required = {"feature_names", "label_mapping", "preprocess_config"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("숙도 모델 파일 형식이 올바르지 않습니다.")
    if "models" in bundle:
        models = bundle["models"]
        if not isinstance(models, dict) or set(models) != set(FRUITS):
            raise ValueError("과일별 모델에는 banana, mandarin, tomato가 모두 필요합니다.")
        for fruit, entry in models.items():
            if not isinstance(entry, dict) or not {"model", "model_name", "fruit"}.issubset(entry):
                raise ValueError(f"{fruit} 모델 형식이 올바르지 않습니다.")
            if entry["fruit"] != fruit:
                raise ValueError(f"{fruit} 모델의 과일 정보가 다릅니다.")
    elif "model" not in bundle:
        raise ValueError("저장된 분류기가 없습니다.")
    return bundle


def select_model(bundle, fruit):
    if fruit not in FRUITS:
        raise ValueError(f"과일 종류를 지정하세요: {', '.join(FRUITS)}")
    if "models" in bundle:
        entry = bundle["models"].get(fruit)
        if entry is None:
            raise ValueError(f"{fruit} 모델이 없습니다. 과일별 학습을 먼저 실행하세요.")
        return entry["model"], entry["model_name"]
    return bundle["model"], bundle.get("model_name", "unknown")
