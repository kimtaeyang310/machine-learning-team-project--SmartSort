#6. predict : 새 사진에 같은 처리 과정을 적용해 예측

"""폴더의 이미지를 순차 예측하고 처리 시간을 측정.

실행:
    python src/predict.py
또는
    python -m src.predict

학습 때 저장한 전처리 설정과 특징 순서를 그대로 사용합니다.
"""

from pathlib import Path
import sys
import re
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


def predict(image_path, model_path=MODEL_PATH, *, fruit=None):
    """기존 단일 이미지 호출과의 호환성을 유지합니다."""
    bundle = load_model(model_path)
    model, name = select_model(bundle, fruit)
    result = predict_loaded(resolve_image_path(image_path), bundle, model)
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


def predict_folder(folder_path=PREDICT_DIR, model_path=MODEL_PATH):
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
    load_started = perf_counter()
    bundle = load_model(model_path)
    load_ms = (perf_counter() - load_started) * 1000
    print(f"\n폴더: {folder} / 사진: {len(paths)}장")
    print("파일명으로 과일을 구분해 해당 모델을 적용합니다.")
    print(f"모델 로딩: {load_ms:.2f} ms (이미지 처리 시간과 별도)")
    results = []
    batch_started = perf_counter()
    for index, path in enumerate(paths, 1):
        started = perf_counter()
        try:
            fruit = infer_fruit(path)
            model, name = select_model(bundle, fruit)
            result = predict_loaded(path, bundle, model)
            result.update(status="ok", error="", fruit=fruit, model=name)
            result["total_ms"] = (perf_counter() - started) * 1000
        except Exception as error:
            result = {"image_path": str(path), "status": "failed", "error": str(error),
                      "total_ms": (perf_counter() - started) * 1000}
        results.append(result)
        if result["status"] == "ok":
            probability = result["probabilities"]
            confidence = f" / 확률 {probability[result['ripeness']]:.1%}" if probability else ""
            print(f"[{index}/{len(paths)}] {path.name} → {result['fruit']} / {result['model']} / {result['ripeness']}{confidence}"
                  f" | 준비 {result['feature_ms']:.2f} + 예측 {result['prediction_ms']:.2f}"
                  f" | 전체 {result['total_ms']:.2f} ms")
        else:
            print(f"[{index}/{len(paths)}] {path.name} → 실패 ({result['total_ms']:.2f} ms): {result['error']}")
    batch_ms = (perf_counter() - batch_started) * 1000
    successful = [r for r in results if r["status"] == "ok"]
    times = np.asarray([r["total_ms"] for r in successful])
    total_processing_ms = sum(r["total_ms"] for r in results)
    summary = {"count": len(results), "success_count": len(successful),
               "failure_count": len(results) - len(successful), "model_load_ms": load_ms,
               "processing_ms": total_processing_ms, "batch_ms": batch_ms,
               "mean_ms": float(times.mean()) if len(times) else None,
               "p95_ms": float(np.percentile(times, 95)) if len(times) else None,
               "fps": len(successful) * 1000 / total_processing_ms if total_processing_ms else 0.0}
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
    print(f"\n완료: 성공 {summary['success_count']} / 실패 {summary['failure_count']}")
    print(f"전체 순차 실행: {batch_ms / 1000:.3f}초 (결과 출력 포함, 모델 로딩 제외)")
    if len(times):
        print(f"성공 사진 평균: {summary['mean_ms']:.2f} ms / P95: {summary['p95_ms']:.2f} ms")
    print(f"처리 FPS: {summary['fps']:.2f} (성공 장수 / 실패 포함 처리 시간, 결과 출력 제외)")
    goal = "판정 보류: 처리 실패 있음" if summary['failure_count'] else ("달성" if summary['fps'] >= 25 else "미달성")
    print(f"25 FPS 목표: {goal}")
    print("첫 이미지 포함 측정이며 워밍업은 하지 않습니다. 준비 시간은 이미지 읽기·전처리·영역 분리·특징 구성을 포함합니다.")
    return results, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="폴더 전체 숙도 예측 및 시간 측정")
    parser.add_argument("folder", nargs="?", default=str(PREDICT_DIR), help="기본: data/predict_list")
    args = parser.parse_args()
    try:
        predict_folder(args.folder)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n예측 중단:\n{error}")
        raise SystemExit(1)
