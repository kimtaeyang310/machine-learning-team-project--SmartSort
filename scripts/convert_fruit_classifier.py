"""Export the trained sklearn head without retraining or changing rejection thresholds.

Run with the environment that trained the input sklearn estimator. The resulting
artifact uses a NumPy softmax head and can be read in the project's teamTC env.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from src.fruit_classifier import MODEL_PATH, LinearSoftmaxClassifier


def convert(model_path=MODEL_PATH):
    model_path = Path(model_path)
    original_bytes = model_path.read_bytes()
    source_sha = hashlib.sha256(original_bytes).hexdigest()
    bundle = joblib.load(model_path)
    if isinstance(bundle["classifier"], LinearSoftmaxClassifier):
        print("이미 호환 가능한 NumPy 분류기 형식입니다.")
        return
    estimator = bundle["classifier"]
    exported = LinearSoftmaxClassifier.from_estimator(estimator)
    features = np.concatenate([bundle["reference"], np.random.default_rng(42).normal(
        size=(100, exported.n_features_in_))])
    expected = estimator.predict_proba(features)
    actual = exported.predict_proba(features)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    directory = ROOT / "outputs" / "fruit_classifier" / ("compatibility_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True)
    backup = directory / "original_classifier.joblib"
    shutil.copy2(model_path, backup)
    bundle["classifier"] = exported
    bundle["inference_format"] = "numpy_multinomial_softmax_v1"
    bundle["compatibility_migration"] = {"source_model_sha256": source_sha,
                                          "backup": backup.relative_to(ROOT).as_posix()}
    temporary = model_path.with_suffix(".portable.tmp")
    joblib.dump(bundle, temporary, protocol=4)
    restored = joblib.load(temporary)
    np.testing.assert_allclose(restored["classifier"].predict_proba(features), expected, rtol=1e-12, atol=1e-12)
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("변환 중 모델이 변경됐습니다. 기존 모델은 덮어쓰지 않습니다.")
    temporary.replace(model_path)
    report = {"source_model_sha256": source_sha,
              "converted_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "compared_vectors": len(features), "max_probability_difference": float(np.max(np.abs(actual - expected))),
              "weights_unchanged": True, "thresholds_unchanged": True,
              "backup": str(backup), "inference_format": bundle["inference_format"]}
    (directory / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="과일 종류 모델을 버전 독립적인 softmax 형식으로 변환")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    convert(parser.parse_args().model_path)
