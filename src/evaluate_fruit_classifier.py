"""Re-evaluate frozen fruit identity/rejection policy; never recalibrate on Test."""
from datetime import datetime, timezone
import argparse
import hashlib
from pathlib import Path

from src.fruit_classifier import FruitClassifier, MODEL_PATH, ROOT
from src.train_fruit_classifier import encode_cached, compare_methods, write_reports


def evaluate(model_path=MODEL_PATH):
    detector = FruitClassifier(model_path)
    bundle = detector.bundle
    samples = bundle["samples"]
    for sample in samples:
        path = ROOT / sample["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"학습 당시 자료와 다릅니다. 재학습이 필요합니다: {path}")
    embeddings = encode_cached(detector.encoder, samples)
    probabilities, scores = detector.score(embeddings)
    report = {"version": bundle["version"], "selected_method": bundle["selected_method"], "known_acceptance_target": 0.95,
              "preprocess_mode": bundle.get("preprocess_mode", "center_crop"),
              "thresholds": bundle["thresholds"],
              "model_sha256": detector.sha256,
              "selection": compare_methods(samples, probabilities, scores, bundle["thresholds"],
                                           "selection" if bundle["version"] == 2 else "calibration"),
              "test": compare_methods(samples, probabilities, scores, bundle["thresholds"], "test")}
    if bundle["version"] == 2:
        report["calibration"] = compare_methods(samples, probabilities, scores, bundle["thresholds"], "calibration")
        report["evaluation_status"] = bundle["evaluation_status"]
    directory = ROOT / "outputs" / "fruit_classifier" / ("evaluation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    write_reports(directory, samples, probabilities, scores, bundle["thresholds"], report)
    print(f"보고서: {directory}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    evaluate(parser.parse_args().model_path)
