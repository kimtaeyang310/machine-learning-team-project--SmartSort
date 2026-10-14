"""저장 모델을 바꾸지 않고 RF 병렬 설정·K-means 속도·Test F1을 비교합니다.

실행: python -m src.benchmark_prediction
실험 결과는 outputs/prediction_benchmark.json에 별도 저장합니다.
Test는 회귀 확인용이며, 새 촬영일·조명에 대한 독립 성능 증명은 아닙니다.
"""
import contextlib
import io
import json
from time import perf_counter

import cv2
import numpy as np
from sklearn.metrics import f1_score

from src.evaluate import load_inputs, TRAIN_CSV, TEST_CSV, RAW_DIR
from src.model_bundle import MODEL_PATH, FRUITS, select_model
from src.predict import (
    PREDICT_DIR, PROJECT_ROOT, IMAGE_EXTENSIONS, infer_fruit,
    extract_image_features, predict_loaded, configure_prediction_jobs,
)


def benchmark():
    bundle, test = load_inputs(MODEL_PATH, TRAIN_CSV, TEST_CSV)
    report = {"opencv_threads": cv2.getNumThreads(), "test_count": len(test),
              "rf_jobs": {}, "folder": {}, "test": {}}
    paths = sorted(p for p in PREDICT_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError("속도를 비교할 예측 이미지가 없습니다.")
    cached = [(infer_fruit(p), extract_image_features(p, bundle)) for p in paths]
    for fruit in FRUITS:
        model, name = select_model(bundle, fruit)
        if name != "random_forest":
            continue
        reference = None
        report["rf_jobs"][fruit] = {}
        for jobs in (1, 2, 4, -1):
            configure_prediction_jobs(model, jobs)
            elapsed, probabilities = [], []
            for _ in range(3):
                for kind, features in cached:
                    if kind != fruit:
                        continue
                    start = perf_counter()
                    probabilities.append(model.predict_proba(features)[0])
                    elapsed.append((perf_counter() - start) * 1000)
            if reference is None:
                reference = probabilities
            np.testing.assert_allclose(probabilities, reference, rtol=1e-12, atol=1e-12)
            report["rf_jobs"][fruit][str(jobs)] = {
                "median_ms": float(np.median(elapsed)), "mean_ms": float(np.mean(elapsed))}
        configure_prediction_jobs(model, 1)
    print("RF 비교 완료", report["rf_jobs"], flush=True)

    original_config = dict(bundle["preprocess_config"])
    baseline_features = None
    for variant, attempts, preview in (("3", 3, True), ("1", 1, True),
                                       ("3_no_preview", 3, False)):
        bundle["preprocess_config"] = {**original_config, "kmeans_attempts": attempts,
                                       "include_preview": preview}
        rows = []
        for _ in range(3):
            for path in paths:
                model, _ = select_model(bundle, infer_fruit(path))
                rows.append(predict_loaded(path, bundle, model))
        report["folder"][variant] = {
            "images": len(paths), "repeats": 3,
            **{key: float(np.mean([r[key] for r in rows]))
               for key in ("feature_ms", "prediction_ms", "total_ms")},
            "p95_ms": float(np.percentile([r["total_ms"] for r in rows], 95)),
            "stages_ms": {key: float(np.mean([r["preprocessing_ms"][key] for r in rows]))
                          for key in rows[0]["preprocessing_ms"]},
        }
        predictions, failures, feature_rows = [], [], []
        start = perf_counter()
        for index, sample in enumerate(test.to_dict("records"), 1):
            try:
                path = (RAW_DIR / sample["image_path"]).resolve()
                path.relative_to(RAW_DIR.resolve())
                with contextlib.redirect_stdout(io.StringIO()):
                    features = extract_image_features(path, bundle)
                model, _ = select_model(bundle, sample["fruit"])
                predictions.append(int(model.predict(features)[0]))
                feature_rows.append(features.to_numpy()[0])
            except Exception as error:
                failures.append({"image_path": sample["image_path"], "error": str(error)})
            if index % 100 == 0 or index == len(test):
                print(f"{variant} Test: {index}/{len(test)}", flush=True)
        result = {"failures": failures, "elapsed_s": perf_counter() - start}
        if not failures:
            frame = test.copy()
            frame["predicted_label"] = predictions
            result["macro_f1"] = {
                scope: float(f1_score(part["ripeness_label"], part["predicted_label"],
                                     labels=sorted(bundle["label_mapping"].values()),
                                     average="macro", zero_division=0))
                for scope, part in [("overall", frame)] + list(frame.groupby("fruit"))}
            result["predictions"] = predictions
            if variant == "3":
                baseline_features = np.asarray(feature_rows)
            elif baseline_features is not None:
                result["changed_feature_rows"] = int(np.count_nonzero(
                    np.any(baseline_features != np.asarray(feature_rows), axis=1)))
        report["test"][variant] = result
    output = PROJECT_ROOT / "outputs" / "prediction_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", output, flush=True)
    for attempts, result in report["test"].items():
        print(attempts, result.get("macro_f1"), "실패", len(result["failures"]))
    return report


if __name__ == "__main__":
    benchmark()
