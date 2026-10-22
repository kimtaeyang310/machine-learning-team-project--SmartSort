"""Specimen-disjoint preprocessing experiment with frozen, class-conditional calibration.

No legacy Test image enters candidate selection. Legacy Test is a diagnostic set,
since previous user-driven error analysis already inspected its outcomes.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import shutil

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from src.fruit_classifier import (ROOT, MODEL_PATH, FRUITS, PREPROCESS_MODES, ImageEncoder,
                                 unfamiliarity_scores, rejection_threshold, LinearSoftmaxClassifier)
from src.train_fruit_classifier import prepare_samples, encode_cached, compare_methods, write_reports


def specimen_weights(samples):
    """Equal fruit totals, equal stages within fruit, equal specimens within each stage.

    Ten photos of a specimen do not outweigh ten distinct specimens. These are
    weights on existing photos, not new specimens or synthetic data.
    """
    counts = Counter((s["fruit"], s["ripeness"], s["group"]) for s in samples)
    groups = defaultdict(set)
    stages = defaultdict(set)
    for s in samples:
        groups[s["fruit"], s["ripeness"]].add(s["group"])
        stages[s["fruit"]].add(s["ripeness"])
    weights = np.array([1 / (len(stages[s["fruit"]]) * len(groups[s["fruit"], s["ripeness"]])
                            * counts[s["fruit"], s["ripeness"], s["group"]]) for s in samples])
    return weights / weights.mean()


def calibrate_per_fruit(scores, probabilities, samples):
    predicted = np.array(FRUITS)[probabilities.argmax(axis=1)]
    calibration = np.array([s["phase"] == "calibration" for s in samples])
    truth = np.array([s["fruit"] for s in samples])
    result = {}
    for method, values in scores.items():
        result[method] = {}
        for fruit in FRUITS:
            actual = calibration & (truth == fruit)
            correct = actual & (predicted == fruit)
            count = int(actual.sum())
            required = int(np.ceil(0.95 * count))
            if count < 10 or int(correct.sum()) < required:
                raise ValueError(f"{fruit}: Calibration에서 95%를 종류 정답과 함께 통과시킬 수 없습니다.")
            # Classification mistakes consume the same 5% error budget; never use
            # misclassified tomato photos to relax the mandarin threshold.
            ordered = np.sort(values[correct])
            # float32 cosine dot products can vary slightly with query batch size.
            # Store the numerical guard in the actual threshold used at inference.
            result[method][fruit] = max(rejection_threshold(ordered), float(ordered[required - 1])) + 1e-6
    return result


def candidate_key(row):
    """Only separate Selection results are accepted here, never legacy Test."""
    return (row["unknown_false_acceptance_rate"], row["known_balanced_error_rate"],
            row["tomato_false_rejection_rate"], row["method"] != "embedding",
            PREPROCESS_MODES.index(row["preprocess_mode"]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(model_path=MODEL_PATH):
    directory = ROOT / "outputs" / "fruit_classifier" / ("specimen_v2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True)
    model_path = Path(model_path)
    previous_sha = digest(model_path) if model_path.is_file() else None
    if previous_sha:
        shutil.copy2(model_path, directory / "previous_classifier.joblib")
    samples = prepare_samples()
    dev = [s for s in samples if s["phase"] != "test"]
    legacy_test = [s for s in samples if s["phase"] == "test"]
    train = np.array([s["phase"] == "train" for s in dev])
    calibration = np.array([s["phase"] == "calibration" for s in dev])
    labels = np.array([s["fruit"] for s in dev])
    training_samples = [s for s in dev if s["phase"] == "train"]
    weights = specimen_weights(training_samples)
    print("개체 단위 분할:", Counter((s["fruit"], s["phase"]) for s in samples), flush=True)
    (directory / "split_manifest.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = defaultdict(lambda: {"photos": 0, "training_weight_sum": 0.0})
    for s in samples:
        inventory[s["fruit"], s["phase"], s["group"], s["ripeness"]]["photos"] += 1
    for s, weight in zip(training_samples, weights):
        inventory[s["fruit"], s["phase"], s["group"], s["ripeness"]]["training_weight_sum"] += float(weight)
    coverage = []
    for fruit in FRUITS:
        for stage in ("unripe", "ripe", "overripe"):
            row = {"fruit": fruit, "ripeness": stage}
            for phase in ("train", "calibration", "selection", "test"):
                subset = [s for s in samples if s["fruit"] == fruit and s["ripeness"] == stage and s["phase"] == phase]
                row[phase + "_photos"] = len(subset)
                row[phase + "_specimens"] = len({s["group"] for s in subset})
            coverage.append(row)
    with (directory / "stage_coverage.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
        writer.writeheader(); writer.writerows(coverage)
    with (directory / "specimen_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fruit", "phase", "group", "ripeness", "photos", "training_weight_sum"])
        writer.writeheader()
        for (fruit, phase, group, ripeness), item in sorted(inventory.items()):
            writer.writerow(dict(fruit=fruit, phase=phase, group=group, ripeness=ripeness, **item))
    candidates, best, invalid = [], None, []
    shared_state = None
    for mode in PREPROCESS_MODES:
        print(f"\n전처리 후보: {mode} (Test는 아직 사용하지 않습니다)", flush=True)
        encoder = ImageEncoder(shared_state, mode)
        if shared_state is None:
            shared_state = encoder.state()
        embeddings = encode_cached(encoder, dev)
        classifier = LogisticRegression(C=10, max_iter=3000, random_state=42)
        classifier.fit(embeddings[train], labels[train], sample_weight=weights)
        if tuple(classifier.classes_) != FRUITS:
            raise ValueError("Fruit class order changed")
        probabilities = classifier.predict_proba(embeddings)
        reference = embeddings[train]
        scores = unfamiliarity_scores(probabilities, embeddings, reference)
        try:
            thresholds = calibrate_per_fruit(scores, probabilities, dev)
        except ValueError as error:
            invalid.append({"preprocess_mode": mode, "reason": str(error)})
            print(f"후보 제외: {mode}: {error}", flush=True)
            continue
        global_thresholds = {method: rejection_threshold(values[calibration]) for method, values in scores.items()}
        per_fruit_selection = compare_methods(dev, probabilities, scores, thresholds, "selection")
        global_selection = compare_methods(dev, probabilities, scores, global_thresholds, "selection")
        for scope, metrics in (("per_fruit", per_fruit_selection), ("global_comparison_only", global_selection)):
            for method, m in metrics.items():
                row = {"preprocess_mode": mode, "method": method, "threshold_scope": scope,
                       "known_count": m["known_count"], "known_false_rejection_rate": m["known_false_rejection_rate"],
                       "known_balanced_error_rate": m["known_balanced_error_rate"],
                       "unknown_count": m["unknown_count"], "unknown_false_acceptance_rate": m["unknown_false_acceptance_rate"],
                       "tomato_false_rejection_rate": m["by_fruit"]["tomato"]["false_rejection_rate"]}
                candidates.append(row)
                print("선택용 결과:", row, flush=True)
                if scope == "per_fruit" and (best is None or candidate_key(row) < candidate_key(best["row"])):
                    best = {"row": row, "encoder": encoder, "classifier": classifier,
                            "reference": reference, "thresholds": thresholds, "embeddings": embeddings,
                            "probabilities": probabilities, "scores": scores,
                            "calibration": compare_methods(dev, probabilities, scores, thresholds, "calibration"),
                            "selection": per_fruit_selection}
    (directory / "invalid_candidates.json").write_text(json.dumps(invalid, ensure_ascii=False, indent=2), encoding="utf-8")
    if best is None:
        raise ValueError("과일별 Calibration 95% 기준을 충족하는 후보가 없습니다. 기존 모델을 유지합니다.")
    with (directory / "candidate_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    winner = best["row"]
    print("\nSelection으로 확정한 정책:", winner, flush=True)
    # Policy is frozen before any legacy-Test extraction, scoring or reporting.
    bundle = {"version": 2, "classes": FRUITS, "encoder_state": shared_state,
              "preprocess_mode": winner["preprocess_mode"],
              "classifier": LinearSoftmaxClassifier.from_estimator(best["classifier"]),
              "inference_format": "numpy_multinomial_softmax_v1",
              "reference": best["reference"], "thresholds": best["thresholds"],
              "selected_method": winner["method"], "known_acceptance_target": 0.95,
              "encoder": "torchvision/mobilenet_v3_small/IMAGENET1K_V1/frozen/avgpool576",
              "samples": samples, "group_policy": "fruit_and_specimen_id_banana_original_group",
              "training_weight_policy": "equal_fruit_stage_specimen",
              "run_directory": directory.relative_to(ROOT).as_posix(),
              "selection_rule": "unknown false acceptance, balanced known error, tomato rejection, embedding, simple preprocessing",
              "evaluation_status": "legacy_test_regression_after_prior_error_analysis",
              "software_versions": {"sklearn": sklearn.__version__, "torch": best["encoder"].torch.__version__},
              "additional_independent_specimens": 0,
              "previous_model_sha256": previous_sha}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if previous_sha and digest(model_path) != previous_sha:
        raise RuntimeError("비교 중 저장 모델이 변경됐습니다. 기존 모델을 덮어쓰지 않습니다.")
    temporary = model_path.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(model_path)
    shutil.copy2(model_path, directory / "selected_classifier.joblib")
    (directory / "frozen_policy.json").write_text(json.dumps({"selected": winner, "thresholds": best["thresholds"],
        "model_sha256": digest(model_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
    test_embeddings = encode_cached(best["encoder"], legacy_test)
    test_probabilities = best["classifier"].predict_proba(test_embeddings)
    test_scores = unfamiliarity_scores(test_probabilities, test_embeddings, best["reference"])
    report = {"version": 2, "selected_method": winner["method"], "preprocess_mode": winner["preprocess_mode"],
              "known_acceptance_target": 0.95, "thresholds": best["thresholds"],
              "calibration": best["calibration"], "selection": best["selection"],
              "test": compare_methods(legacy_test, test_probabilities, test_scores, best["thresholds"], "test"),
              "evaluation_status": bundle["evaluation_status"], "model_sha256": digest(model_path),
              "additional_independent_specimens": 0,
              "candidate_selection": "No Test data used; separate known Selection and unknown Selection",
              "split_counts": {f"{a}/{b}": n for (a, b), n in Counter((s['fruit'], s['phase']) for s in samples).items()},
              "specimen_counts": {f"{a}/{b}": len({s['group'] for s in samples if s['fruit'] == a and s['phase'] == b})
                                  for a, b in {(s['fruit'], s['phase']) for s in samples}}}
    ordered = dev + legacy_test
    probabilities = np.concatenate([best["probabilities"], test_probabilities])
    scores = {method: np.r_[best["scores"][method], test_scores[method]] for method in test_scores}
    write_reports(directory, ordered, probabilities, scores, best["thresholds"], report)
    write_experiment_summary(directory, candidates, report)
    print(f"모델: {model_path}\n비교 보고서: {directory}", flush=True)
    return report


def write_experiment_summary(directory, candidates, report):
    lines = ["# 개체 분리·전처리 비교", "",
             "아래 비교는 별도 Selection 자료만 사용했습니다. 임계값은 그와 다른 Calibration에서 계산했습니다.",
             "새 독립 개체를 추가한 수는 0입니다. 기존 자료의 개체·숙도별 가중치를 보완했습니다.", "",
             "| 전처리 | 점수 | 기준 | 알려진 과일 오거부 | 토마토 오거부 | 모르는 과일 오수락 |",
             "|---|---|---|---|---|---|"]
    for row in candidates:
        lines.append(f"| {row['preprocess_mode']} | {row['method']} | {row['threshold_scope']} | "
                     f"{row['known_false_rejection_rate']:.2%} | {row['tomato_false_rejection_rate']:.2%} | "
                     f"{row['unknown_false_acceptance_rate']:.2%} |")
    lines += ["", f"선택: {report['preprocess_mode']} / {report['selected_method']} / 과일별 임계값", "",
              "전체 공통 기준은 비교용으로만 기록했습니다. 배포 후보는 과일별 기준만 포함했습니다.",
              "선택 우선순위는 미지 오수락 → 알려진 과일별 평균(거부 또는 종류 오분류) 오류 → 토마토 오거부입니다.",
              "동률이면 임베딩 거리, 다시 동률이면 center_crop → letterbox → foreground_crop 순입니다.",
              "기존 Test는 앞서 분석에 사용됐으므로 새 독립 최종 성능으로 해석하면 안 됩니다.",
              "원본 사진·숙도 모델·숙도용 분할 메타데이터는 변경하지 않았습니다.",
              "이전 과일 종류 모델은 previous_classifier.joblib, 선택 모델은 selected_classifier.joblib에 보관했습니다."]
    (directory / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
