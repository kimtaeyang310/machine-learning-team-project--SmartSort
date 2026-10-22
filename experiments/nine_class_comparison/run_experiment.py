"""Isolated stored/matched two-stage vs joint-nine-class YOLO experiment.

All writes stay inside this experiment directory. Original source, metadata and
production models are hashed before and after, and are never used as outputs.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ["YOLO_CONFIG_DIR"] = str(HERE / "runtime" / "ultralytics")
os.environ["MPLCONFIGDIR"] = str(HERE / "runtime" / "matplotlib")
os.environ["YOLO_AUTOINSTALL"] = "false"
for name in ("ultralytics", "matplotlib"):
    (HERE / "runtime" / name).mkdir(parents=True, exist_ok=True)
(HERE / "runtime" / "tmp").mkdir(exist_ok=True)
os.environ["TEMP"] = str(HERE / "runtime" / "tmp")
os.environ["TMP"] = str(HERE / "runtime" / "tmp")

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from ultralytics import YOLO

from src.fruit_classifier import FruitClassifier, unfamiliarity_scores, threshold_values
from src.fruit_identity_experiment import calibrate_per_fruit
from src.predict import load_model, predict_loaded
from src.model_bundle import select_model

FRUITS = ("banana", "mandarin", "tomato")
STAGES = ("unripe", "ripe", "overripe")
CLASSES = sorted(f"{fruit}_{stage}" for fruit in FRUITS for stage in STAGES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_files():
    files = list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py", ROOT / "README.md",
        ROOT / "requirements.txt", ROOT / "requirements-yolo.txt", ROOT / "data/metadata/fruit_labels.csv",
        ROOT / "models/fruit_classifier.joblib", ROOT / "models/ripeness_by_fruit.joblib"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files if p.is_file()}


def materialize(samples, destination, fruit=None):
    """Copy only fitting photos; val has one Train photo/class for the API, not scoring."""
    destination = Path(destination)
    classes = CLASSES if fruit is None else sorted(STAGES)
    for split in ("train", "val"):
        for label in classes:
            (destination / split / label).mkdir(parents=True, exist_ok=True)
    seen = set()
    for sample in samples:
        if sample["phase"] != "train" or sample["fruit"] not in FRUITS:
            continue
        if fruit and sample["fruit"] != fruit:
            continue
        label = f"{sample['fruit']}_{sample['ripeness']}" if fruit is None else sample["ripeness"]
        source = ROOT / sample["path"]
        name = hashlib.sha256(sample["path"].encode()).hexdigest()[:24] + source.suffix.lower()
        target = destination / "train" / label / name
        if not target.exists():
            shutil.copy2(source, target)
        if label not in seen:
            shutil.copy2(source, destination / "val" / label / name)
            seen.add(label)
    if set(seen) != set(classes):
        raise ValueError(f"Missing fitting classes: {set(classes) - seen}")
    return destination.resolve()


def fit(data, destination, weights, config):
    checkpoint = destination / "weights" / "last.pt"
    history = destination / "results.csv"
    completed = 0
    if history.is_file():
        with history.open(encoding="utf-8-sig", newline="") as file:
            completed = max((int(float(row["epoch"])) for row in csv.DictReader(file)), default=0)
    if checkpoint.is_file() and completed >= config["epochs"]:
        print(f"완료된 실험 체크포인트 재사용: {checkpoint}", flush=True)
        return checkpoint
    network = YOLO(str(weights))
    started = perf_counter()
    network.train(data=str(data), project=str(destination.parent.resolve()), name=destination.name,
        epochs=config["epochs"], imgsz=config["imgsz"], batch=config["batch"], device="cpu",
        workers=0, seed=42, deterministic=True, patience=0, val=False, plots=False,
        save=True, exist_ok=True, auto_augment=None, erasing=0., hsv_h=0., hsv_s=0., hsv_v=0.,
        verbose=False)
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing trained checkpoint: {checkpoint}")
    dump(destination / "experiment_training.json", {"wall_seconds": perf_counter() - started,
        "epochs": config["epochs"], "checkpoint_sha256": sha(checkpoint),
        "validation": "Train-only API loader; metrics ignored; fixed last.pt selected"})
    return checkpoint


class NinePredictor:
    """One YOLO forward yields both probabilities and the pre-linear embedding."""
    def __init__(self, checkpoint, imgsz):
        self.network = YOLO(str(checkpoint))
        self.imgsz = imgsz
        self.captured = []
        buffer = self.captured
        # Ultralytics deep-copies the model into its inference backend. A bound
        # method would recursively copy this predictor (including thread locks).
        # A plain function keeps only a shared embedding buffer across that copy.
        def capture(module, args):
            buffer.clear()
            buffer.append(args[0].detach().cpu().numpy().copy())
        self.hook = self.network.model.model[-1].linear.register_forward_pre_hook(capture)
        self.policy = None

    def raw(self, path):
        self.captured.clear()
        result = self.network.predict(str(path), imgsz=self.imgsz, device="cpu", verbose=False)[0]
        if set(result.names.values()) != set(CLASSES):
            raise ValueError(f"Unexpected nine-class labels: {result.names}")
        probabilities = np.zeros(9)
        values = result.probs.data.detach().cpu().numpy()
        for i, name in result.names.items():
            probabilities[CLASSES.index(name)] = float(values[i])
        if not self.captured or self.captured[-1].shape[0] != 1:
            raise ValueError("Missing YOLO pre-classifier embedding")
        vector = self.captured[-1][0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        return probabilities, vector

    def predict(self, path):
        probabilities, embedding = self.raw(path)
        candidate = CLASSES[int(probabilities.argmax())]
        fruit, stage = candidate.split("_", 1)
        scores = unfamiliarity_scores(probabilities[None], embedding[None], self.policy["reference"])
        method = self.policy["method"]
        threshold = self.policy["thresholds"][method][fruit]
        return {"status": "ok" if scores[method][0] <= threshold else "unknown",
                "candidate_fruit": fruit, "candidate_ripeness": stage,
                "candidate_class": candidate, "score": float(scores[method][0]),
                "threshold": threshold, "method": method,
                "probabilities": dict(zip(CLASSES, probabilities.tolist()))}


class TwoStagePredictor:
    def __init__(self, identity_path, ripeness_path=None, matched=None, imgsz=224):
        self.identity = FruitClassifier(identity_path)
        self.bundle = load_model(ripeness_path) if ripeness_path else None
        self.matched = {fruit: YOLO(str(path)) for fruit, path in (matched or {}).items()}
        self.imgsz = imgsz

    def predict(self, path):
        identity = self.identity.classify(path)
        if not identity["accepted"]:
            return {"status": "unknown", "candidate_fruit": identity["candidate_fruit"],
                    "candidate_ripeness": None, "candidate_class": None,
                    "score": identity["score"], "threshold": identity["threshold"], "method": identity["method"]}
        fruit = identity["fruit"]
        if self.bundle:
            estimator, _ = select_model(self.bundle, fruit)
            output = predict_loaded(path, self.bundle, estimator)
            stage = output["ripeness"]
        else:
            result = self.matched[fruit].predict(str(path), imgsz=self.imgsz, device="cpu", verbose=False)[0]
            stage = result.names[int(result.probs.top1)]
            if stage not in STAGES:
                raise ValueError("Unexpected matched ripeness label")
        return {"status": "ok", "candidate_fruit": fruit, "candidate_ripeness": stage,
                "candidate_class": f"{fruit}_{stage}", "score": identity["score"],
                "threshold": identity["threshold"], "method": identity["method"]}


def rejection_comparison(samples, probabilities, scores, thresholds, phase):
    truth = np.array([s["fruit"] for s in samples])
    predicted = np.array([CLASSES[i].split("_", 1)[0] for i in probabilities.argmax(1)])
    known = np.array([s["phase"] == phase and s["fruit"] in FRUITS for s in samples])
    unknown = np.array([s["phase"] == "selection" and s["fruit"] not in FRUITS for s in samples])
    result = {}
    for method, values in scores.items():
        accepted = values <= threshold_values(thresholds[method], predicted)
        rates = [float(np.mean(~accepted[known & (truth == fruit)])) for fruit in FRUITS]
        result[method] = {"unknown_false_acceptance_rate": float(accepted[unknown].mean()),
            "known_false_rejection_rate": float((~accepted[known]).mean()),
            "balanced_known_error": float(np.mean([np.mean((~accepted | (predicted != fruit))[known & (truth == fruit)]) for fruit in FRUITS])),
            "by_fruit_false_rejection_rate": dict(zip(FRUITS, rates)),
            "known_count": int(known.sum()), "unknown_count": int(unknown.sum())}
    return result


def calibrate_nine(predictor, samples, directory):
    dev = [s for s in samples if s["phase"] != "test"]
    probabilities, embeddings = [], []
    for i, sample in enumerate(dev):
        p, e = predictor.raw(ROOT / sample["path"])
        probabilities.append(p); embeddings.append(e)
        if (i + 1) % 200 == 0:
            print(f"9클래스 Train·검증 특징: {i + 1}/{len(dev)}", flush=True)
    probabilities, embeddings = np.asarray(probabilities), np.asarray(embeddings)
    train = np.array([s["phase"] == "train" for s in dev])
    reference = embeddings[train]
    scores = unfamiliarity_scores(probabilities, embeddings, reference)
    np.savez_compressed(directory / "nine_development_predictions.npz", probabilities=probabilities,
                        embeddings=embeddings, msp=scores["msp"], embedding=scores["embedding"])
    fruit_choice = np.array([FRUITS.index(CLASSES[i].split("_", 1)[0]) for i in probabilities.argmax(1)])
    # Same true-fruit 95% correct-kind-and-accepted target as the production identity gate.
    fruit_probabilities = np.eye(3)[fruit_choice]
    thresholds = calibrate_per_fruit(scores, fruit_probabilities, dev)
    comparison = rejection_comparison(dev, probabilities, scores, thresholds, "selection")
    method = min(comparison, key=lambda key: (comparison[key]["unknown_false_acceptance_rate"],
        comparison[key]["balanced_known_error"], comparison[key]["by_fruit_false_rejection_rate"]["tomato"], key != "embedding"))
    predictor.policy = {"reference": reference, "thresholds": thresholds, "method": method}
    joblib.dump(predictor.policy, directory / "nine_rejection_policy.joblib")
    dump(directory / "rejection_selection.json", {"thresholds": thresholds, "selected_method": method,
        "selection": comparison, "calibration": rejection_comparison(dev, probabilities, scores, thresholds, "calibration"),
        "reference_count": int(train.sum()), "selection_uses_test": False})
    return method


def performance_metrics(rows):
    known = [r for r in rows if r["truth_fruit"] in FRUITS]
    unknown = [r for r in rows if r["truth_fruit"] not in FRUITS]
    truth = [r["truth_class"] for r in known]
    predicted = [r["candidate_class"] if r["status"] == "ok" else r["status"] for r in known]
    result = {"known_count": len(known), "unknown_count": len(unknown),
        "nine_class_macro_f1": float(f1_score(truth, predicted, labels=CLASSES, average="macro", zero_division=0)),
        "joint_accuracy_including_rejections": float(accuracy_score(truth, predicted)),
        "known_false_rejection_count": sum(r["status"] == "unknown" for r in known),
        "unknown_false_acceptance_count": sum(r["status"] == "ok" for r in unknown),
        "failure_count": sum(r["status"] == "failed" for r in rows), "per_fruit": {}}
    result["known_false_rejection_rate"] = result["known_false_rejection_count"] / len(known)
    result["unknown_false_acceptance_rate"] = result["unknown_false_acceptance_count"] / len(unknown)
    if all(r.get("candidate_class") for r in known):
        result["without_rejection_nine_class_macro_f1"] = float(f1_score(
            truth, [r["candidate_class"] for r in known], labels=CLASSES, average="macro", zero_division=0))
    for fruit in FRUITS:
        selected = [r for r in known if r["truth_fruit"] == fruit]
        stage_truth = [r["truth_ripeness"] for r in selected]
        stage_predicted = [r["candidate_ripeness"] if r["status"] == "ok" and r["candidate_fruit"] == fruit
                           else "wrong_fruit" if r["status"] == "ok" else r["status"] for r in selected]
        result["per_fruit"][fruit] = {"count": len(selected),
            "ripeness_macro_f1": float(f1_score(stage_truth, stage_predicted, labels=list(STAGES), average="macro", zero_division=0)),
            "correct_kind_and_stage_accuracy": float(accuracy_score(stage_truth, stage_predicted)),
            "false_rejection_count": sum(r["status"] == "unknown" for r in selected)}
    times = np.array([r["total_ms"] for r in rows])
    result["timing"] = {"mean_ms": float(times.mean()), "p95_ms": float(np.percentile(times, 95)),
        "fps": float(len(rows) * 1000 / times.sum()), "total_ms": float(times.sum())}
    result["confusion_labels"] = CLASSES + ["unknown", "failed"]
    result["known_confusion_matrix"] = confusion_matrix(truth, predicted, labels=result["confusion_labels"]).tolist()
    return result


def benchmark(predictors, test, rounds, directory):
    # Warm every active network once; load time is measured separately.
    for name, predictor in predictors.items():
        for fruit in FRUITS:
            sample = next(s for s in test if s["fruit"] == fruit)
            predictor.predict(ROOT / sample["path"])
    order = np.random.default_rng(42).permutation(len(test))
    rows_by_system = {name: [] for name in predictors}
    names = list(predictors)
    for repeat in range(rounds):
        for name in names if repeat % 2 == 0 else names[::-1]:
            print(f"처리 시간 비교: {name} / {repeat + 1}/{rounds}", flush=True)
            for count, index in enumerate(order, 1):
                sample = test[index]
                start = perf_counter()
                try:
                    row = predictors[name].predict(ROOT / sample["path"])
                    row["error"] = ""
                except Exception as error:
                    row = {"status": "failed", "candidate_class": None, "candidate_fruit": None,
                           "candidate_ripeness": None, "error": str(error)}
                row.update(total_ms=(perf_counter() - start) * 1000, repeat=repeat + 1,
                    path=sample["path"], truth_fruit=sample["fruit"], truth_ripeness=sample["ripeness"],
                    truth_class=f"{sample['fruit']}_{sample['ripeness']}" if sample["fruit"] in FRUITS else "unknown")
                rows_by_system[name].append(row)
                if count % 150 == 0:
                    print(f"{name}: {count}/{len(test)}", flush=True)
    metrics = {}
    for name, rows in rows_by_system.items():
        # Quality uses one observation per photo; timing combines all repeated passes.
        metrics[name] = performance_metrics([r for r in rows if r["repeat"] == 1])
        metrics[name]["timing"] = performance_metrics(rows)["timing"]
        metrics[name]["timing"]["rounds"] = rounds
        first = {r["path"]: (r["status"], r["candidate_class"]) for r in rows if r["repeat"] == 1}
        metrics[name]["repeat_disagreements"] = sum(first[r["path"]] != (r["status"], r["candidate_class"])
                                                    for r in rows if r["repeat"] > 1)
        fields = sorted({key for r in rows for key in r})
        with (directory / (name + "_predictions.csv")).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()})
    return metrics


def write_report(directory, result):
    lines = ["# 9클래스 통합 모델 비교 실험", "", "## 비교 조건", "",
        "- stored_current: 현재 저장 모델 스냅샷. 숙도 모델은 더 많은 기존 Train 사진으로 학습했습니다.",
        "- matched_two_stage: 현재 종류·모름 판별기 + 동일 Train으로 새로 학습한 과일별 YOLO 숙도 모델 3개.",
        "- joint_nine: 동일 Train으로 새로 학습한 9클래스 YOLO 1개 + 자체 특징/확률 기반 모름 판정.",
        f"- matched_two_stage와 joint_nine에서 일치시킨 조건: Train 1,223장, 사전학습 가중치, {result['config']['epochs']} epochs, 224 입력, batch 16, CPU, seed 42.",
        "- stored_current 숙도 모델의 원래 Train은 2,129장이고 최종 epoch는 바나나·귤 6, 토마토 5입니다.",
        "- 고정 epoch의 last.pt를 사용합니다. API용 val은 Train 사진 일부이며 점수는 무시합니다.",
        "- Calibration은 모름 임계값, Selection은 모름 점수 선택에만 사용했습니다.",
        "- 기존 Test는 이전 오류 분석에 사용된 자료입니다. 새 독립 일반화 성능으로 해석할 수 없습니다.",
        "- 동일 자료 비교여도 한 모델/여러 모델의 최적화 단계와 배치 구성은 다릅니다. 이번 비교는 한 시드·한 학습 길이입니다.", "",
        "- Ultralytics optimizer=auto를 사용했으므로 클래스 수에 따라 초기 학습률도 다릅니다. 엄밀한 단일 요인 실험으로 해석하면 안 됩니다.", "",
        "## 품질·오류율", "", "| 방식 | 9클래스 Macro F1 | 알려진 과일 오거부 | 미지 과일 오수락 | 실패 |",
        "|---|---|---|---|---|"]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | {m['nine_class_macro_f1']:.4f} | {m['known_false_rejection_count']}/{m['known_count']} ({m['known_false_rejection_rate']:.2%}) | "
            f"{m['unknown_false_acceptance_count']}/{m['unknown_count']} ({m['unknown_false_acceptance_rate']:.2%}) | {m['failure_count']} |")
    lines += ["", "거부·처리 실패 사진도 알려진 정답 클래스의 놓친 예측으로 F1에 반영했습니다.",
              "과일별 숙도 평가는 실제 과일 종류로 묶고, 다른 과일로 분류하거나 거부하면 실패로 반영합니다.", "",
              "| 방식 | 바나나 숙도 Macro F1 | 귤 숙도 Macro F1 | 토마토 숙도 Macro F1 |", "|---|---|---|---|"]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | " + " | ".join(f"{m['per_fruit'][f]['ripeness_macro_f1']:.4f}" for f in FRUITS) + " |")
    raw_f1 = result["metrics"]["joint_nine"].get("without_rejection_nine_class_macro_f1")
    if raw_f1 is not None:
        lines += ["", f"참고: 통합 모델에서 모름 판정을 생략하고 알려진 사진만 9종 분류한 Macro F1은 {raw_f1:.4f}입니다.",
                  "이 경우 미지 과일을 거부하는 기능을 빼므로 위 운영 비교와 같은 조건은 아닙니다."]
    lines += ["", "## 전체 처리 시간", "",
        "모델 로딩은 별도입니다. 신경망별 워밍업 후 같은 사진 순서로 반복하며 시스템 실행 순서는 번갈아 바꿉니다.",
        "사진 읽기·전처리·모름 점수·숙도 예측을 포함하고, 에이전트 후속 정책·결과 출력·CSV 저장은 제외합니다. OS 파일 캐시의 영향을 받습니다.", "",
        "| 방식 | 평균 ms | P95 ms | FPS | 모델 로딩 초 |", "|---|---|---|---|---|"]
    for name, m in result["metrics"].items():
        t = m["timing"]
        lines.append(f"| {name} | {t['mean_ms']:.2f} | {t['p95_ms']:.2f} | {t['fps']:.2f} | {result['model_loading_seconds'][name]:.2f} |")
    lines += ["", "9클래스 방식은 1회의 YOLO forward에서 확률과 분류 직전 특징을 함께 추출합니다.",
        "로딩 시간은 생성 순서·캐시의 영향을 받는 1회 참고값이며, 처리 시간은 로딩·초기 설정 워밍업 후 측정했습니다.",
        "미지 성능은 포도·레몬 표본에 한정됩니다. 숙도별 독립 개체 수가 적으며 특히 미숙 토마토·적숙 귤이 부족합니다.",
        "현재 기본 프로젝트 모델은 교체하지 않았습니다. 원본 해시 검증 결과는 source_integrity.json에 있습니다."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(directory / "report.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--timing-rounds", type=int, default=2)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.timing_rounds < 1:
        raise ValueError("epochs and timing rounds must be positive")
    run = (args.run_dir or HERE / "artifacts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if not run.is_relative_to(HERE):
        raise ValueError("실험 출력은 별도 실험 폴더 안에만 저장할 수 있습니다.")
    run.mkdir(parents=True, exist_ok=True)
    config = {"epochs": args.epochs, "imgsz": 224, "batch": 16, "seed": 42,
              "device": "cpu", "timing_rounds": args.timing_rounds}
    if (run / "config.json").exists():
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Existing experiment config differs")
    else:
        dump(run / "config.json", config)
        dump(run / "source_before.json", source_files())
    snapshot = run / "snapshot"; snapshot.mkdir(exist_ok=True)
    for source, name in ((ROOT / "models/fruit_classifier.joblib", "fruit_identity.joblib"),
                         (ROOT / "models/ripeness_by_fruit.joblib", "ripeness_original.joblib"),
                         (ROOT / "yolov8n-cls.pt", "pretrained.pt")):
        if not (snapshot / name).exists():
            shutil.copy2(source, snapshot / name)
    identity = joblib.load(snapshot / "fruit_identity.joblib")
    samples = identity["samples"]
    groups = {}
    for sample in samples:
        source = ROOT / sample["path"]
        if sha(source) != sample["sha256"]:
            raise ValueError(f"Image changed: {source}")
        groups.setdefault(sample["group"], set()).add(sample["phase"])
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError("Specimen leakage across experiment splits")
    dump(run / "split_manifest.json", samples)
    ripeness = joblib.load(snapshot / "ripeness_original.joblib")
    source_checkpoints = {}
    for fruit, entry in ripeness["models"].items():
        checkpoint = ROOT / entry["model"].checkpoint
        target = snapshot / (fruit + "_current.pt")
        if not target.exists():
            shutil.copy2(checkpoint, target)
        source_checkpoints[checkpoint.relative_to(ROOT).as_posix()] = sha(checkpoint)
        entry["model"].checkpoint = target.relative_to(ROOT).as_posix()
    joblib.dump(ripeness, snapshot / "ripeness_isolated.joblib")
    dump(run / "source_checkpoints.json", source_checkpoints)
    print("실험 폴더:", run, flush=True)
    print("분할:", Counter((s["fruit"], s["phase"]) for s in samples), flush=True)
    torch.set_num_threads(4)
    nine_data = materialize(samples, run / "datasets" / "nine")
    nine_checkpoint = fit(nine_data, run / "training" / "nine", snapshot / "pretrained.pt", config)
    matched = {}
    for fruit in FRUITS:
        data = materialize(samples, run / "datasets" / fruit, fruit)
        matched[fruit] = fit(data, run / "training" / fruit, snapshot / "pretrained.pt", config)
    torch.set_num_threads(4)
    start = perf_counter(); joint = NinePredictor(nine_checkpoint, 224)
    loading = {"joint_nine": perf_counter() - start}
    calibrate_nine(joint, samples, run)
    start = perf_counter(); stored = TwoStagePredictor(snapshot / "fruit_identity.joblib", snapshot / "ripeness_isolated.joblib")
    loading["stored_current"] = perf_counter() - start
    start = perf_counter(); controlled = TwoStagePredictor(snapshot / "fruit_identity.joblib", matched=matched)
    loading["matched_two_stage"] = perf_counter() - start
    metrics = benchmark({"stored_current": stored, "matched_two_stage": controlled, "joint_nine": joint},
                        [s for s in samples if s["phase"] == "test"], args.timing_rounds, run)
    before = json.loads((run / "source_before.json").read_text())
    after = source_files()
    changed = [p for p in before if before[p] != after.get(p)]
    changed += [p for p, value in source_checkpoints.items() if sha(ROOT / p) != value]
    dump(run / "source_integrity.json", {"unchanged": not changed, "changed_files": changed,
                                          "before": before, "after": after})
    result = {"config": config, "metrics": metrics, "model_loading_seconds": loading,
              "nine_rejection_method": joint.policy["method"], "original_project_unchanged": not changed,
              "evaluation_status": "previously_inspected_legacy_test_regression",
              "split_counts": dict(Counter(s["phase"] for s in samples)),
              "fairness": "stored baseline has more ripeness fitting data; matched baseline and joint use identical fitting photos"}
    write_report(run, result)
    print(json.dumps({"run": str(run), **result}, ensure_ascii=False, indent=2), flush=True)
    if changed:
        raise RuntimeError("기존 프로젝트 파일의 외부 변경이 확인됐습니다. source_integrity.json을 확인하세요.")


if __name__ == "__main__":
    main()
