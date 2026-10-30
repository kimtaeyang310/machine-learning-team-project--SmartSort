"""Shared nine-class YOLO training, calibration and scoring primitives.

Promoted from the verified isolated experiment. No legacy model is loaded here.
"""
from pathlib import Path
import csv, hashlib, json, shutil
from time import perf_counter
import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from ultralytics import YOLO
from src.rejection import unfamiliarity_scores, threshold_values, calibrate_per_fruit, acceptance_mask
ROOT = Path(__file__).resolve().parents[1]
FRUITS = ("banana", "mandarin", "tomato")
STAGES = ("unripe", "ripe", "overripe")
CLASSES = sorted(f"{fruit}_{stage}" for fruit in FRUITS for stage in STAGES)
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")




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
        source = path if isinstance(path, np.ndarray) else str(path)
        result = self.network.predict(source, imgsz=self.imgsz, device="cpu", verbose=False)[0]
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
        threshold = self.policy.get("class_thresholds", {}).get(candidate, threshold)
        accepted = bool(acceptance_mask(probabilities[None], scores, self.policy, CLASSES)[0])
        ordered = np.sort(probabilities)
        return {"status": "ok" if accepted else "unknown",
                "acceptance_rule": self.policy.get("acceptance", {"kind": "distance"}),
                "joint_confidence": float(ordered[-1]), "joint_margin": float(ordered[-1] - ordered[-2]),
                "candidate_fruit": fruit, "candidate_ripeness": stage,
                "candidate_class": candidate, "score": float(scores[method][0]),
                "threshold": threshold, "method": method,
                "probabilities": dict(zip(CLASSES, probabilities.tolist()))}




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


