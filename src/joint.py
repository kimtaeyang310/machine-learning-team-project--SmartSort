"""Production entry points for a single nine-class YOLO and rejection policy."""
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import hashlib
import json
import shutil
import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "joint_nine"
MODEL_NAME = "yolo26n_cls_joint_nine"
PRETRAINED_WEIGHTS = ROOT / "models" / "pretrained" / "yolo26n-cls.pt"
PREDICT_DIR = ROOT / "data" / "predict_list"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


class JointClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        from src.joint_core import NinePredictor
        self.directory = resolve(model_dir)
        metadata = json.loads((self.directory / "bundle.json").read_text(encoding="utf-8"))
        self.model_name = metadata.get("model_name", "yolov8n_cls_joint_nine")
        for name, expected in metadata["sha256"].items():
            actual = hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"통합 모델 파일이 변경됐습니다: {name}")
        torch.set_num_threads(4)
        self.engine = NinePredictor(self.directory / "model.pt", metadata["imgsz"])
        self.engine.policy = joblib.load(self.directory / "policy.joblib")
        self.sha = metadata["sha256"]["model.pt"]

    def predict(self, image_path):
        from src.agent import decide_next_action
        path = resolve(image_path)
        start = perf_counter()
        result = self.engine.predict(path)
        # Preserve joint class probabilities; agent confidence uses the winning
        # JOINT score, without renormalizing away uncertainty over fruit kinds.
        accepted = result["status"] == "ok"
        fruit = result["candidate_fruit"]
        stage = result["candidate_ripeness"]
        stage_scores = {s: p for label, p in result["probabilities"].items()
                        for f, s in [label.split("_", 1)] if f == fruit}
        result["nine_probabilities"] = result["probabilities"]
        result.update(image_path=str(path), fruit=fruit if accepted else "unknown",
                      ripeness=stage if accepted else None, model=getattr(self, "model_name", "yolov8n_cls_joint_nine"),
                      probabilities=stage_scores if accepted else None, error="")
        result["fruit_identity"] = {"accepted": accepted, "candidate_fruit": fruit,
            "method": result["method"], "score": result["score"],
            "threshold": result["threshold"], "model_sha256": self.sha}
        elapsed = (perf_counter() - start) * 1000
        result.update(total_ms=elapsed, feature_ms=0., prediction_ms=elapsed)
        result["agent_decision"] = decide_next_action(result) if accepted else {
            "action": "human_review", "reason": "모르는 과일입니다. 숙도 결과를 표시하지 않습니다.",
            "suggested_label": None, "confidence": None, "requires_confirmation": True}
        return result


def predict_folder(folder_path=PREDICT_DIR, model_dir=MODEL_DIR):
    folder = resolve(folder_path)
    if not folder.is_dir():
        raise ValueError(f"예측 폴더가 없습니다: {folder}")
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"지원하는 이미지가 없습니다: {folder}")
    start = perf_counter()
    classifier = JointClassifier(model_dir)
    load_ms = (perf_counter() - start) * 1000
    # Match experiment: initialization and a warmup are outside timed inference.
    for path in paths:
        try:
            classifier.predict(path)
            break
        except Exception:
            continue
    results = []
    for index, path in enumerate(paths, 1):
        start = perf_counter()
        try:
            result = classifier.predict(path)
        except Exception as error:
            result = {"image_path": str(path), "status": "failed", "error": str(error),
                      "model": classifier.model_name,
                      "total_ms": (perf_counter() - start) * 1000}
        results.append(result)
        label = (f"{result['fruit']} / {result['ripeness']}" if result["status"] == "ok"
                 else "모르는 과일" if result["status"] == "unknown" else f"실패: {result['error']}")
        print(f"[{index}/{len(paths)}] {path.name} → {label} ({result['total_ms']:.2f} ms)")
    summary = timing_summary(results)
    summary["model_load_ms"] = load_ms
    print(f"\n숙도 판정 {summary['success_count']} / 모르는 과일 {summary['unknown_count']} / 실패 {summary['failure_count']}")
    print(f"평균 {summary['mean_ms']:.2f} ms / P95 {summary['p95_ms']:.2f} ms / 처리 FPS: {summary['fps']:.2f}")
    print("워밍업 후 측정. 읽기·전처리·YOLO·거리 판정 포함, 모델 로딩·에이전트 정책·출력·저장 제외.")
    return results, summary


def timing_summary(rows):
    times = np.array([r["total_ms"] for r in rows])
    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "unknown", "failed")}
    return {"count": len(rows), "success_count": counts["ok"], "unknown_count": counts["unknown"],
            "failure_count": counts["failed"], "processing_ms": float(times.sum()),
            "mean_ms": float(times.mean()), "p95_ms": float(np.percentile(times, 95)),
            "fps": (counts["ok"] + counts["unknown"]) * 1000 / max(float(times.sum()), 1e-12)}


def publish(checkpoint, policy, manifest, origin, destination=MODEL_DIR, *, model_name=MODEL_NAME):
    """Install a version directory and retain its predecessor before promotion."""
    from src.joint_core import dump, sha
    destination = resolve(destination)
    staging = destination.with_name(destination.name + "_staging_" + stamp())
    staging.mkdir(parents=True)
    for source, name in [(checkpoint, "model.pt"), (policy, "policy.joblib"), (manifest, "split_manifest.json")]:
        shutil.copy2(source, staging / name)
    dump(staging / "bundle.json", {"format": 1, "imgsz": 224, "source": str(origin),
        "model_name": model_name,
        "sha256": {name: sha(staging / name) for name in ("model.pt", "policy.joblib", "split_manifest.json")}})
    JointClassifier(staging)  # Validate loadability before switching.
    if destination.exists():
        backup = ROOT / "outputs" / "joint_nine" / ("previous_" + stamp())
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(destination, backup)
        # Only replace the three known artifacts and metadata, never recursively remove files.
        for name in ("model.pt", "policy.joblib", "split_manifest.json", "bundle.json"):
            shutil.copy2(staging / name, destination / name)
    else:
        staging.rename(destination)
    return destination


def train(epochs=6):
    from src.joint_core import materialize, fit, calibrate_nine, NinePredictor, dump, sha
    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")
    run = ROOT / "outputs" / "joint_nine" / ("training_" + stamp())
    run.mkdir(parents=True)
    samples = json.loads((MODEL_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    for s in samples:
        if sha(ROOT / s["path"]) != s["sha256"]:
            raise ValueError(f"학습 기준 사진이 변경됐습니다: {s['path']}")
    groups = {}
    for s in samples:
        groups.setdefault(s["group"], set()).add(s["phase"])
    if any(len(phases) > 1 for phases in groups.values()):
        raise ValueError("분할 사이 개체 중복이 있습니다.")
    dump(run / "split_manifest.json", samples)
    config = {"epochs": epochs, "imgsz": 224, "batch": 16, "seed": 42, "device": "cpu"}
    dump(run / "config.json", config)
    torch.set_num_threads(4)
    data = materialize(samples, run / "dataset")
    checkpoint = fit(data, run / "training", PRETRAINED_WEIGHTS, config)
    predictor = NinePredictor(checkpoint, 224)
    calibrate_nine(predictor, samples, run)
    publish(checkpoint, run / "nine_rejection_policy.joblib", run / "split_manifest.json", run)
    print(f"통합 모델 학습 완료: {run}")
    return run


def evaluate():
    from src.joint_core import performance_metrics, dump
    from src.predict import save_prediction_artifacts
    samples = json.loads((MODEL_DIR / "split_manifest.json").read_text(encoding="utf-8"))
    test = [s for s in samples if s["phase"] == "test"]
    classifier = JointClassifier()
    classifier.predict(ROOT / test[0]["path"])
    rows = []
    for i in np.random.default_rng(42).permutation(len(test)):
        s = test[i]
        from src.joint_core import sha
        if sha(ROOT / s["path"]) != s["sha256"]:
            raise ValueError(f"평가 기준 사진이 변경됐습니다: {s['path']}")
        start = perf_counter()
        try:
            result = classifier.predict(ROOT / s["path"])
        except Exception as error:
            result = {"image_path": str(ROOT / s["path"]), "status": "failed", "error": str(error),
                      "model": classifier.model_name,
                      "total_ms": (perf_counter() - start) * 1000}
        result.update(truth_fruit=s["fruit"], truth_ripeness=s["ripeness"],
                      truth_class=f"{s['fruit']}_{s['ripeness']}")
        rows.append(result)
    metrics = performance_metrics(rows)
    run = ROOT / "outputs" / "joint_nine" / ("evaluation_" + stamp())
    dump(run / "metrics.json", metrics)
    dump(run / "predictions.json", rows)
    save_prediction_artifacts(rows, run / "predictions.csv")
    text = f"9클래스 Macro F1: {metrics['nine_class_macro_f1']:.4f}\n"
    text += f"오거부율: {metrics['known_false_rejection_rate']:.2%}\n오수락률: {metrics['unknown_false_acceptance_rate']:.2%}\n"
    for fruit, value in metrics["per_fruit"].items():
        text += f"{fruit} 숙도 Macro F1: {value['ripeness_macro_f1']:.4f}\n"
    text += f"FPS: {timing_summary(rows)['fps']:.2f}\n기존에 확인한 Test 자료를 이용한 회귀 평가입니다.\n"
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text, f"결과: {run}")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="통합 YOLO 학습·평가")
    parser.add_argument("task", choices=("train", "evaluate"))
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    train(args.epochs) if args.task == "train" else evaluate()
