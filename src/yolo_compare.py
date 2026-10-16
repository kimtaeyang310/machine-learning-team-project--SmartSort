"""YOLO helpers for unified training and readers for legacy comparison runs.

python -m src.train --epochs 30 --device cpu
python -m src.yolo_compare evaluate --run outputs/yolo_comparison/<run>
python -m src.yolo_compare predict --run outputs/yolo_comparison/<run> --folder data/predict_list
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.dataset import RIPENESS_LABELS
from src.evaluate import load_inputs, extract_test_features
from src.model_bundle import FRUITS, PROJECT_ROOT
from src.predict import infer_fruit, IMAGE_EXTENSIONS
SEED = 42

RAW = PROJECT_ROOT / "data" / "raw"
OUTPUT = PROJECT_ROOT / "outputs" / "yolo_comparison"
YOLO_NAME = "yolov8n_cls"


class YoloClassifier:
    """Serializable checkpoint reference; load the network only at inference."""
    input_kind = "image"
    classes_ = np.array([0, 1, 2])

    def __init__(self, checkpoint, config):
        self.checkpoint = str(Path(checkpoint).resolve().relative_to(PROJECT_ROOT.resolve()))
        self.config = dict(config)
        self._model = None

    def __getstate__(self):
        return dict(self.__dict__, _model=None)

    def load(self):
        if self._model is None:
            path = PROJECT_ROOT / self.checkpoint
            if not path.is_file():
                raise FileNotFoundError(f"YOLO 체크포인트가 없습니다: {path}")
            self._model = yolo_class()(str(path))
        return self

    def predict_proba(self, paths):
        self.load()
        return probabilities(self._model, paths, self.config)

    def predict(self, paths):
        return self.classes_[self.predict_proba(paths).argmax(1)]


def compare_fruit(samples, splits, root, config):
    """Train-only fold scoring plus a full-Train refit; Test is never passed."""
    frame = pd.DataFrame(samples)
    scores = []
    for fold, (a, v) in enumerate(splits, 1):
        folder = root / f"fold_{fold}"
        folder.mkdir(parents=True)
        frame.iloc[a].drop(columns=["path"], errors="ignore").to_csv(folder / "train_samples.csv", index=False)
        frame.iloc[v].drop(columns=["path"], errors="ignore").to_csv(folder / "valid_samples.csv", index=False)
        data = materialize(frame.iloc[a], frame.iloc[v], folder / "dataset")
        checkpoint = fit_yolo(data, folder / "training", config)
        model = yolo_class()(str(checkpoint))
        values = probabilities(model, [image_path(p) for p in frame.iloc[v].image_path], config).argmax(1)
        scores.append(f1_score(frame.iloc[v].ripeness_label, values, labels=[0, 1, 2],
                               average="macro", zero_division=0))
    row = {"model": YOLO_NAME, "cv_macro_f1_mean": float(np.mean(scores)),
           "cv_macro_f1_std": float(np.std(scores)),
           **{f"fold_{i}_macro_f1": score for i, score in enumerate(scores, 1)}}
    # API needs a val loader; final Train-only metrics are ignored.
    data = materialize(frame, frame, root / "final" / "dataset")
    checkpoint = fit_yolo(data, root / "final" / "training", config)
    return row, YoloClassifier(checkpoint, config)


def yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO 의존성이 없습니다. python -m pip install -r requirements-yolo.txt 실행 후 다시 시도하세요.") from error
    return YOLO


def image_path(relative):
    path = (RAW / relative).resolve()
    path.relative_to(RAW.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def materialize(train, valid, destination):
    """Explicit class folders prevent Ultralytics from making a random split."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for split, frame in (("train", train), ("val", valid)):
        for label in RIPENESS_LABELS:
            (destination / split / label).mkdir(parents=True)
        for row in frame.to_dict("records"):
            source = image_path(row["image_path"])
            key = hashlib.sha256(row["image_path"].encode()).hexdigest()[:20]
            shutil.copy2(source, destination / split / row["ripeness"] / (key + source.suffix.lower()))
    return destination.resolve()


def probabilities(model, paths, config):
    """Map alphabetical YOLO class indices to the project's explicit label IDs."""
    result = []
    for path in paths:
        prediction = model.predict(source=str(path), imgsz=config["imgsz"],
                                   device=config["device"], verbose=False)[0]
        names = prediction.names
        if set(names.values()) != set(RIPENESS_LABELS):
            raise ValueError(f"YOLO 클래스가 숙도 3개와 다릅니다: {names}")
        values = prediction.probs.data.cpu().numpy()
        ordered = np.zeros(3, dtype=float)
        for index, name in names.items():
            ordered[RIPENESS_LABELS[name]] = values[index]
        if not np.isfinite(ordered).all() or (ordered < 0).any() or not np.isclose(ordered.sum(), 1, atol=1e-4):
            raise ValueError("YOLO 확률 형식이 올바르지 않습니다.")
        result.append(ordered)
    return np.asarray(result).reshape(-1, 3)


def fit_yolo(data, output, config):
    # Each fold starts independently from the SAME pretrained checkpoint.
    model = yolo_class()(config["weights"])
    model.train(data=str(data), project=str(output.parent.resolve()), name=output.name,
                epochs=config["epochs"], imgsz=config["imgsz"], batch=config["batch"],
                device=config["device"], workers=0, seed=SEED, deterministic=True,
                patience=0, val=False, plots=False, save=True, exist_ok=False,
                auto_augment=None, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    # Fixed epoch budget: never select a checkpoint using the scoring fold or Test.
    checkpoint = Path(model.trainer.save_dir) / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"학습된 last.pt가 없습니다: {checkpoint}")
    return checkpoint


def train(config):
    """Backward-compatible alias for the unified training entry point."""
    from src.train import train as unified_train
    return unified_train(**config)


def load_run(run, device=None):
    run = Path(run).resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "models.json").read_text(encoding="utf-8"))
    if set(manifest) != set(FRUITS):
        raise ValueError("과일별 YOLO 모델이 모두 필요합니다.")
    if device is not None:
        config["device"] = device
    return run, config, manifest


def evaluate(run, device=None):
    run, config, manifest = load_run(run, device)
    bundle, test = load_inputs(run / "baseline.joblib", run / "train_samples.csv", run / "test_samples.csv")
    X = extract_test_features(test, bundle, run)
    rows, predictions = [], []
    for fruit in FRUITS:
        positions = np.flatnonzero(test.fruit.eq(fruit).to_numpy())
        frame = test.iloc[positions].copy()
        if frame.empty:
            raise ValueError(f"{fruit} Test 사진이 없습니다.")
        model = yolo_class()(str(run / manifest[fruit]))
        yp = probabilities(model, [image_path(p) for p in frame.image_path], config).argmax(1)
        baseline = bundle["models"][fruit]
        bp = baseline["model"].predict(X.iloc[positions])
        for name, values in ((YOLO_NAME, yp), (baseline["model_name"], bp)):
            rows.append({"fruit": fruit, "model": name, "test_dataset_id": config["test_dataset_id"],
                         "accuracy": accuracy_score(frame.ripeness_label, values),
                         "macro_f1": f1_score(frame.ripeness_label, values, labels=[0, 1, 2], average="macro", zero_division=0)})
            detail = frame.assign(model=name, predicted_label=values)
            detail["predicted_ripeness"] = detail.predicted_label.map({v:k for k,v in RIPENESS_LABELS.items()})
            predictions.append(detail)
            pd.DataFrame(confusion_matrix(frame.ripeness_label, values, labels=[0, 1, 2]),
                         index=list(RIPENESS_LABELS), columns=list(RIPENESS_LABELS)).to_csv(run / f"{fruit}_{name}_confusion.csv")
    summary = pd.DataFrame(rows)
    summary.to_csv(run / "test_comparison.csv", index=False)
    pd.concat(predictions).to_csv(run / "test_predictions.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def predict_folder(run, folder, device=None):
    run, config, manifest = load_run(run, device)
    paths = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError("예측할 이미지가 없습니다.")
    started = perf_counter()
    models = {fruit: yolo_class()(str(run / manifest[fruit])) for fruit in FRUITS}
    print(f"모델 로딩: {(perf_counter()-started)*1000:.2f} ms")
    rows = []
    inverse = {v:k for k,v in RIPENESS_LABELS.items()}
    for path in paths:
        started = perf_counter()
        try:
            fruit = infer_fruit(path)
            probs = probabilities(models[fruit], [path], config)[0]
            row = dict(image_path=str(path), fruit=fruit, ripeness=inverse[int(probs.argmax())],
                       confidence=float(probs.max()), status="ok")
        except Exception as error:
            row = dict(image_path=str(path), status="failed", error=str(error))
        row["total_ms"] = (perf_counter()-started)*1000
        rows.append(row)
        print(row)
    total = sum(r["total_ms"] for r in rows)
    success = sum(r["status"] == "ok" for r in rows)
    print(f"성공 {success}/{len(rows)}, FPS {success*1000/total:.2f}; 첫 추론 포함, 로딩·출력 제외")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="기존 5개 모델 + YOLOv8n-cls 별도 비교")
    parser.add_argument("action", choices=["train", "evaluate", "predict"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--folder", type=Path, default=PROJECT_ROOT / "data" / "predict_list")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", default="yolov8n-cls.pt")
    args = parser.parse_args(argv)
    if args.action == "train":
        if min(args.epochs, args.imgsz, args.batch) < 1:
            parser.error("epochs, imgsz, batch는 양수여야 합니다.")
        train(dict(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                   device=args.device or "cpu", weights=args.weights))
    elif args.run is None:
        parser.error("--run에 완료된 비교 실행 폴더를 지정하세요.")
    elif args.action == "evaluate":
        evaluate(args.run, args.device)
    else:
        predict_folder(args.run, args.folder, args.device)


if __name__ == "__main__":
    main()
