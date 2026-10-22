"""Image-based fruit identification and open-set rejection (no ripeness prediction)."""

from pathlib import Path
import hashlib

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRUITS = ("banana", "mandarin", "tomato")
MODEL_PATH = ROOT / "models" / "fruit_classifier.joblib"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PREPROCESS_MODES = ("center_crop", "letterbox", "foreground_crop")


class LinearSoftmaxClassifier:
    """Portable multinomial linear head: NumPy arrays, independent of sklearn internals."""

    def __init__(self, coefficients, intercept, classes):
        self.coef_ = np.array(coefficients, dtype=np.float64, copy=True)
        self.intercept_ = np.array(intercept, dtype=np.float64, copy=True)
        self.classes_ = np.array(classes, copy=True)
        if (self.coef_.ndim != 2 or len(self.classes_) < 3
                or self.coef_.shape[0] != len(self.classes_)
                or self.intercept_.shape != (len(self.classes_),)
                or not np.isfinite(self.coef_).all() or not np.isfinite(self.intercept_).all()):
            raise ValueError("Invalid multinomial softmax coefficients")
        self.n_features_in_ = self.coef_.shape[1]

    @classmethod
    def from_estimator(cls, estimator):
        if isinstance(estimator, cls):
            return estimator
        # Only the multinomial lbfgs model trained by this project is supported.
        # OvR logistic probabilities have different semantics and must not be converted.
        if (getattr(estimator, "solver", None) != "lbfgs"
                or getattr(estimator, "multi_class", "auto") not in ("auto", "deprecated", "multinomial")):
            raise ValueError("Only multinomial lbfgs classifiers can be exported")
        return cls(estimator.coef_, estimator.intercept_, estimator.classes_)

    def predict_proba(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_ or not np.isfinite(features).all():
            raise ValueError("Invalid fruit identity features")
        logits = features @ self.coef_.T + self.intercept_
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def prepare_identity_image(image, mode):
    """Deterministic label-free crop. Color is preserved; no ripeness masks are used."""
    from PIL import ImageOps
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocessing: {mode}")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mode == "center_crop":
        return image
    if mode == "foreground_crop":
        import cv2
        # Work on a small proxy. Components are selected without knowing fruit type.
        proxy = ImageOps.contain(image, (640, 640))
        hsv = cv2.cvtColor(np.asarray(proxy), cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 25)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(areas.max())
            if largest >= 0.005 * proxy.width * proxy.height:
                ids = np.where(areas >= max(largest * 0.2, 0.002 * proxy.width * proxy.height))[0] + 1
                yy, xx = np.where(np.isin(labels, ids))
                x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
                margin = int(max(x1 - x0, y1 - y0) * 0.12) + 1
                sx, sy = image.width / proxy.width, image.height / proxy.height
                box = (max(0, int((x0 - margin) * sx)), max(0, int((y0 - margin) * sy)),
                       min(image.width, int((x1 + margin) * sx)), min(image.height, int((y1 + margin) * sy)))
                image = image.crop(box)
        # No reliable foreground: preserve the whole image instead of declaring unknown.
    return ImageOps.pad(image, (224, 224), color=(255, 255, 255))


def threshold_values(threshold, predicted):
    """Per-predicted-fruit thresholds at inference; scalar v1 thresholds remain supported."""
    if isinstance(threshold, dict):
        return np.array([threshold[str(fruit)] for fruit in predicted], dtype=float)
    return np.full(len(predicted), float(threshold))


def rejection_threshold(scores, acceptance=0.95):
    """Higher scores mean unfamiliar. Ties pass; empirical acceptance is >= target."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < acceptance <= 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores or acceptance target")
    return float(np.quantile(scores, acceptance, method="higher"))


def unfamiliarity_scores(probabilities, embeddings, reference, k=5):
    reference = np.asarray(reference, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not len(reference) or not np.isfinite(embeddings).all():
        raise ValueError("Missing reference embeddings or invalid input")
    count = min(k, len(reference))
    if count < 1:
        raise ValueError("k must be positive")
    # All embeddings are L2-normalized by the encoder.
    distance = np.clip(1 - embeddings @ reference.T, 0, 2)
    nearest = np.partition(distance, count - 1, axis=1)[:, :count].mean(axis=1)
    return {"msp": 1 - np.max(probabilities, axis=1), "embedding": nearest}


def rejection_metrics(known_scores, unknown_scores, threshold):
    known_scores, unknown_scores = np.asarray(known_scores), np.asarray(unknown_scores)
    return {
        "known_count": int(len(known_scores)),
        "known_false_rejection_count": int(np.sum(known_scores > threshold)),
        "known_false_rejection_rate": float(np.mean(known_scores > threshold)),
        "unknown_count": int(len(unknown_scores)),
        "unknown_false_acceptance_count": int(np.sum(unknown_scores <= threshold)),
        "unknown_false_acceptance_rate": float(np.mean(unknown_scores <= threshold)),
    }


class ImageEncoder:
    """Frozen ImageNet MobileNetV3-small features + a separately trained softmax head.

    The pooled 576-D backbone output is the embedding immediately before our head.
    Stored backbone state makes prediction independent of network/cache availability.
    """

    def __init__(self, state=None, preprocess_mode="center_crop"):
        import torch
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        torch.set_num_threads(min(4, torch.get_num_threads()))
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        if state is None:
            torch.hub.set_dir(str(ROOT / "models" / ".torch_cache"))
            net = mobilenet_v3_small(weights=weights)
        else:
            net = mobilenet_v3_small(weights=None)
            net.features.load_state_dict({key: torch.from_numpy(value) for key, value in state.items()})
        self.features = net.features.eval()
        self.preprocess_mode = preprocess_mode
        if preprocess_mode not in PREPROCESS_MODES:
            raise ValueError(f"Unknown preprocessing: {preprocess_mode}")
        if preprocess_mode == "center_crop":
            self.transform = weights.transforms()
        else:
            from torchvision.transforms import Compose, ToTensor, Normalize
            self.transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.torch = torch

    def state(self):
        return {key: value.cpu().numpy() for key, value in self.features.state_dict().items()}

    def encode(self, paths, batch_size=32):
        from PIL import Image, ImageOps
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                tensors = []
                for path in paths[start:start + batch_size]:
                    with Image.open(path) as im:
                        tensors.append(self.transform(prepare_identity_image(im, self.preprocess_mode)))
                x = self.torch.stack(tensors)
                pooled = self.features(x).mean((2, 3))
                batches.append(self.torch.nn.functional.normalize(pooled, dim=1).numpy())
        return np.concatenate(batches) if batches else np.empty((0, 576), dtype=np.float32)


class FruitClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.path = Path(model_path)
        if not self.path.is_file():
            raise FileNotFoundError("과일 종류 모델이 없습니다. 먼저 python -m src.train_fruit_classifier 를 실행하세요.")
        self.bundle = joblib.load(self.path)
        if self.bundle.get("version") not in (1, 2) or tuple(self.bundle.get("classes", [])) != FRUITS:
            raise ValueError("과일 종류 모델 형식이 올바르지 않습니다.")
        self.encoder = ImageEncoder(self.bundle["encoder_state"], self.bundle.get("preprocess_mode", "center_crop"))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def score(self, embeddings):
        probabilities = self.bundle["classifier"].predict_proba(embeddings)
        scores = unfamiliarity_scores(probabilities, embeddings, self.bundle["reference"])
        return probabilities, scores

    def classify(self, image_path):
        probabilities, scores = self.score(self.encoder.encode([image_path]))
        candidate = str(self.bundle["classifier"].classes_[probabilities[0].argmax()])
        method = self.bundle["selected_method"]
        score = float(scores[method][0])
        threshold = float(threshold_values(self.bundle["thresholds"][method], [candidate])[0])
        accepted = score <= threshold
        return {"fruit": candidate if accepted else "unknown", "candidate_fruit": candidate,
                "accepted": bool(accepted), "method": method, "score": score,
                "threshold": threshold, "scores": {k: float(v[0]) for k, v in scores.items()},
                "probabilities": {str(k): float(v) for k, v in
                                  zip(self.bundle["classifier"].classes_, probabilities[0])},
                "preprocess_mode": self.bundle.get("preprocess_mode", "center_crop"),
                "model_sha256": self.sha256}
