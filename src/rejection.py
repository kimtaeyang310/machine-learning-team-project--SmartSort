"""Shared rejection scores and calibration for the joint nine-class model."""

import numpy as np

FRUITS = ("banana", "mandarin", "tomato")

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
