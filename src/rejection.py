"""Shared rejection scores and calibration for the joint nine-class model."""
# 최고 점수 후보라도 학습 자료와 너무 다르면 unknown으로 거부합니다.
# 낯섦 점수는 클수록 낯선 사진을 뜻하며 임계값 이하이면 기본 통과 조건을 만족합니다.

import numpy as np

FRUITS = ("banana", "mandarin", "tomato")


# 사진별 수락 여부 배열을 반환하며 클래스별 기준이 과일별 기준보다 우선합니다.
def acceptance_mask(probabilities, scores, policy, classes):
    """Shared runtime/evaluation gate; optional hybrid config preserves old policies."""
    probabilities = np.asarray(probabilities, dtype=float)
    indices = probabilities.argmax(axis=1)
    labels = [classes[i] for i in indices]
    fruits = [label.split("_", 1)[0] for label in labels]
    method = policy["method"]
    thresholds = np.array([policy.get("class_thresholds", {}).get(label,
        policy["thresholds"][method][fruit]) for label, fruit in zip(labels, fruits)])
    values = np.asarray(scores[method])
    # NaN이나 무한대가 섞인 결과는 어떤 규칙에서도 통과시키지 않습니다.
    valid = np.isfinite(probabilities).all(axis=1) & np.isfinite(values)
    baseline = values <= thresholds
    config = policy.get("acceptance", {"kind": "distance"})
    kind = config["kind"]
    if kind == "distance":
        return valid & baseline
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    # margin은 1위와 2위 확률의 차이로 상위 후보 간 혼동 정도를 나타냅니다.
    margin = ordered[:, -1] - ordered[:, -2]
    # 같은 과일의 세 숙도 확률을 더해 종류에 대한 확신도 따로 확인합니다.
    fruit_scores = np.column_stack([probabilities[:, [i for i, label in enumerate(classes)
        if label.startswith(fruit + "_")]].sum(axis=1) for fruit in FRUITS])
    chosen = np.array([FRUITS.index(fruit) for fruit in fruits])
    fruit_confidence = fruit_scores[np.arange(len(chosen)), chosen]
    others = fruit_scores.copy()
    others[np.arange(len(chosen)), chosen] = -np.inf
    fruit_margin = fruit_confidence - others.max(axis=1)
    strong = ((confidence >= config.get("min_confidence", 0.)) &
              (margin >= config.get("min_margin", 0.)) &
              (fruit_confidence >= config.get("min_fruit_confidence", 0.)) &
              (fruit_margin >= config.get("min_fruit_margin", -1.)))
    if kind == "confidence":
        accepted = confidence >= config["min_confidence"]
    elif kind == "margin":
        accepted = margin >= config["min_margin"]
    elif kind in {"rescue", "gated_distance"}:
        cap = config.get("distance_multiplier")
        bounded = np.ones(len(values), dtype=bool) if cap is None else values <= thresholds * cap
        # rescue는 기본 통과에 강한 확신 사례를 추가하고 gated_distance는 확신과 거리 조건을 함께 요구합니다.
        accepted = baseline | (strong & bounded) if kind == "rescue" else strong & bounded
    else:
        raise ValueError(f"Unknown acceptance rule: {kind}")
    return valid & accepted

# 예측 과일 순서대로 임계값 배열을 만들며 예전 단일 숫자 기준도 지원합니다.
def threshold_values(threshold, predicted):
    """Per-predicted-fruit thresholds at inference; scalar v1 thresholds remain supported."""
    if isinstance(threshold, dict):
        return np.array([threshold[str(fruit)] for fruit in predicted], dtype=float)
    return np.full(len(predicted), float(threshold))

# 목표 수락 비율을 충족하는 분위수를 고르며 경계값과 같은 점수도 통과시킵니다.
def rejection_threshold(scores, acceptance=0.95):
    """Higher scores mean unfamiliar. Ties pass; empirical acceptance is >= target."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < acceptance <= 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores or acceptance target")
    return float(np.quantile(scores, acceptance, method="higher"))

# 최대 확률의 여집합(MSP)과 가까운 학습 특징 k개의 평균 코사인 거리를 반환합니다.
def unfamiliarity_scores(probabilities, embeddings, reference, k=5):
    reference = np.asarray(reference, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not len(reference) or not np.isfinite(embeddings).all():
        raise ValueError("Missing reference embeddings or invalid input")
    count = min(k, len(reference))
    if count < 1:
        raise ValueError("k must be positive")
    # All embeddings are L2-normalized by the encoder.
    # 정규화된 벡터의 코사인 거리는 0에 가까울수록 학습 사진과 비슷합니다.
    distance = np.clip(1 - embeddings @ reference.T, 0, 2)
    # 전체 정렬 대신 가까운 k개만 골라 평균하여 계산을 줄입니다.
    nearest = np.partition(distance, count - 1, axis=1)[:, :count].mean(axis=1)
    return {"msp": 1 - np.max(probabilities, axis=1), "embedding": nearest}

# 정답 과일별 보정 사진의 95%가 종류까지 맞히고 통과하도록 임계값을 계산합니다.
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
            # 종류 오답도 5% 오류 허용량을 소모하므로 종류 정답 사진만의 95%로 계산하지 않습니다.
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
