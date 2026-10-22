"""예측 신뢰도와 취약 클래스를 바탕으로 다음 선별 행동을 결정합니다."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


DEFAULT_CONFIDENCE_THRESHOLD = 0.70
# 현재 검증 결과에서 토마토 과숙 분류는 취약하므로 사람 확인을 우선합니다.
REVIEW_REQUIRED_CLASSES = {("tomato", "overripe")}


class PredictionInput(TypedDict, total=False):
    """판단에 필요한 예측 결과 필드. 다른 예측 메타데이터는 무시합니다."""

    fruit: str
    ripeness: str
    probabilities: dict[str, float]
    image_path: str
    model: str


class Decision(TypedDict):
    """정책 판단만 담는 고정 반환 형태. 확정 결과 필드는 포함하지 않습니다."""

    action: Literal["auto_classify", "human_review", "retake"]
    reason: str
    suggested_label: str | None
    confidence: float | None
    requires_confirmation: bool


class FinalizedDecision(Decision):
    """사용자 또는 정책에 따라 라벨까지 확정한 결과 형태."""

    final_label: str
    finalized_by: Literal["model", "user"] | None


def decide_next_action(
    prediction: PredictionInput,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Decision:
    """모델 예측을 정책에 따라 다음 작업 제안으로 변환합니다.

    입력에는 과일(``fruit``), 예측 숙도(``ripeness``), 숙도별 점수
    (``probabilities``)가 들어옵니다. 예측 숙도에 해당하는 점수를
    ``confidence``로 사용합니다.

    판단 순서는 다음과 같습니다.

    1. 검증상 취약한 과일·숙도 조합이면 사람 확인을 요청합니다.
    2. 예측 확률을 얻지 못하면 사람 확인을 요청합니다.
    3. 예측 숙도의 확률이 기준 미만이면 재촬영을 권고합니다.
    4. 위 조건이 없으면 자동 분류를 제안합니다.

    반환값에는 action, reason, suggested_label, confidence,
    requires_confirmation만 들어갑니다. 저장이나 사용자 확정은 여기서 하지 않습니다.
    ``retake``는 낮은 확률만으로 내리는 권고이며, 이미지 품질을 직접
    검사하거나 재촬영 효과를 보장하지 않습니다.
    """
    # 범위를 벗어난 설정값이 잘못된 분기를 만들지 않도록 먼저 확인합니다.
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold는 0과 1 사이여야 합니다.")

    fruit = prediction.get("fruit")
    ripeness = prediction.get("ripeness")
    probabilities = prediction.get("probabilities")
    confidence = _predicted_class_confidence(probabilities, ripeness)

    # 기본 제안은 자동 분류이며, 아래 위험 조건에 해당하면 덮어씁니다.
    result: Decision = {
        "action": "auto_classify",
        "reason": "예측 결과가 자동 분류 기준을 충족했습니다.",
        "suggested_label": ripeness,
        "confidence": confidence,
        "requires_confirmation": False,
    }

    # 취약 클래스 예외를 확률 검사보다 먼저 적용해 자동 확정을 막습니다.
    if (fruit, ripeness) in REVIEW_REQUIRED_CLASSES:
        result.update(
            action="human_review",
            reason="토마토 과숙 분류는 검증 성능이 낮아 사람 확인이 필요합니다.",
            requires_confirmation=True,
        )
    elif confidence is None:
        result.update(
            action="human_review",
            reason="예측 확률을 확인할 수 없어 자동 분류하지 않습니다.",
            requires_confirmation=True,
        )
    elif confidence < confidence_threshold:
        # 기준과 같은 확률은 통과하고, 기준 미만일 때만 재촬영을 권고합니다.
        result.update(
            action="retake",
            reason=(
                f"최고 예측 확률이 {confidence:.1%}로 기준 "
                f"{confidence_threshold:.0%}보다 낮습니다. 조명과 구도를 확인해 재촬영해 주세요."
            ),
            requires_confirmation=True,
        )

    return result


def _predicted_class_confidence(
    probabilities: Any,
    ripeness: Any,
) -> float | None:
    """입력 확률표에서 예측 라벨의 점수만 꺼냅니다. 사용할 수 없으면 None입니다."""
    if not isinstance(probabilities, dict) or ripeness not in probabilities:
        return None
    try:
        return float(probabilities[ripeness])
    except (TypeError, ValueError):
        return None


def finalize_decision(
    decision: Decision,
    confirmed_ripeness: str | None = None,
) -> FinalizedDecision:
    """확정 라벨과 확정 주체를 결과 딕셔너리에 반영해 반환합니다.

    ``confirmed_ripeness``가 주어지면 사용자 선택값으로 확정합니다. 생략하면
    ``auto_classify`` 결과만 모델 예측값으로 확정할 수 있습니다. 확인이나
    재촬영이 필요한 결과는 사용자 확인값 없이 확정할 수 없습니다.

    이 함수는 결과만 반환하며 파일이나 데이터베이스에는 저장하지 않습니다.
    """
    # 입력 결정을 덮어쓰지 않고, 동일한 고정 형태의 새 결과를 반환합니다.
    finalized = dict(decision)
    if confirmed_ripeness is not None:
        finalized.update(
            final_label=confirmed_ripeness,
            finalized_by="user",
            requires_confirmation=False,
        )
    elif decision.get("action") == "auto_classify":
        finalized.update(
            final_label=decision.get("suggested_label"),
            finalized_by="model",
            requires_confirmation=False,
        )
    else:
        raise ValueError("확인이 필요한 예측입니다. 사용자가 확인한 숙도 라벨을 전달하세요.")
    return finalized
