"""예측 결과에 정책 판단을 적용하고 두 단계의 결과를 묶습니다."""

from __future__ import annotations

from typing import TypedDict

from src.agent.decision import Decision, PredictionInput, decide_next_action


class AgentResult(TypedDict):
    """호출자에게 돌려주는 예측 결과와 판단 결과의 묶음."""

    prediction: PredictionInput
    decision: Decision


def run_agent(prediction: PredictionInput) -> AgentResult:
    """예측 결과 하나를 입력받아 판단을 실행하고 두 결과를 묶어 반환합니다.

    모델 실행과 이미지 처리는 하지 않습니다. ``prediction``은 모델 단계에서
    이미 만들어진 값이며, 판단 단계에는 그 값만 전달합니다.
    """
    decision = decide_next_action(prediction)
    return {"prediction": prediction, "decision": decision}
