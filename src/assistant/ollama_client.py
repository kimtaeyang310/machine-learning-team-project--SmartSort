"""Ollama 대화와 허용된 SmartSort 작업 실행을 연결합니다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ollama import chat

from src.assistant.tools import (
    ALLOWED_ACTIONS,
    ALLOWED_FRUITS,
    execute_smartfarm_task,
)


MODEL_NAME = "qwen3:4b"

FRUIT_NAMES = {
    "banana": "바나나",
    "mandarin": "감귤",
    "tomato": "토마토",
}

RIPENESS_NAMES = {
    "unripe": "미숙",
    "ripe": "적숙",
    "overripe": "과숙",
}

SYSTEM_MESSAGE = """
당신은 SmartSort의 한국어 비서입니다.

할 수 있는 작업은 다음과 같습니다.
- 예측 폴더 이미지 분석
- 최근 예측 결과 요약
- 모르는 과일 이미지 목록
- 현재 모델과 판정 기준 설명

작업이 필요하면 smartfarm_task 도구만 사용하세요.
한 번의 요청에서 도구 호출은 최대 한 번만 하세요.
도구 실행 결과에 없는 수치나 파일 경로를 만들어 말하지 마세요.
정답 라벨이 없으면 정확도나 F1을 계산했다고 말하지 마세요.

폴더 분석 요청에 사용할 폴더가 정해지지 않았으면 먼저 물어보세요.
기본 폴더 또는 사용자가 폴더 선택 창에서 고른 폴더만 사용할 수 있습니다.
사용자가 폴더를 선택하지 않았고 기본 폴더 사용도 확인하지 않았다면 분석 도구를 실행하지 마세요.

재학습, 판정 기준 변경, 사진 수정·이동·삭제, 임의 코드나 터미널 명령 실행은 지원하지 않습니다.
"""

SMARTFARM_TOOL = {
    "type": "function",
    "function": {
        "name": "smartfarm_task",
        "description": (
            "SmartSort의 허용된 작업 하나를 실행합니다. "
            "과일을 생략하거나 all로 지정하면 전체 과일을 대상으로 합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(ALLOWED_ACTIONS),
                    "description": "실행할 SmartSort 작업",
                },
                "fruit": {
                    "type": "string",
                    "enum": ["all", "banana", "mandarin", "tomato"],
                    "description": (
                        "집계할 과일. 생략하거나 all이면 전체 과일입니다."
                    ),
                },
            },
            "required": ["action"],
        },
    },
}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "analyze_folder",
                "summarize_recent",
                "list_unknown",
                "explain_model",
                "list_supported_fruits",
                "chat",
                "clarify",
            ],
        },
        "fruit_mentions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "scope": {
            "type": "string",
            "enum": ["specific", "all", "unspecified"],
        },
        "answer": {"type": "string"},
    },
    "required": ["action", "fruit_mentions", "scope", "answer"],
}


INTENT_SYSTEM_MESSAGE = """
당신은 SmartSort의 요청 분류기입니다.
가장 최근 사용자 메시지만 보고 아래 JSON 형식으로 분류하세요.
작업을 실행하지 마세요.

action의 의미:
- analyze_folder: 이미지 숙도 예측 실행 요청
- summarize_recent: 최근 예측 결과, 처리 속도 또는 FPS 조회
- list_unknown: 최근 모르는 과일 이미지 목록 조회
- explain_model: 모델이나 판정 기준 질문
- list_supported_fruits: 측정 가능한 과일 종류 질문
- chat: SmartSort 관련 일반 대화
- clarify: 요청의 뜻이 분명하지 않아 확인이 필요한 경우

규칙:
- 과일 이름이 나오면 fruit_mentions에 원문 그대로 모두 넣으세요.
  포도, 레몬처럼 지원되지 않는 과일 이름도 빼면 안 됩니다.
- 과일 이름이 하나 이상 나오면 scope는 specific입니다.
- 사용자가 전체 과일 또는 모든 사진이라고 명시했을 때만 scope는 all입니다.
- 과일도 전체 범위도 지정하지 않았다면 scope는 unspecified입니다.
- “측정할 수 있는 과일 항목” 질문은 list_supported_fruits입니다.
- “처리 속도”나 “FPS” 질문은 새 분석이 아니라 summarize_recent입니다.
- chat이면 answer에 한국어 답변을 작성하세요.
- 전체, 모두, 전부, 모든 과일은 과일 이름이 아니라 분석 범위입니다.
  이 단어들을 fruit_mentions에 넣지 마세요.
- "전체 과일 숙도 분석해줘"는 다음처럼 분류하세요:
  action=analyze_folder, fruit_mentions=[], scope=all
- "포도 숙도 분석해줘"는 다음처럼 분류하세요:
  action=analyze_folder, fruit_mentions=["포도"], scope=specific
"""

def _classify_intent(text: str, model_name: str) -> dict[str, Any]:
    """현재 메시지를 구조화된 작업 의도로 분류합니다."""
    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_MESSAGE},
            {"role": "user", "content": text},
        ],
        format=INTENT_SCHEMA,
        think=False,
        options={"temperature": 0},
        stream=False,
    )

    content = response.message.content or "{}"
    result = json.loads(content)

    if not isinstance(result, dict):
        raise ValueError("요청 분류 결과가 올바른 JSON 객체가 아닙니다.")

    return result


def _format_task_result(action: str, result: dict[str, Any]) -> str:
    """구조화된 작업 결과를 수치를 보존한 한국어 답변으로 만듭니다."""
    if not result.get("ok"):
        return (
            "요청을 완료하지 못했습니다: "
            f"{result.get('error', '원인을 확인할 수 없습니다.')}"
        )

    if action in {"analyze_folder", "summarize_recent"}:
        lines = ["예측 결과를 정리했어요."]

        counts = result.get("ripeness_counts", {})
        fruit_filter = result.get("fruit_filter", "all")

        if fruit_filter == "all":
            for fruit, stages in counts.items():
                fruit_name = FRUIT_NAMES.get(fruit, fruit)
                stage_text = " · ".join(
                    f"{RIPENESS_NAMES.get(stage, stage)} {count}장"
                    for stage, count in stages.items()
                )
                lines.append(f"{fruit_name}: {stage_text}")
        else:
            fruit_name = FRUIT_NAMES.get(fruit_filter, fruit_filter)
            stage_text = " · ".join(
                f"{RIPENESS_NAMES.get(stage, stage)} {count}장"
                for stage, count in counts.items()
            )
            lines.append(f"{fruit_name}: {stage_text}")

        lines.append(f"모르는 과일 {result.get('unknown_count', 0)}장")
        lines.append(f"처리 실패 {result.get('failure_count', 0)}장")

        performance = result.get("performance", {})
        fps = performance.get("fps")
        if fps is not None:
            lines.append(f"처리 속도 {float(fps):.1f} FPS")

        retake_images = result.get("retake_images", [])
        lines.append(f"재촬영 권고 {len(retake_images)}장")

        for item in retake_images:
            fruit_name = FRUIT_NAMES.get(
                item.get("fruit"),
                item.get("fruit") or "과일",
            )
            stage_name = RIPENESS_NAMES.get(
                item.get("ripeness"),
                item.get("ripeness") or "숙도 미상",
            )

            confidence = item.get("confidence")
            confidence_text = ""
            if confidence is not None:
                confidence_text = f", 확률 {float(confidence):.1%}"

            lines.append(
                f"- {item.get('image', '이미지 이름 미상')}: "
                f"{fruit_name} {stage_name}{confidence_text}"
            )

        if retake_images:
            lines.append(
                "재촬영 권고 사진은 조명과 구도를 확인해 다시 촬영한 뒤 "
                "분석해 주세요."
            )

        csv_path = result.get("csv_path")
        if csv_path:
            lines.append(f"결과 CSV: {csv_path}")

        return "\n".join(lines)

    if action == "list_unknown":
        images = result.get("images", [])
        lines = [
            f"최근 결과에서 모르는 과일은 {result.get('unknown_count', 0)}장입니다."
        ]

        for item in images:
            candidate = item.get("candidate_class") or "후보 정보 없음"
            lines.append(f"- {item.get('image', '이미지 이름 미상')}: {candidate}")

        if not images:
            lines.append("표시할 모르는 과일 이미지가 없습니다.")

        return "\n".join(lines)

    if action == "explain_model":
        model_name = result.get("model_name") or "정보 없음"
        image_size = result.get("image_size") or "정보 없음"
        threshold = result.get("agent_confidence_threshold")

        lines = [
            f"모델: {model_name}",
            f"입력 이미지 크기: {image_size}",
        ]

        if threshold is not None:
            lines.append(f"숙도 자동 분류 확률 기준: {float(threshold):.0%}")

        lines.append(str(result.get("note", "")))
        return "\n".join(line for line in lines if line)

    return json.dumps(result, ensure_ascii=False, indent=2)


def _resolve_fruit_scope(action, fruit_mentions, normalized_text):
    """과일 이름과 전체 범위를 검증합니다. 실행할 수 없는 요청은 기존 안내 문구로 거부합니다."""
    aliases = {
        "바나나": "banana",
        "banana": "banana",
        "감귤": "mandarin",
        "귤": "mandarin",
        "mandarin": "mandarin",
        "토마토": "tomato",
        "tomato": "tomato",
    }

    supported_fruits = []
    unsupported_fruits = []

    scope_words = {
        "전체",
        "전체 과일",
        "모든 과일",
        "모든",
        "모두",
        "전부",
        "all",
        "all fruits",
    }

    for raw_name in fruit_mentions:
        name = raw_name.strip().lower()

        if not name or name in scope_words:
            continue

        canonical = aliases.get(name)

        if canonical not in ALLOWED_FRUITS:
            unsupported_fruits.append(raw_name.strip())
        elif canonical not in supported_fruits:
            supported_fruits.append(canonical)

    # 미지원 과일이 명시되면 폴더 확인이나 YOLO 실행 전에 종료합니다.
    if unsupported_fruits:
        names = ", ".join(unsupported_fruits)
        raise ValueError(
            f"{names}은 현재 숙도 판별 지원 항목에 없어 "
            "예측을 실행하지 않았어요. "
            "지원하는 과일은 바나나, 감귤, 토마토입니다."
        )

    fruit = None
    all_fruits = False

    if len(supported_fruits) == 1:
        fruit = supported_fruits[0]

    elif len(supported_fruits) > 1:
        # 현재 도구는 과일 하나 또는 세 종류 전체 집계를 지원합니다.
        if set(supported_fruits) == ALLOWED_FRUITS:
            fruit = None
            all_fruits = action == "analyze_folder"
        else:
            raise ValueError(
                "현재는 과일 하나 또는 전체 과일을 대상으로 처리할 수 있어요. "
                "과일 하나를 지정하거나 “전체 과일”이라고 말씀해 주세요."
            )

    elif action == "analyze_folder":
        # 분류기의 scope뿐 아니라 사용자 원문도 함께 확인합니다.
        all_scope_phrases = (
            "전체 과일",
            "모든 과일",
            "전체 사진",
            "모든 사진",
            "폴더 전체",
            "전부 분석",
            "모두 분석",
        )

        explicit_all = any(
            phrase in normalized_text
            for phrase in all_scope_phrases
        )

        if explicit_all:
            all_fruits = True
        else:
            raise ValueError(
                "어떤 과일을 분석할까요? "
                "바나나, 감귤, 토마토 중 하나를 지정하거나 "
                "“전체 과일”이라고 말씀해 주세요."
            )
    return fruit, all_fruits


class OllamaAssistant:
    """요청 분류, 지원 과일 검사, 폴더 확인과 작업 실행을 관리합니다."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.messages: list[Any] = []

        # 작업명, 과일, 전체 분석 승인 여부를 함께 기억합니다.
        self.pending_intent: tuple[str, str | None, bool] | None = None

    def reset(self) -> None:
        """대화 기록과 대기 중인 작업을 초기화합니다."""
        self.messages = []
        self.pending_intent = None

    def _reply(self, answer: str) -> str:
        """답변을 대화 기록에 추가하고 반환합니다."""
        self.messages.append({
            "role": "assistant",
            "content": answer,
        })
        return answer

    def _run_task(
        self,
        action: str,
        fruit: str | None,
        *,
        selected_folder: str | Path | None,
        all_fruits: bool = False,
    ) -> str:
        """검증된 작업을 실행하고 Python이 결과 문구를 만듭니다."""
        try:
            result = execute_smartfarm_task(
                action=action,
                fruit=fruit,
                selected_folder=selected_folder,
                all_fruits=all_fruits,
            )
            return _format_task_result(action, result)
        except Exception as error:
            return f"작업 중 오류가 발생했습니다: {error}"

    def ask(
        self,
        user_text: str,
        *,
        selected_folder: str | Path | None = None,
        default_folder_confirmed: bool = False,
    ) -> str:
        """현재 메시지를 분류하고 지원 여부를 검사한 뒤 실행합니다."""
        user_text = user_text.strip()
        if not user_text:
            return "메시지를 입력해 주세요."

        self.messages.append({
            "role": "user",
            "content": user_text,
        })

        normalized_text = " ".join(user_text.lower().split())

        confirmations = {
            "네",
            "예",
            "응",
            "맞아",
            "진행해",
            "진행해줘",
            "기본 폴더로",
            "기본 폴더로 해",
            "기본 폴더로 진행해",
            "이 폴더로 진행해",
        }

        # 폴더 확인에 대한 응답이면 기존 요청을 이어서 실행합니다.
        if (
            self.pending_intent is not None
            and normalized_text in confirmations
            and (
                selected_folder is not None
                or default_folder_confirmed
            )
        ):
            action, fruit, all_fruits = self.pending_intent
            self.pending_intent = None

            return self._reply(
                self._run_task(
                    action,
                    fruit,
                    selected_folder=selected_folder,
                    all_fruits=all_fruits,
                )
            )

        # 다른 질문이 들어오면 이전 폴더 확인 대기 작업은 해제합니다.
        self.pending_intent = None

        try:
            intent = _classify_intent(
                user_text,
                self.model_name,
            )
        except Exception as error:
            return self._reply(
                "요청 내용을 분류하지 못했어요. "
                f"다시 말씀해 주세요. 자세한 내용: {error}"
            )

        action = intent.get("action")
        fruit_mentions = intent.get("fruit_mentions")
        scope = intent.get("scope")
        answer_text = intent.get("answer", "")

        # 모델이 반환한 JSON의 값도 프로그램에서 검사합니다.
        if (
            not isinstance(action, str)
            or not isinstance(fruit_mentions, list)
            or not all(
                isinstance(name, str)
                for name in fruit_mentions
            )
            or scope not in {"specific", "all", "unspecified"}
            or not isinstance(answer_text, str)
        ):
            return self._reply(
                "요청 분류 결과가 올바르지 않아 실행하지 않았어요. "
                "요청을 다시 말씀해 주세요."
            )

        # 지원 과일 목록 질문은 이미지 예측을 실행하지 않습니다.
        if action == "list_supported_fruits":
            return self._reply(
                "현재 숙도 측정이 가능한 과일은 "
                "바나나, 감귤, 토마토입니다. "
                "포도와 레몬 등 다른 과일의 숙도 판별은 지원하지 않습니다."
            )

        if action in {"chat", "clarify"}:
            return self._reply(
                answer_text.strip()
                or "어떤 작업을 원하시는지 조금 더 자세히 말씀해 주세요."
            )

        if action not in ALLOWED_ACTIONS:
            return self._reply(
                "지원하지 않는 작업이라 실행하지 않았어요."
            )

        try:
            fruit, all_fruits = _resolve_fruit_scope(action, fruit_mentions, normalized_text)
        except ValueError as error:
            return self._reply(str(error))

        # 최근 결과 조회 등은 과일을 생략하면 전체 결과를 조회합니다.
        # 새 분석 작업에서만 폴더 승인이 필요합니다.
        if (
            action == "analyze_folder"
            and selected_folder is None
            and not default_folder_confirmed
        ):
            self.pending_intent = (
                action,
                fruit,
                all_fruits,
            )

            return self._reply(
                "분석할 폴더를 선택하거나, 기본 예측 폴더를 사용할지 "
                "확인해 주세요."
            )

        return self._reply(
            self._run_task(
                action,
                fruit,
                selected_folder=selected_folder,
                all_fruits=all_fruits,
            )
        )