"""SmartSort Ollama 비서의 채팅 창과 폴더 선택 UI입니다."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from src.assistant.ollama_client import OllamaAssistant
from src.assistant.tools import PREDICT_DIR


SUPPORTED_IMAGE_TYPES = (
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp",
    "*.tif",
    "*.tiff",
)

DEFAULT_CONFIRMATIONS = {
    "네",
    "예",
    "응",
    "맞아",
    "기본 폴더",
    "기본 폴더로",
    "기본 예측 폴더",
    "기본 예측 폴더로",
    "기본 폴더로 진행해",
    "기본 폴더로 해",
}


class AssistantWindow:
    """OllamaAssistant를 사용하는 Tkinter 채팅 창입니다."""

    def __init__(
        self,
        root: tk.Tk | None = None,
        assistant: OllamaAssistant | None = None,
    ) -> None:
        self.root = root or tk.Tk()
        self.assistant = assistant or OllamaAssistant()

        self.selected_folder: Path | None = None
        self.use_default_next = False
        self.waiting_for_default_confirmation = False
        self.busy = False

        self.root.title("SmartSort 비서")
        self.root.geometry("760x620")
        self.root.minsize(560, 420)

        self._build_ui()
        self._show_message(
            "assistant",
            "안녕하세요. SmartSort 예측 결과를 확인하거나 "
            "모델 정보를 물어볼 수 있어요.\n"
            "사진 분석 폴더가 불분명하면 먼저 폴더를 확인할게요.",
        )
        self._update_folder_label()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        folder_frame = ttk.Frame(self.root, padding=(10, 10, 10, 4))
        folder_frame.grid(row=0, column=0, sticky="ew")
        folder_frame.columnconfigure(0, weight=1)

        self.folder_label = ttk.Label(
            folder_frame,
            text="분석 폴더: 선택되지 않음",
            anchor="w",
        )
        self.folder_label.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.choose_folder_button = ttk.Button(
            folder_frame,
            text="폴더 선택",
            command=self._choose_folder,
        )
        self.choose_folder_button.grid(row=0, column=1, padx=3)

        self.default_folder_button = ttk.Button(
            folder_frame,
            text="기본 폴더 사용",
            command=self._use_default_folder,
        )
        self.default_folder_button.grid(row=0, column=2, padx=3)

        self.clear_folder_button = ttk.Button(
            folder_frame,
            text="선택 해제",
            command=self._clear_folder,
        )
        self.clear_folder_button.grid(row=0, column=3, padx=3)

        chat_frame = ttk.Frame(self.root, padding=(10, 4, 10, 8))
        chat_frame.grid(row=1, column=0, sticky="nsew")
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        self.chat_log = tk.Text(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("맑은 고딕", 10),
        )
        self.chat_log.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            chat_frame,
            orient="vertical",
            command=self.chat_log.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_log.configure(yscrollcommand=scrollbar.set)

        self.chat_log.tag_configure(
            "user",
            foreground="#1557a0",
            spacing1=8,
            spacing3=8,
        )
        self.chat_log.tag_configure(
            "assistant",
            foreground="#222222",
            spacing1=8,
            spacing3=8,
        )
        self.chat_log.tag_configure(
            "status",
            foreground="#666666",
            spacing1=4,
            spacing3=4,
        )

        input_frame = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)

        self.input_box = tk.Text(
            input_frame,
            height=3,
            wrap="word",
            font=("맑은 고딕", 10),
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.input_box.bind("<Control-Return>", self._send_from_shortcut)

        self.send_button = ttk.Button(
            input_frame,
            text="보내기",
            command=self._send,
        )
        self.send_button.grid(row=0, column=1, sticky="ns")

        self.status_label = ttk.Label(
            self.root,
            text="준비됨",
            anchor="w",
            padding=(10, 0, 10, 8),
        )
        self.status_label.grid(row=3, column=0, sticky="ew")

    def _show_message(self, speaker: str, message: str) -> None:
        self.chat_log.configure(state="normal")

        if speaker == "user":
            self.chat_log.insert("end", f"나: {message}\n\n", "user")
        elif speaker == "assistant":
            self.chat_log.insert(
                "end",
                f"SmartSort 비서: {message}\n\n",
                "assistant",
            )
        else:
            self.chat_log.insert("end", f"{message}\n", "status")

        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")

    def _update_folder_label(self) -> None:
        if self.selected_folder is not None:
            text = f"분석 폴더: {self.selected_folder}"
        elif self.use_default_next:
            text = f"다음 분석 폴더: 기본 폴더 ({PREDICT_DIR})"
        else:
            text = "분석 폴더: 미선택 — 요청이 모호하면 비서가 확인합니다."

        self.folder_label.configure(text=text)
    def _resume_pending_analysis(self, *, use_default_folder: bool) -> bool:
        """폴더 확인 대기 중인 분석을 백그라운드에서 이어서 실행합니다."""
        if self.busy or self.assistant.pending_intent is None:
            return False

        # 이번 실행에 기본 폴더 설정을 사용하므로 다음 요청용 표시는 해제합니다.
        self.use_default_next = False
        self.waiting_for_default_confirmation = False
        self._update_folder_label()

        self._set_busy(True)
        self.status_label.configure(text="선택한 폴더로 분석하고 있어요...")
        self._show_message(
            "status",
            "폴더가 확인되어 대기 중인 분석을 시작합니다.",
        )

        worker = threading.Thread(
            target=self._ask_in_background,
            args=(
                "네",
                self.selected_folder,
                use_default_folder,
            ),
            daemon=True,
        )
        worker.start()
        return True

    def _choose_folder(self) -> None:
        if self.busy:
            return

        folder = filedialog.askdirectory(
            title="분석할 사진 폴더 선택",
            mustexist=True,
        )

        # 선택 창에서 취소하면 대기 중인 요청을 유지합니다.
        if not folder:
            return

        self.selected_folder = Path(folder).resolve()
        self.use_default_next = False
        self.waiting_for_default_confirmation = False
        self._update_folder_label()

        # 대기 중인 분석이 있으면 선택한 폴더로 즉시 실행합니다.
        if self._resume_pending_analysis(use_default_folder=False):
            return

        self._show_message(
            "status",
            f"분석 폴더를 선택했어요: {self.selected_folder}\n"
            "채팅에 분석 요청을 입력해 주세요.",
        )

    def _use_default_folder(self) -> None:
        if self.busy:
            return

        self.selected_folder = None
        self.use_default_next = True
        self.waiting_for_default_confirmation = False
        self._update_folder_label()

        # 대기 중인 분석이 있으면 기본 폴더로 즉시 실행합니다.
        if self._resume_pending_analysis(use_default_folder=True):
            return

        self._show_message(
            "status",
            f"다음 분석 요청에는 기본 폴더를 사용합니다: {PREDICT_DIR}",
        )

    def _clear_folder(self) -> None:
        self.selected_folder = None
        self.use_default_next = False
        self.waiting_for_default_confirmation = False
        self._update_folder_label()
        self._show_message(
            "status",
            "폴더 선택을 해제했어요. 다음 분석 요청에서 폴더를 확인합니다.",
        )

    def _send_from_shortcut(self, _event: tk.Event) -> str:
        self._send()
        return "break"

    def _is_default_confirmation(self, text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        return normalized in DEFAULT_CONFIRMATIONS

    def _send(self) -> None:
        if self.busy:
            return

        user_text = self.input_box.get("1.0", "end").strip()
        if not user_text:
            return

        self.input_box.delete("1.0", "end")
        self._show_message("user", user_text)

        use_default_folder = self.use_default_next
        self.use_default_next = False

        if (
            self.waiting_for_default_confirmation
            and self._is_default_confirmation(user_text)
        ):
            use_default_folder = True

        self.waiting_for_default_confirmation = False
        self._update_folder_label()

        self._set_busy(True)
        self.status_label.configure(text="요청을 처리하고 있어요...")
        self._show_message("status", "요청을 처리하고 있어요...")

        selected_folder = self.selected_folder

        worker = threading.Thread(
            target=self._ask_in_background,
            args=(user_text, selected_folder, use_default_folder),
            daemon=True,
        )
        worker.start()

    def _ask_in_background(
        self,
        user_text: str,
        selected_folder: Path | None,
        use_default_folder: bool,
    ) -> None:
        try:
            answer = self.assistant.ask(
                user_text,
                selected_folder=selected_folder,
                default_folder_confirmed=use_default_folder,
            )
            error: Exception | None = None
        except Exception as exc:
            answer = ""
            error = exc

        self.root.after(
            0,
            self._finish_response,
            answer,
            error,
        )

    def _finish_response(
        self,
        answer: str,
        error: Exception | None,
    ) -> None:
        self._set_busy(False)

        if error is not None:
            answer = f"요청 처리 중 오류가 발생했습니다: {error}"

        self._show_message("assistant", answer)
        self.status_label.configure(text="준비됨")

        # OllamaAssistant가 기본 폴더 여부를 물으면 다음 답을 확인으로 처리합니다.
        if "분석할 폴더를 선택하거나, 기본 예측 폴더를 사용할지 확인해 주세요." in answer:
            self.waiting_for_default_confirmation = True

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"

        self.send_button.configure(state=state)
        self.choose_folder_button.configure(state=state)
        self.default_folder_button.configure(state=state)
        self.clear_folder_button.configure(state=state)

    def run(self) -> None:
        self.root.mainloop()


def launch_assistant() -> None:
    """비서 UI를 실행하는 진입 함수입니다."""
    window = AssistantWindow()
    window.run()