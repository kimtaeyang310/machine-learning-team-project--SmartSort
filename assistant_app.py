"""SmartSort Ollama 비서 앱 실행 파일."""
# 비서 앱의 시작점이며 실제 화면과 대화 처리는 src.assistant에 있습니다.

from src.assistant.ui import launch_assistant


def main() -> None:
    launch_assistant()


if __name__ == "__main__":
    main()