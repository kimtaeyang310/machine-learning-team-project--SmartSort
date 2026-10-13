#8. main.py 기능들을 순서대로 호출하는 실행 진입점


"""SmartSort - 메인 실행 프로그램.

실행:
    python main.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """사용자가 선택한 작업을 실행합니다."""

    print()
    print("=" * 50)
    print("SmartSort - 농산물 숙도 판별 시스템")
    print("=" * 50)
    print("1. 모델 학습")
    print("2. 모델 평가")
    print("3. 새 이미지 숙도 예측")
    print("0. 종료")
    print("=" * 50)

    choice = input("실행할 작업을 선택하세요: ").strip()

    if choice == "1":
        from src.train import main as train_main

        train_main()

    elif choice == "2":
        from src.evaluate import evaluate

        evaluate()

    elif choice == "3":
        from src.predict import predict

        image_path = input(
            "예측할 이미지 경로를 입력하세요: "
        ).strip().strip('"')

        if not image_path:
            print("이미지 경로를 입력하지 않았습니다.")
            return

        fruit = input("과일 종류 (banana=바나나 / mandarin=귤 / tomato=토마토): ").strip().lower()
        fruit = {"바나나": "banana", "귤": "mandarin", "토마토": "tomato"}.get(fruit, fruit)
        predict(image_path, fruit=fruit)

    elif choice == "0":
        print("프로그램을 종료합니다.")

    else:
        print("잘못된 선택입니다. 0~3 중에서 선택하세요.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n작업 중단:\n{error}")
        raise SystemExit(1)
