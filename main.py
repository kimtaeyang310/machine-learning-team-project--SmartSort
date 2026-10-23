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
    print("1. 모델 학습 (기존 5개 + YOLOv8n-cls)")
    print("2. 모델 평가")
    print("3. 폴더 전체 숙도 예측 및 시간 측정")
    print("4. 과일 종류 학습·전처리 비교 (개체 분리 / 과일별 95% 통과)")
    print("5. 과일 종류·모르는 과일 평가 (저장된 기준 사용)")
    print("0. 종료")
    print("=" * 50)

    choice = input("실행할 작업을 선택하세요: ").strip()

    if choice == "1":
        from src.train import main as train_main

        train_main([])

    elif choice == "2":
        from src.evaluate import evaluate

        evaluate()

    elif choice == "3":
        from datetime import datetime, timezone
        from src.predict import predict_folder, PREDICT_DIR, save_prediction_artifacts

        folder = input("예측 폴더 경로 (Enter: data/predict_list): ").strip().strip('"')

        source = folder or PREDICT_DIR
        results, _ = predict_folder(source)
        target = Path(__file__).resolve().parent / "outputs" / "predictions" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv")
        csv_path, manifest = save_prediction_artifacts(results, target, source_folder=source)
        print(f"결과: {csv_path}\n실행 정보: {manifest}")

    elif choice == "4":
        from src.train_fruit_classifier import train
        train()

    elif choice == "5":
        from src.evaluate_fruit_classifier import evaluate as evaluate_fruit
        evaluate_fruit()

    elif choice == "0":
        print("프로그램을 종료합니다.")

    else:
        print("잘못된 선택입니다. 0~5 중에서 선택하세요.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n작업 중단:\n{error}")
        raise SystemExit(1)
