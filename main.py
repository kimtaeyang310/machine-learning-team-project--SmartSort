#8. main.py 기능들을 순서대로 호출하는 실행 진입점


"""SmartSort - 메인 실행 프로그램.

실행:
    python main.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """사용자가 선택한 작업을 실행합니다."""

    print()
    print("=" * 50)
    print("SmartSort - 농산물 숙도 판별 시스템")
    print("=" * 50)
    print("1. 통합 9클래스 YOLO 학습 및 모름 기준 설정")
    print("2. 통합 모델 평가 (F1·오거부·오수락·FPS)")
    print("3. 폴더 전체 숙도 예측 및 시간 측정")
    print("4. 현재 통합 모델 및 모름 판정 기준 확인")
    print("5. 이전 원본 보관 위치 및 실행 방법")
    print("6. 선별 테스트 과일 시연")
    print("0. 종료")
    print("=" * 50)

    choice = input("실행할 작업을 선택하세요: ").strip()

    # 선택한 메뉴에 필요한 기능을 불러옵니다. 실제 학습·평가·예측 구현은 src에 있습니다.
    if choice == "1":
        from src.joint import train
        train()

    elif choice == "2":
        from src.joint import evaluate

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
        import json
        from src.joint import MODEL_DIR
        import joblib
        print((MODEL_DIR / "bundle.json").read_text(encoding="utf-8"))
        policy = joblib.load(MODEL_DIR / "policy.joblib")
        print(json.dumps({"method": policy["method"], "thresholds": policy["thresholds"],
                          "reference_count": len(policy["reference"])}, ensure_ascii=False, indent=2))

    elif choice == "5":
        print("이전 원본: archive/before_joint_nine_20261023")
        print("해당 폴더로 이동한 뒤 기존 teamTC Python으로 main.py를 실행하세요.")
        print("복원 및 보관 내용: archive/before_joint_nine_20261023/RESTORE.md")

    elif choice == "6":
        from src.presentation import run_presentation
        run_presentation(display=True)

    elif choice == "0":
        print("프로그램을 종료합니다.")

    else:
        print("잘못된 선택입니다. 0~6 중에서 선택하세요.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n작업 중단:\n{error}")
        raise SystemExit(1)
