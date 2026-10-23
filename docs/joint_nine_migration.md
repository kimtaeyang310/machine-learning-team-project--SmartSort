# 통합 모델 전환 기록

2026-10-23 기존 기본 실행을 한 개의 YOLOv8n-cls + 임베딩 거리 모름 판정으로 전환했습니다.

- 이전 원본: `archive/before_joint_nine_20261023` (3,085개 파일의 SHA256 검증).
- 채택 실행: `experiments/nine_class_comparison/artifacts/20261023T071957744434Z`.
- 실행 모델: `models/joint_nine/model.pt`, `policy.joblib`, `split_manifest.json`, `bundle.json`.
- 원래 `fruit_classifier.joblib`, `ripeness_by_fruit.joblib`와 기존 실험 기록은 삭제·덮어쓰기하지 않았습니다.
- 배포용 코드는 `src/joint.py`, `src/joint_core.py`로 분리하여 실험 스크립트를 실행하거나 import하지 않습니다.

## 검증

채택 실행의 Test 538장에 대해 새 경로의 상태·9클래스 후보·확률을 비교했습니다.
상태와 후보 차이 0장, 최대 확률 차이 0입니다.
`outputs/joint_nine/evaluation_20261023T085914631316Z/promotion_parity.json`에 기록했습니다.
최신 채택 실행의 Macro F1은 0.6720, 오거부는 36/498 (7.23%), 포도·레몬 오수락은 0/40입니다.
최초 실험의 F1 0.6809·오거부 31/498과는 별도 재학습 실행입니다. 최신 실행을 기본으로 채택했습니다.
최신 실험 당시 FPS는 39.70이며 전환 검증에서는 37.37이었습니다.

main.py의 실제 3번 실행에서도 predict_list 47장 중 숙도 판정 43, 모름 4, 실패 0을 확인했습니다.
이 폴더는 큰 원본 사진 비중이 달라 17.72 FPS로 측정됐습니다. 538장 Test와 사진 구성이 달라 직접 속도 비교하면 안 됩니다.
FPS는 입력 해상도·파일 읽기·사진 구성·캐시·동시 실행 부하에 영향을 받습니다.

보관본 위치에서 기존 과일별 YOLO 모델 로딩과 실제 사진 숙도 예측도 확인했습니다.
보관 폴더의 RESTORE.md에 실행 및 복원 범위를 기록했습니다.

## 앞으로 실행

`main.py` 1번은 새 통합 모델 학습과 기준 설정, 2번은 통합 평가, 3번은 통합 폴더 예측입니다.
4번은 모델 출처와 모름 기준, 5번은 이전 원본 보관 위치를 보여줍니다.
`python -m src.predict`도 기본값은 통합 방식입니다.
`python -m src.joint train --epochs 6`, `python -m src.joint evaluate`도 지원합니다.

모름 판정 기준은 신경망이 아닌 기준 특징과 거리 규칙입니다. 모델·기준 특징·거리 기준을 한 묶음으로 관리해야 합니다.
학습은 기존 manifest의 개체 분할을 재사용하며 Test는 학습·기준 선택에 사용하지 않습니다.
새 사진을 자동으로 편입하지 않으므로 자료를 추가할 때는 별도 분할 갱신이 필요합니다.
학습 성공 시 이전 통합 모델은 outputs/joint_nine/previous_*에 복사합니다.
