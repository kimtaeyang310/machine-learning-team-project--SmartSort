# SmartSort — 통합 과일·숙도 분류

기본 방식은 **YOLOv8n-cls 한 개 + 특징 거리 기반 모름 판정**입니다.
바나나·귤·토마토 각각 미숙/적숙/과숙의 총 9클래스를 한 번에 예측합니다.
YOLO의 분류 직전 특징을 학습 기준 특징들과 비교하고 과일별 거리 기준을 넘으면
**모르는 과일**로 표시하며 숙도 결과는 숨깁니다. 종류 분류용 MobileNet이나 별도 숙도 신경망은 로딩하지 않습니다.

## 실행

프로젝트 최상위 폴더의 PowerShell에서:

```powershell
& C:/Users/user1/anaconda3/envs/teamTC/python.exe main.py
```

- 1번: 통합 모델 6 epochs 재학습 + Calibration·Selection으로 모름 기준 설정. 성공하면 이전 모델을 outputs/joint_nine에 보관하고 교체합니다.
- 2번: 저장된 기준으로 9클래스 Macro F1, 과일별 숙도 F1, 오거부·오수락률과 FPS 평가.
- 3번: 폴더 사진 예측. Enter는 data/predict_list. 결과 CSV·실행 정보는 outputs/predictions에 저장.
- 4번: 현재 모델 출처, 기준 및 기준 특징 수 확인.
- 5번: 이전 원본 보관 위치 안내.

```powershell
# 폴더 예측
& C:/Users/user1/anaconda3/envs/teamTC/python.exe -m src.predict
# 평가만 실행
& C:/Users/user1/anaconda3/envs/teamTC/python.exe -m src.joint evaluate
# 재학습 (선택 사항)
& C:/Users/user1/anaconda3/envs/teamTC/python.exe -m src.joint train --epochs 6
```

## 채택 모델과 측정

models/joint_nine에 model.pt, policy.joblib, split_manifest.json, bundle.json을 함께 저장합니다.
채택 출처는 experiments/nine_class_comparison/artifacts/20261023T071957744434Z입니다.
실험 당시 통합 방식은 39.70 FPS였으며 실행 환경에 따라 달라집니다.
FPS에는 이미지 읽기·전처리·YOLO·모름 거리 판정을 포함합니다. 모델 로딩·워밍업·에이전트 정책·출력·저장은 제외합니다.
이전 보고서 40.43 FPS는 최초 실험 측정값입니다.

과일별 Calibration 사진의 약 95% 종류 정답·수락 목표를 사용하고,
Selection의 포도·레몬 오수락을 우선 줄이는 기준을 선택합니다.
Test는 이미 비교에 사용한 자료이므로 회귀 평가입니다. 실제 운영에서 과일별 95% 통과를 보장하지는 않습니다.
현재 재학습은 검증된 manifest에 기록된 기존 사진·개체 분할을 재사용하며 변경된 사진은 중단합니다.
새 자료 추가는 개체 ID를 포함한 새 분할 설계가 필요합니다.
에이전트는 9클래스 중 예측 클래스의 확률을 사용하며 과일 내부 확률로 다시 정규화하지 않습니다.

## 이전 원본 보관 및 복원

archive/before_joint_nine_20261023에 전환 전 코드·데이터·모델·기존 학습 체크포인트가 있습니다.
archive_manifest.json은 3,085개 원본 파일의 SHA256 기록입니다. 보관 폴더로 이동한 뒤 teamTC Python으로 main.py를 실행할 수 있습니다.
가상환경과 캐시는 공용 환경에서 재생성합니다. 과거 outputs와 experiments는 현재 위치에 계속 보존됩니다.
자세한 보관 범위는 보관 폴더 RESTORE.md를 확인하세요.
이전 소스와 모델을 복원할 때는 현재 폴더도 먼저 별도로 보관한 뒤 archive_manifest.json의 상대 경로대로 복사하세요.

src/train.py, src/evaluate.py 및 과일 종류 전용 모듈은 기존 실험·호환 호출을 위해 유지한 이전 구현입니다.
새 기본 학습·평가는 main.py 또는 src.joint를 사용하세요.
폴더 예측의 --fruit-mode image/filename은 이전 방식 명시적 호환 옵션이며 기본값은 joint입니다.

## 출처

- [Mendeley Data 데이터셋 (버전 2)](https://data.mendeley.com/datasets/ptfscwtnyz/2)
- [Kaggle — Fruit Ripeness: Unripe, Ripe and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)
- [Kaggle — Riped and Unriped Tomato Dataset](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset)
- [universe.roboflow — Tomatos Ripeness Computer Vision Dataset](https://universe.roboflow.com/models/object-detection)

