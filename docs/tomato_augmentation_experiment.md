# 토마토 증강·클래스 가중치 비교 실험

## 목적

토마토 숙도 분류에서 `class_weight="balanced"`와 Train 전용 이미지 증강의 효과를 기존 SVM 기준선과 비교합니다. 기존 학습·평가 코드는 바꾸지 않고 별도 스크립트에서 실행합니다.

## 실행

프로젝트 루트(`machine-learning-team-project--SmartSort`)에서:

```powershell
python -m src.compare_tomato_augmentation
```

필요한 패키지는 프로젝트의 기존 `requirements.txt`와 같습니다. 데이터 라벨과 split은 `data/metadata/fruit_labels.csv` 및 `data/raw/`를 사용합니다. 실행 시각 폴더 아래에 결과가 저장됩니다.

```text
outputs/tomato_augmentation_comparison/<실행시각>/
├── summary.csv
├── fold_scores.csv
└── test_class_reports.json
```

## 비교 조건

| 조건 | 이미지 증강 | SVM 클래스 가중치 | 비교 목적 |
|---|---|---|---|
| `baseline_balanced` | 없음 | `balanced` | 현재 SVM 기준선 |
| `augmentation_balanced` | Train fold 안에서만 적용 | `balanced` | 권장 조합의 효과 |
| `augmentation_unweighted` | Train fold 안에서만 적용 | 없음 | 증강만 적용했을 때와 가중치의 차이 |

증강은 fold별 학습 표본에서 클래스별 이미지 수가 가장 큰 클래스의 수에 도달하도록 소수 클래스를 뽑아 수행합니다. 작은 회전·이동·확대, 좌우 반전, 약한 밝기·대비 변화를 사용하며 색조·채도는 변경하지 않습니다. 증강 수와 결과는 실행마다 재현 가능한 고정 seed를 사용합니다.

세 조건은 동일한 3-fold `StratifiedGroupKFold` 분할을 사용합니다. 증강 이미지는 해당 fold의 학습 쪽에만 생성합니다. 검증과 최종 Test에는 원본만 사용하고, Test는 학습 또는 설정 선택에 포함하지 않습니다. 최종 Test 지표는 모든 CV 조건을 실행한 뒤 각 조건별 전체 Train 재학습 모델에 대해 한 번 기록합니다.

## 결과 읽기

- `fold_scores.csv`: 조건과 fold별 Macro F1, 학습 fold 원본의 클래스 수, 검증 표본 수
- `summary.csv`: 조건별 CV Macro F1 평균·표준편차와 Test Accuracy·Macro F1
- `test_class_reports.json`: Test의 숙도별 Precision·Recall·F1·지원 수

우선 `cv_macro_f1_mean`을 보고 조건 간 차이를 확인하고, `cv_macro_f1_std`와 각 fold 점수도 함께 봅니다. 평균이 조금 높더라도 특정 fold의 성능이 크게 무너지거나 특정 숙도의 Recall이 악화되면 안정적인 개선으로 보기 어렵습니다. 현재 Test는 52장이지만 unripe 표본이 2장뿐이므로 Test 점수만으로 승자를 고르지 않습니다. 테스트 결과를 보고 증강 강도를 반복 조정하면 Test가 튜닝에 쓰인 셈이 됩니다.

## 해석상 한계

- 증강은 원본 이미지 수나 독립 과일 개체 수를 늘리지 않습니다. 새로운 개체·촬영일·조명에서의 일반화 개선을 보장하지 않습니다.
- 현재 클래스별 Train 표본 수가 작고 Test가 클래스별 2~3장이라 점수 불확실성이 큽니다.
- 이 스크립트는 프로젝트가 선택한 토마토 SVM과 특징 기반 파이프라인을 대상으로 합니다. YOLO 모델 비교는 포함하지 않습니다.
- 특징 기반 분류기는 증강 이미지에서 특징을 다시 계산합니다. 원본 특징 행을 복사해 오버샘플링하는 실험과 다릅니다.
- 증강 변환은 합리적인 초기 실험 범위입니다. 실제 촬영 조건과 맞지 않는다고 판단되는 변환이 있으면 결과를 채택하지 말고 범위를 조정한 별도 실험을 설계해야 합니다.

## 결과 채택 기준

`augmentation_balanced`가 baseline보다 CV Macro F1에서 일관된 이득을 보이고, 숙도별 Recall에 큰 악화가 없을 때만 후보로 채택합니다. `augmentation_unweighted`는 가중치의 추가 효과를 판단하는 대조군입니다. Test 결과는 사전에 정한 조건을 최종 확인하는 용도로만 보고, 작은 Test 표본의 한계를 함께 기록합니다.

## 첫 실행 결과 (2026-10-16)

실행 시점의 실제 이미지 스캔 결과는 토마토 Train **279장 / 34그룹**, Test **52장**이었습니다. 클래스 ID는 `unripe=0`, `ripe=1`, `overripe=2`이며 Train 분포는 **59 / 80 / 140**, Test 분포는 **2 / 30 / 20**입니다. metadata CSV의 행은 이미지 한 장씩이 아니라 그룹별 라벨이므로, CSV 행 수를 사진 수로 해석하면 안 됩니다.

| 조건 | CV Macro F1 평균 | CV 표준편차 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| 기준선: balanced, 증강 없음 | 0.7346 | 0.0531 | 0.5192 | 0.4596 |
| balanced + Train 증강 | 0.7424 | 0.0697 | 0.5000 | 0.4412 |
| 가중치 없음 + Train 증강 | 0.7424 | 0.0697 | 0.5000 | 0.4412 |

이번 실행에서는 증강 조건의 CV 평균이 **0.0078** 높았지만 fold 편차가 커졌고, Test Macro F1은 **0.0184 낮아졌습니다**. 이 결과만으로 증강이 개선이라고 판단할 근거는 부족합니다. 증강으로 fold 학습 이미지 수를 균등하게 맞추자 `balanced` 유무의 점수가 같았습니다. 표본 가중치가 추가 이득을 만들지 않았다는 결과이며, 현재 기준선의 가중치를 제거하라는 뜻은 아닙니다. 이후 새 데이터나 더 강건한 검증으로 확인하기 전까지 기존 `class_weight="balanced"`를 유지하고, 증강 조건은 실험 후보로만 보관하는 것을 권합니다.

재현된 결과 파일: `outputs/tomato_augmentation_comparison/20261016_170638/summary.csv`, `fold_scores.csv`, `test_class_reports.json`.
