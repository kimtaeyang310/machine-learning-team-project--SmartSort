> 2026-10-26 소스 정리 이후의 안내: 이 폴더는 과거 비교 실험 기록입니다. 이전 src 모듈을 사용하는 실행 명령은 현재 구성에서 지원하지 않습니다. 정리 전 소스·스크립트는 `archive/src_cleanup_20261026/`에 보관했습니다. 현재 학습·평가는 최상위 README의 `src.joint` 명령을 사용하세요.

# 별도 실험: 9클래스 YOLO와 현재 방식 비교

## YOLO26n-cls 교체 비교

`compare_yolo26.py`는 기존 완료 실행 `20261023T071957744434Z`의 YOLOv8n-cls 통합 모델과
동일한 분할·학습 조건으로 YOLO26n-cls를 학습합니다. 두 모델의 모름 판정을 각각 보정하고
같은 실행에서 Test 품질과 CPU 처리 시간을 다시 비교합니다.

```powershell
& .venv-yolo/Scripts/python.exe experiments/nine_class_comparison/compare_yolo26.py
```

결과: `artifacts/yolo26n_comparison_20261026/report.md` 및 `report.json`.
공식 사전학습 가중치는 최초 실행 시 다운로드가 필요합니다.
완료된 6 epoch 체크포인트는 재사용합니다. 기존 실행 폴더와 데이터가 있어야 하며
사진 내용 해시가 달라지면 평가를 중단합니다.

T08·T09의 적숙 재라벨 및 T08의 Train 이동을 반영해 두 모델을 모두 재학습하려면:

```powershell
& .venv-yolo/Scripts/python.exe experiments/nine_class_comparison/compare_yolo26.py --relabel-t08-t09
```

CSV에서 두 개체의 변경을 확인하고 이전 manifest의 다른 분할은 유지합니다.
출력은 `artifacts/yolo26n_relabel_t08_t09_20261026/`에 별도 저장합니다.
기존 결과는 보존되며 Test 구성이 바뀌었으므로 이전 점수와 직접 비교할 수 없습니다.

과숙 토마토 6장 교환 실험:

```powershell
& .venv-yolo/Scripts/python.exe experiments/nine_class_comparison/compare_yolo26.py --tomato-swap
```

이전 재라벨 실행의 manifest를 기준으로 T17·T18·T19·T20·T21·T23을 Train에 넣고
T27~T32를 Test로 옮깁니다. CSV의 라벨·분할과 일치하는지 확인합니다.
T28은 이전 Calibration이었으므로 Calibration에서도 제외됩니다.
Train 1,234장, Calibration 479장, Selection 466장, Test 528장입니다.
두 모델 모두 사전학습 가중치부터 재학습하며 결과는 `artifacts/yolo26n_tomato_swap_20261026/`에 저장합니다.
이전 Test 오류에 따라 분할을 수정한 개발 실험으로, 독립 Test 성능 향상을 증명하는 실험은 아닙니다.

추가로 T27·T29를 Train으로, T33·T34를 Test로 교환한 실험:

```powershell
& .venv-yolo/Scripts/python.exe experiments/nine_class_comparison/compare_yolo26.py --tomato-swap-two
```

직전 6장 교환 manifest의 나머지 분할을 유지하고 변경된 4장만 반영합니다.
두 모델을 새로 학습하며 `artifacts/yolo26n_tomato_swap_two_20261026/`에 결과를 보존합니다.
오류 분석 후 Test를 수정한 개발 실험이라는 해석상 한계는 동일합니다.

기존 main.py, src, 기본 모델과 원본 데이터는 수정하지 않습니다.
이 폴더 안에만 실험 코드, 데이터 복사본, 체크포인트, 설정, 결과를 저장합니다.
출력 경로를 이 폴더 밖으로 지정하면 중단합니다.

## 실행

프로젝트 루트에서 현재 teamTC 환경으로 실행합니다.

```powershell
& C:/Users/user1/anaconda3/envs/teamTC/python.exe experiments/nine_class_comparison/run_experiment.py
```

기본값은 CPU, 입력 224, batch 16, 고정 6 epochs, seed 42, 시간 측정 2회입니다.
현재 저장된 과일별 YOLO가 5~6 epochs로 최종 학습된 점을 참고해 실험 길이를 미리 정했습니다.
성능에 따라 epoch를 조정하지 않습니다. 수렴·최적 튜닝 결과라고 해석하면 안 됩니다.
작업 중단 후 같은 조건을 이어가려면 `--run-dir`로 해당 artifacts 실행 폴더를 지정합니다.
완료된 학습 체크포인트는 재사용하며 조건이 달라지면 중단합니다.

## 세 가지 비교

1. `stored_current`: 현재 종류/모름 모델과 과일별 숙도 모델의 스냅샷.
2. `matched_two_stage`: 현재 종류/모름 모델과, 새 통합 모델과 같은 Train 사진으로 각각 6 epochs 학습한 숙도 모델 3개.
3. `joint_nine`: 같은 Train 사진으로 6 epochs 학습한 9클래스 YOLO + YOLO 확률/특징 거리 기반 모름 판정.

현재 저장된 숙도 모델은 종류 분류기의 Calibration/Selection 사진까지 포함한 더 큰 기존 Train에서 학습됐습니다.
따라서 현재 저장 모델과의 비교는 실제 교체 효과에 관한 것이며 엄밀한 동일 학습 자료 비교는 아닙니다.
`matched_two_stage`와 `joint_nine`은 학습 사진·사전학습 가중치·epoch·입력 크기·seed를 일치시킵니다.
한 모델과 여러 모델의 배치·최적화 단계는 다릅니다. 종류 모델의 가중치 학습 방식도 다릅니다.

분할은 현재 종류 모델의 저장 manifest를 그대로 사용하며 개체 ID 중복과 사진 내용 해시를 검사합니다.
원래 CSV를 갱신하는 dataset.load_dataset 호출은 하지 않습니다.
YOLO가 요구하는 val 디렉터리는 클래스별 Train 사진 하나씩으로 만들며, 그 점수는 보고하거나 선택에 쓰지 않습니다.
고정 epoch의 last.pt만 사용합니다. Test는 학습·임계값·점수 선택에 사용하지 않습니다.

통합 모델의 MSP/임베딩 거리는 별도 Calibration에서 과일별 95% 종류 정답·수락 목표로 보정합니다.
모름 방식은 별도 Selection에서 포도·레몬 오수락 우선, 과일별 평균 종류 오류/거부 최소 순으로 선택합니다.
한 번의 YOLO forward에 pre-linear hook을 달아 확률과 특징을 함께 얻습니다.

## 결과

각 실행은 `artifacts/실행시각/`에 저장합니다.

- `report.md`, `report.json`: 9클래스 Macro F1, 과일별 숙도 F1, 두 거부 오류율, 전체 시간 비교
- `*_predictions.csv`: 사진별 예측·거부 점수·시간, 반복별 결과
- `rejection_selection.json`: 통합 모델의 기준과 선택용 결과
- `split_manifest.json`: 같은 자료를 썼는지 확인할 분할 기록
- `snapshot/`: 현재 모델·사전학습 가중치 복사본
- `training/`: 실험용 9클래스·과일별 3클래스 모델
- `source_before.json`, `source_integrity.json`: 기존 프로젝트 파일·모델의 내용 해시 확인

알려진 과일을 거부한 사진을 F1에서 제외하지 않습니다. 다른 과일로 분류한 경우도 과일별 숙도 평가에서 실패로 셉니다.
시간에는 읽기·전처리·신경망·거리 계산·숙도 판정을 포함하고 모델 로딩은 별도로 기록합니다.
각 신경망 워밍업 후 같은 사진을 반복 측정하므로 파일 캐시 영향을 받습니다.
현재 구현과 일치하게 OpenCV 이미지 읽기를 사용하며, Test는 이전 분석에 이미 사용됐으므로 회귀 비교 자료입니다.
포도·레몬 이외 미지 과일에 대한 성능은 보장하지 않습니다.
