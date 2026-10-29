# 🍎 SmartSort

## 농산물 종류·숙도 자동 분류 시스템

SmartSort는 농산물 사진에서 **과일 종류와 숙도를 함께 예측**하는 머신러닝 팀 프로젝트입니다.
바나나·귤·토마토를 대상으로 미숙·적숙·과숙을 구분하며,
지원 과일로 받아들이기 어려운 사진은 **모르는 과일**로 표시합니다.

현재 기본 모델은 **YOLO26n-cls 한 개 + 특징 거리 기반 모름 판정**입니다.
초기에는 OpenCV·HSV 특징과 과일별 머신러닝 모델을 사용했고,
이후 과일 종류 자동 인식과 YOLO 비교 실험을 거쳐 통합 9클래스 방식으로 전환했습니다.

이 README는 현재 `main.py`의 동작을 기준으로 작성했습니다.
이전 방식의 상세 설명은 개발 이력과 보관본에서 확인할 수 있습니다.

---

## 목차

1. [프로젝트 정보와 팀](#1-프로젝트-정보와-팀)
2. [문제 정의와 목표](#2-문제-정의와-목표)
3. [현재 시스템의 전체 흐름](#3-현재-시스템의-전체-흐름)
4. [개발 환경과 설치](#4-개발-환경과-설치)
5. [빠른 시작과 메뉴 안내](#5-빠른-시작과-메뉴-안내)
6. [데이터 수집과 촬영 규칙](#6-데이터-수집과-촬영-규칙)
7. [숙도 라벨과 CSV 관리](#7-숙도-라벨과-csv-관리)
8. [데이터 분할과 최신 CSV 반영](#8-데이터-분할과-최신-csv-반영)
9. [통합 9클래스 모델 학습](#9-통합-9클래스-모델-학습)
10. [모르는 과일 판정](#10-모르는-과일-판정)
11. [모델 묶음과 무결성 확인](#11-모델-묶음과-무결성-확인)
12. [평가 지표와 결과 해석](#12-평가-지표와-결과-해석)
13. [평가 이력과 자동 그래프](#13-평가-이력과-자동-그래프)
14. [폴더 예측과 결과 파일](#14-폴더-예측과-결과-파일)
15. [처리 시간과 FPS](#15-처리-시간과-fps)
16. [판정 이후의 행동 제안](#16-판정-이후의-행동-제안)
17. [프로젝트 구조와 주요 파일](#17-프로젝트-구조와-주요-파일)
18. [개발 이력과 이전 원본](#18-개발-이력과-이전-원본)
19. [협업·점검·문제 해결](#19-협업점검문제-해결)
20. [데이터 출처와 참고 문서](#20-데이터-출처와-참고-문서)

---

## 1. 프로젝트 정보와 팀

| 항목 | 내용 |
|---|---|
| 프로젝트명 | SmartSort |
| 저장소명 | `machine-learning-team-project--SmartSort` |
| 주제 | 농산물 종류 및 숙도 이미지 분류 |
| 프로젝트 유형 | 머신러닝 팀 프로젝트 |
| 팀 구성 | 2명 |
| 초기 작업 계획 | 6일 계획으로 시작, 이후 실험과 기능 개선 진행 |
| 주요 언어 | Python |
| 현재 기본 모델 | YOLO26n-cls 통합 9클래스 분류 |
| 기본 실행 환경 | CPU |

### 팀원과 역할

| 이름 | 담당 | 주요 역할 |
|---|---|---|
| 김태양 | Data & Analysis | 이미지 수집·정리, 라벨링, 데이터 품질 점검, EDA, 시각화, 발표 자료 |
| 류채연 | Model & Evaluation | 전처리·특징 추출, 모델 학습·비교, 튜닝, 성능 평가, 오분류 분석 에이전트 확장 |

데이터 최종 검수, 모델 선정, 결과 해석, README와 발표 자료 작성은 공동으로 진행합니다.
역할은 초기 분담 기준이며 실제 개선 작업은 함께 검토합니다.

### 주요 기술

- 이미지 분류와 전이학습
- YOLO26n-cls, PyTorch, Ultralytics
- 특징 벡터와 코사인 거리 기반 모름 판정
- 개체 단위 데이터 분할
- Macro F1, 혼동행렬, 오거부·오수락 분석
- OpenCV·HSV·K-means·Scikit-learn 기반 초기 비교 실험

---

## 2. 문제 정의와 목표

### 입력과 출력

입력은 농산물이 담긴 이미지 파일입니다.
현재 기본 경로는 파일명으로 과일을 선택하지 않고 사진 내용으로 종류와 숙도를 함께 예측합니다.

| 출력 | 의미 |
|---|---|
| `banana` | 바나나 |
| `mandarin` | 귤 |
| `tomato` | 토마토 |
| `unripe` | 미숙: 덜 익음 |
| `ripe` | 적숙: 적당히 익음 |
| `overripe` | 과숙: 너무 익음 |
| `unknown` | 지원 과일로 받아들이지 못함, 숙도 표시 생략 |
| `failed` | 이미지 읽기 또는 처리 중 오류 |

9클래스는 과일 3종과 숙도 3단계의 조합입니다.

```text
banana_unripe     banana_ripe     banana_overripe
mandarin_unripe   mandarin_ripe   mandarin_overripe
tomato_unripe     tomato_ripe     tomato_overripe
```

### 목표 성능

| 항목 | 프로젝트 목표 |
|---|---|
| 숙도 분류 | Macro F1 0.85 이상 |
| 처리 속도 | 25 FPS 이상 |
| 지원하지 않는 입력 | 모름 판정 및 사람 확인 안내 |
| 데이터 관리 | 개체 중복을 통제한 학습·평가와 라벨 이력 확인 |

목표와 실제 측정 결과는 구분합니다.
현재 시스템은 숙도를 분류하며, 식품 안전성·당도·내부 품질을 직접 측정하지 않습니다.
또한 여러 과일의 위치를 찾아 개별 판정하는 객체 탐지 모델이 아니라 이미지 전체를 분류합니다.

---

## 3. 현재 시스템의 전체 흐름

```text
이미지 파일
    ↓
이미지 읽기·분류용 전처리
    ↓
YOLO26n-cls 한 번 실행
    ├── 9클래스 예측 확률
    └── 분류 직전 특징 벡터
                ↓
학습 기준 특징과 거리 비교
                ↓
예측 과일별 기준값 적용
    ├── 수락: 과일 종류 + 숙도
    └── 거부: 모르는 과일, 숙도 생략
                ↓
행동 제안 + CSV·실행 정보 저장
```

현재 기본 경로는 과일 종류 전용 MobileNet이나 별도의 숙도 신경망을 추가 로딩하지 않습니다.
특징 벡터는 YOLO의 마지막 분류층 직전에서 얻습니다.
모름 판정도 같은 YOLO 실행에서 얻은 정보를 사용합니다.

초기 HSV 특징·K-means 마스크 파이프라인은 보관본과 과거 비교 실험 기록에 남아 있습니다.
현재 통합 YOLO 경로에는 포함되지 않습니다.

---

## 4. 개발 환경과 설치

### 주요 환경

- Windows 및 PowerShell 중심 실행
- Python: 현재 팀 환경에서 3.10.22 사용 기록
- Conda 가상환경: 팀에서 `teamTC` 사용
- 편집기: VS Code
- 협업: Git·GitHub

현재 `requirements.txt`는 모든 패키지 버전을 고정한 환경 잠금 파일은 아닙니다.
팀원 간 결과와 속도를 비교할 때는 실제 설치 버전도 확인합니다.

### 설치

프로젝트 최상위 폴더에서 실행합니다.
기존 `teamTC` 환경이 있다면 활성화한 뒤 의존성을 확인합니다.

```powershell
conda activate teamTC
python -m pip install -r requirements.txt
python main.py
```

새 환경을 준비하는 경우의 예시입니다.

```powershell
conda create -n teamTC python=3.10
conda activate teamTC
python -m pip install -r requirements.txt
```

현재 의존성에는 `ultralytics==8.4.177`, `torch`, `torchvision`, `Pillow`,
`numpy`, `pandas`, `joblib`, `opencv-python`, `scikit-learn` 등이 포함됩니다.
기존 비교 실험을 위한 LightGBM·XGBoost도 목록에 있습니다.

### 실행 전 필요한 파일

- 추론: `models/joint_nine/`의 모델 묶음 4개 파일
- 재학습: 기존 manifest에 기록된 이미지와 `data/metadata/fruit_labels.csv`
- 재학습 초기 가중치: `models/pretrained/yolo26n-cls.pt`
- 평가: 현재 manifest의 테스트 이미지

사용자별 Python 절대 경로는 다를 수 있습니다.
환경 활성화가 어려우면 설치된 환경의 실행 파일을 직접 지정할 수 있습니다.

```powershell
& C:/Users/user1/anaconda3/envs/teamTC/python.exe main.py
```

---

## 5. 빠른 시작과 메뉴 안내

```powershell
python main.py
```

| 메뉴 | 기능 | 주요 결과 |
|---|---|---|
| 1 | 통합 9클래스 학습 및 모름 기준 설정 | 새 모델·정책·분할 목록 |
| 2 | 저장 모델 평가 | F1·오거부·오수락·FPS 및 이력 그래프 |
| 3 | 폴더 전체 예측 | 사진별 과일·숙도·시간, CSV·실행 정보 |
| 4 | 현재 모델과 모름 기준 확인 | 모델 출처, 파일 해시, 기준값 |
| 5 | 이전 원본 안내 | 보관 위치와 실행·복원 안내 |
| 0 | 종료 | 프로그램 종료 |

메뉴는 작업 하나를 실행한 뒤 종료합니다.
다음 작업을 수행하려면 `python main.py`를 다시 실행합니다.

### 저장된 모델로 먼저 예측하기

1. `main.py` 실행 후 3번을 선택합니다.
2. 폴더 경로를 입력하거나 Enter로 기본 폴더를 사용합니다.
3. 화면 결과와 `outputs/predictions/`의 파일을 확인합니다.

이미 모델 묶음이 있다면 예측을 위해 먼저 학습할 필요는 없습니다.

### 명령으로 직접 실행하기

```powershell
# 평가
python -m src.joint evaluate

# 학습: 기본 6 epochs
python -m src.joint train --epochs 6

# 기본 폴더 예측
python -m src.predict

# 다른 폴더 예측
python -m src.predict "C:/photos/fruit"
```

현재 기본 학습·평가는 `main.py` 또는 `src.joint`를 사용합니다.
이전 `src.train`·`src.evaluate`와 HSV 기반 모듈은 현재 src에서 제거했습니다.
정리 전 소스는 `archive/src_cleanup_20261026/`에 보관했습니다.


## Ollama 대화형 SmartSort 비서

SmartSort 비서는 Ollama의 로컬 언어 모델과 기존 과일·숙도 예측 코드를 연결한 데스크톱 채팅 앱입니다. 사용자는 채팅으로 작업을 요청하고, 비서는 허용된 SmartSort 작업을 선택해 실행하도록 구성되어 있습니다.

### 구성

- `assistant_app.py`: 비서 앱 실행 파일
- `src/assistant/ui.py`: 채팅 창과 폴더 선택 창
- `src/assistant/ollama_client.py`: Ollama 대화와 도구 호출 연결
- `src/assistant/tools.py`: 허용 작업, 입력 검사, 기존 예측 코드 호출

### 설치 및 실행

Ollama 앱을 설치하고, 사용할 모델을 한 번 내려받습니다.

```powershell
ollama pull qwen3:4b
```

프로젝트 Python 환경에 패키지를 설치합니다.

```powershell
& C:/Users/user1/anaconda3/envs/teamTC/python.exe -m pip install -r requirements.txt
```

Ollama가 실행 중인 상태에서 프로젝트 최상위 폴더에서 비서 앱을 시작합니다.

```powershell
& C:/Users/user1/anaconda3/envs/teamTC/python.exe assistant_app.py
```

### 지원 작업

- 기본 예측 폴더 또는 사용자가 선택한 폴더의 사진 분석
- 최근 예측 결과 요약 및 모르는 과일 목록 확인
- 현재 모델과 판정 기준 안내
- 과일별 숙도 개수, 모르는 과일 수, 처리 실패 수, 처리 시간과 FPS 확인
- 예측 결과 CSV 저장

이미지는 한 번에 최대 60장까지 처리하며, JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF 형식을 지원합니다. 예측 CSV는 `outputs/predictions`에 새 파일로 저장됩니다.

### 실행 제한

비서는 허용된 SmartSort 작업만 실행합니다. 임의 코드나 터미널 명령 실행, 파일 수정·삭제, 재학습, 판정 기준 변경은 지원하지 않습니다. 다른 폴더의 사진은 채팅에 경로를 입력하는 대신 폴더 선택 창에서 지정해야 합니다.

정답 라벨이 없는 예측 실행에서는 정확도나 F1 점수를 계산하지 않습니다. 처리 시간과 FPS는 실행 속도 지표이며, 분류의 정확도를 뜻하지 않습니다.

### 현재 확인 사항

앱 창과 Python 모듈 연결은 확인했습니다. 다만 모델이 요청을 예측 도구 호출로 연결하지 않고 일반 답변을 할 수 있어, 실제 도구 호출과 결과 응답은 계속 확인해야 합니다.


---

## 6. 데이터 수집과 촬영 규칙

직접 촬영한 귤·토마토와 외부 데이터셋을 함께 사용합니다.
자료의 출처와 촬영 조건은 가능한 범위에서 기록합니다.

### 직접 촬영 기준

- 흰색 배경을 기본으로 사용합니다.
- 가능한 일정한 거리와 각도로 촬영합니다.
- 촬영 날짜, 과일 종류, 개체 ID를 남깁니다.
- 조명이 다른 사진은 촬영 조건을 구분합니다.
- 같은 개체를 날짜별로 촬영할 경우 개체 ID를 유지합니다.
- 숙도 판정은 날짜만으로 정하지 않고 사진과 실제 상태를 검토합니다.

### 파일명 예시

```text
20261008_mandarin_M09_indoor_01.jpg
20261008_mandarin_M09_indoor_dim_06.jpg
20261012_tomato_T15_indoor_01.jpeg
```

| 구성 | 의미 |
|---|---|
| `20261008` | 촬영 날짜 |
| `mandarin` | 과일 종류 |
| `M09` | 개체 식별자 |
| `indoor` | 실내 촬영 조건 |
| `indoor_dim` | 어두운 실내 조건 |
| 마지막 번호 | 같은 조건의 사진 구분 |

파일명 규칙은 자료 관리에 사용합니다.
현재 기본 예측 모델은 파일명의 정답 정보를 이용해 숙도를 결정하지 않습니다.

### 데이터 다양성

같은 과일의 연속 사진 50장은 서로 다른 개체 50장과 다릅니다.
조명·품종·배경·촬영 거리·숙도 경계가 다양한 자료를 확보하는 것이 중요합니다.
특히 단일 과일 사진과 여러 개·절단면을 포함한 사진은 구성이 다르므로 별도로 살펴봅니다.

---

## 7. 숙도 라벨과 CSV 관리

직접 촬영한 귤·토마토의 숙도와 학습·테스트 지정은 다음 파일에서 관리합니다.

```text
data/metadata/fruit_labels.csv
```

```csv
fruit,fruit_id,date,ripeness,group_id,split
tomato,T01,2026-10-03,unripe,T01_20261003,train
tomato,T01,2026-10-04,ripe,T01_20261004,train
tomato,T02,2026-10-04,ripe,T02_20261004,test
mandarin,M01,2026-10-06,overripe,M01_20261006,train
```

위 표는 형식 예시이며 현재 데이터의 실제 라벨을 의미하지 않습니다.

| 열 | 설명 |
|---|---|
| `fruit` | `mandarin` 또는 `tomato` 등 과일 이름 |
| `fruit_id` | 개체 ID |
| `date` | 촬영 날짜 |
| `ripeness` | `unripe`, `ripe`, `overripe` |
| `group_id` | 개체·날짜 단위 식별자 |
| `split` | `train` 또는 `test` |

통합 학습의 최신 라벨 반영은 `(fruit, group_id)`와 manifest의 `(fruit, source_group)`을 연결합니다.
같은 키의 CSV 행이 서로 다른 숙도 또는 split을 지정하면 오류로 중단합니다.
CSV의 split에는 `calibration`·`selection`을 직접 입력하지 않습니다.
바나나는 현재 이 CSV 갱신 함수의 대상이 아니며 기존 manifest 정보를 사용합니다.

### 라벨 수정 절차

1. 해당 개체·촬영 날짜의 사진을 확인합니다.
2. 팀원과 숙도 기준을 검토합니다.
3. CSV의 `ripeness` 또는 `split`을 수정하고 저장합니다.
4. 1번 메뉴를 실행합니다.
5. 학습이 완료되면 2번 메뉴로 평가합니다.

CSV만 수정한 뒤 2번 평가를 실행하면 변경이 자동 반영되지는 않습니다.
2번은 현재 설치된 모델 묶음의 manifest를 기준으로 평가합니다.

---

## 8. 데이터 분할과 최신 CSV 반영

통합 모델의 실제 데이터 목록은 다음 파일에 있습니다.

```text
models/joint_nine/split_manifest.json
```

| 내부 phase | 역할 |
|---|---|
| `train` | YOLO 학습 및 기준 특징 구성 |
| `calibration` | 과일별 모름 기준값 계산 |
| `selection` | 모름 판정 방식 비교·선택 |
| `test` | 저장된 모델 평가 |

CSV의 2개 split과 내부 4개 phase는 같은 개념이 아닙니다.
CSV의 `train`에는 기존 내부 보정·선택 자료도 포함될 수 있습니다.

### 1번 메뉴의 반영 규칙

| CSV 지정 | 기존 내부 phase | 반영 후 |
|---|---|---|
| `test` | 어느 phase든 | `test` |
| `train` | `test` | `train` |
| `train` | `train` | `train` 유지 |
| `train` | `calibration` | `calibration` 유지 |
| `train` | `selection` | `selection` 유지 |

숙도는 최신 CSV의 값으로 갱신합니다.
기존 내부 보정·선택 자료를 보존하므로 CSV에서 train으로 지정했다고 모두 신경망 학습에 들어가지는 않습니다.
Test에서 train으로 변경한 자료는 실제 학습으로 이동합니다.

### 개체 중복 검사

`group_id`는 날짜를 포함하지만, 내부 `group`은 `tomato:T01`처럼 개체 단위로 관리합니다.
같은 개체의 사진이 여러 phase에 걸치면 오류로 중단합니다.
따라서 같은 개체의 여러 날짜를 train과 test에 섞어 지정하면 충돌할 수 있습니다.

### 반영 범위와 기록

- 갱신 대상은 기존 manifest에 등록된 귤·토마토입니다.
- 새 CSV 행이나 새 이미지 파일을 자동으로 목록에 추가하지 않습니다.
- 새 자료 편입은 이미지 해시·개체·분할을 포함한 manifest 갱신이 필요합니다.
- 이미지 내용이 기존 SHA-256과 다르면 학습·평가가 중단됩니다.
- 변경 내역은 학습 실행 폴더의 `label_changes.json`에 저장됩니다.
- 학습과 기준 설정이 성공한 뒤 새 manifest가 모델 묶음에 설치됩니다.

---

## 9. 통합 9클래스 모델 학습

```powershell
python -m src.joint train --epochs 6
```

| 설정 | 현재 기본값 |
|---|---|
| 초기 가중치 | `models/pretrained/yolo26n-cls.pt` |
| 학습 횟수 | 6 epochs |
| 입력 크기 | 224 |
| batch | 16 |
| 장치 | CPU |
| seed | 42 |
| PyTorch CPU 스레드 | 4 |
| 체크포인트 | 고정 횟수 학습 후 `last.pt` |

현재 통합 학습은 독립 검증 점수로 `best.pt`를 선택하는 방식이 아닙니다.
API 로더를 위한 val 폴더에 학습 사진 일부를 복사하지만 그 점수를 모델 선택에 쓰지 않습니다.
자동 증강, 영역 지우기, HSV 색상 증강은 코드에서 비활성화되어 있습니다.
분류 모델 자체의 기본 이미지 변환은 적용됩니다.

### 실행 흐름

1. 기존 manifest를 읽고 최신 CSV 숙도·split을 반영합니다.
2. 이미지 해시와 개체 분할을 검사합니다.
3. `train` 자료로 9클래스 학습 폴더를 구성합니다.
4. 사전학습 가중치에서 지정 횟수만큼 학습합니다.
5. 기준 특징과 과일별 모름 기준을 계산합니다.
6. Selection 자료로 판정 방식을 선택합니다.
7. 새 모델 묶음을 로딩 검증한 뒤 현재 모델로 설치합니다.

현재 `train` 명령은 기존 모델에서 이어 학습하는 명령이 아닙니다.
새 실행 폴더를 만들고 사전학습 가중치로 학습을 시작합니다.

### 학습 산출물

```text
outputs/joint_nine/training_<UTC시각>/
├── config.json
├── label_changes.json
├── split_manifest.json
├── dataset/
├── training/
│   ├── results.csv
│   └── weights/last.pt
├── nine_development_predictions.npz
├── nine_rejection_policy.joblib
└── rejection_selection.json
```

학습 횟수를 늘리려면 `--epochs 15`처럼 지정할 수 있습니다.
현재 명령행에서 GPU·입력 크기·batch를 바꾸는 옵션은 제공하지 않습니다.
이 설정 변경은 코드 수정과 별도 검증이 필요합니다.

---

## 10. 모르는 과일 판정

9개 중 가장 높은 확률의 클래스를 후보로 선택한 뒤, 모름 판정을 적용합니다.
현재 구현은 다음 두 방식을 비교하고 하나를 정책에 저장합니다.

| 방식 | 계산 | 큰 값의 의미 |
|---|---|---|
| `msp` | 1 − 가장 높은 9클래스 확률 | 분류 확신이 낮음 |
| `embedding` | 가까운 기준 특징 5개의 평균 코사인 거리 | 학습 기준 특징과 다름 |

특징은 길이를 정규화한 뒤 비교합니다.
임베딩 거리는 같은 과일만 따로 검색하는 방식이 아니라 전체 학습 기준 특징에서 계산합니다.
비교 후에는 후보 과일의 기준값을 적용합니다.

```text
선택 방식의 점수 ≤ 후보 과일의 기준값 → 수락
선택 방식의 점수 > 후보 과일의 기준값 → unknown
```

기준값은 숙도 분류 확률이나 정확도 백분율이 아닙니다.
과일별 숫자 크기만으로 어느 과일 모델이 더 좋은지 판단하지 않습니다.

Calibration의 알려진 과일에 대해 약 95% 종류 정답·수락 목표를 반영해 기준을 구합니다.
Selection에서는 미지 과일 오수락을 우선 줄이는 방식을 선택합니다.
이는 해당 개발 자료에서의 기준 설정이며 모든 새 사진의 통과율을 보장하는 값은 아닙니다.

### 4번 메뉴 읽는 법

- `method`: 실제 사용하는 모름 판정 방식
- `thresholds`: 방식별·과일별 기준값
- `reference_count`: 기준 특징 벡터 수
- `imgsz`: 입력 이미지 크기 설정
- `source`: 현재 모델의 출처 실행 폴더
- `sha256`: 모델 묶음 파일의 내용 확인용 해시

기준은 재학습 때 바뀔 수 있으므로 현재 값은 4번 메뉴에서 확인합니다.
`unknown`은 실제로 미지 과일임을 확정하는 뜻이 아니라 모델이 예측 수락을 보류했다는 뜻입니다.

---

## 11. 모델 묶음과 무결성 확인

```text
models/joint_nine/
├── model.pt
├── policy.joblib
├── split_manifest.json
└── bundle.json
```

| 파일 | 역할 |
|---|---|
| `model.pt` | 9클래스 YOLO 가중치 |
| `policy.joblib` | 선택 방식·기준값·학습 기준 특징 |
| `split_manifest.json` | 사진 경로·숙도·개체·phase·이미지 해시 |
| `bundle.json` | 형식 버전·입력 크기·모델 이름·출처·파일 해시 |

모델 로딩 시 `bundle.json`에 기록된 파일 해시를 확인합니다.
모델·정책·manifest를 서로 다른 실행에서 임의로 섞으면 불일치할 수 있습니다.
팀원에게 모델을 전달할 때는 폴더의 4개 파일을 함께 전달합니다.

`source`의 학습 폴더는 추적용 정보입니다.
추론에는 묶음 파일이 필요하며 학습 출처 폴더 전체를 다시 로딩하지 않습니다.
재학습에는 초기 가중치와 데이터가 추가로 필요합니다.

새 모델 설치 전에는 로딩 가능 여부를 검사합니다.
기존 운영 모델은 `outputs/joint_nine/previous_<UTC시각>/`에 보관합니다.
학습과 정책 생성이 끝나지 않은 중간 파일을 운영 모델로 직접 복사하지 않습니다.

---

## 12. 평가 지표와 결과 해석

```powershell
python -m src.joint evaluate
```

2번 메뉴는 현재 모델 묶음의 `test` 자료를 평가합니다.
CSV를 다시 읽거나 재학습하지 않습니다.
화면에는 주요 점수와 FPS를, 결과 폴더에는 상세 기록을 저장합니다.

| 지표 | 의미 |
|---|---|
| 9클래스 Macro F1 | 과일·숙도 조합 9개의 F1 평균 |
| 과일별 숙도 Macro F1 | 해당 과일의 미숙·적숙·과숙 F1 평균 |
| Accuracy | 정답 과일·숙도를 맞힌 비율, 거부도 실패로 계산 |
| 오거부율 | 알려진 과일을 모름으로 처리한 비율 |
| 오수락률 | 미지 과일을 지원 과일로 받아들인 비율 |
| 혼동행렬 | 정답 클래스별 예측·모름·실패 분포 |

F1은 정밀도와 재현율의 조화평균입니다.
Macro F1은 사진 수가 적은 클래스도 같은 비중으로 반영합니다.
미숙 사진 2장을 모두 놓치면 다른 숙도 대부분을 맞혀도 평균 점수가 크게 내려갈 수 있습니다.
현재 구현은 지정된 3개 숙도·9개 조합을 고정하여 계산하고, 정의되지 않는 F1은 0으로 처리합니다.
클래스가 테스트에 없을 때도 클래스별 사진 수를 함께 확인해야 합니다.

### 최신 저장 평가 예시

README 확인 시점인 **2026-10-26**의 최신 저장 실행 기준입니다.
실행 시각은 **15:39:36 KST**, 폴더 시각은 UTC입니다.

| 항목 | 측정값 |
|---|---:|
| 9클래스 Macro F1 | 0.8141 |
| 바나나 숙도 Macro F1 | 0.8895 |
| 귤 숙도 Macro F1 | 0.8861 |
| 토마토 숙도 Macro F1 | 0.6667 |
| 알려진 과일 | 489장 |
| 미지 과일 | 40장 |
| 오거부 | 15/489장, 3.07% |
| 오수락 | 0/40장, 0.00% |
| 처리 FPS | 41.22 |

근거: `outputs/joint_nine/evaluation_20261026T063936133579Z/metrics.json` 및 `report.md`.
수치는 README 작성 당시 기록이며 이후 평가로 바뀔 수 있습니다.
미지 과일 40장의 오수락 0건은 모든 미지 입력에 대한 보장이 아닙니다.
기존 Test를 반복 확인해 온 평가이므로 실행 화면에서도 회귀 평가로 안내합니다.

---

## 13. 평가 이력과 자동 그래프

2번 메뉴 평가가 완료되면 누적 CSV와 SVG를 자동 갱신합니다.
이전 모델의 평가와 통합 9클래스 평가를 같은 계열에 이어서 표시합니다.
전체·바나나·귤·토마토의 Macro F1 네 계열을 사용합니다.

![전체 및 과일별 Macro F1 추이](outputs/evaluation/evaluation_history.svg)

```text
outputs/evaluation/evaluation_history.csv
outputs/evaluation/evaluation_history.svg
```

- 전체 계열에는 현재 9클래스 Macro F1을 기록합니다.
- 과일 계열에는 각 과일의 숙도 Macro F1을 기록합니다.
- 완료된 기존 통합 평가 기록도 재집계합니다.
- 같은 실행을 다시 집계해도 이력을 중복 추가하지 않습니다.
- 그래프의 실행 시각은 한국 시간으로 표시합니다.
- 점에 마우스를 올리면 정확도·실행 시각·데이터 구분값을 확인할 수 있습니다.
- 그래프 갱신에 실패해도 저장된 평가 결과는 유지하며 메시지를 출력합니다.

학습 메뉴만 실행하거나 폴더 예측을 실행하는 경우 평가 이력을 추가하지 않습니다.
README의 성능 표는 문서 작성 시점의 예시이며, 그래프처럼 자동 수정되지는 않습니다.

그래프만 다시 만들려면 다음 명령을 사용합니다.
이 명령은 모델 평가를 새로 실행하지 않습니다.

```powershell
python -m src.performance_history
```

---

## 14. 폴더 예측과 결과 파일

3번 메뉴의 기본 입력 폴더는 `data/predict_list`입니다.
다른 경로를 입력할 수도 있으며 해당 폴더 바로 아래의 이미지 파일을 처리합니다.
하위 폴더를 재귀적으로 탐색하지는 않습니다.

지원 확장자: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.
확장자는 대소문자를 구분하지 않고 확인합니다.

### 출력 예시

```text
[1/3] sample_a.jpg → banana / ripe (10.20 ms)
[2/3] sample_b.jpg → tomato / unripe (11.30 ms)
[3/3] sample_c.jpg → 모르는 과일 (9.80 ms)

숙도 판정 2 / 모르는 과일 1 / 실패 0
```

위 시간과 판정은 출력 형식 예시입니다.
실제 결과에는 실행 환경·이미지·모델의 영향이 반영됩니다.

### 예측 저장 파일

```text
outputs/predictions/<UTC시각>.csv
outputs/predictions/<UTC시각>.manifest.json
```

| 주요 CSV 필드 | 설명 |
|---|---|
| `image_path` | 입력 파일 |
| `status` | `ok`, `unknown`, `failed` |
| `fruit`, `ripeness` | 수락된 과일·숙도 |
| `candidate_class` | 모름 판정 전 9클래스 후보 |
| `nine_probabilities_json` | 9클래스 확률 |
| `rejection_score` | 모름 판정 점수 |
| `rejection_threshold` | 적용 기준값 |
| `agent_decision_json` | 행동 제안 |
| `total_ms` | 사진별 처리 시간 |
| `error` | 실패 시 오류 내용 |

실행 정보에는 모델 해시, 운영체제·Python 정보, 입력 폴더, 처리 건수와 시간 요약 등이 들어갑니다.
모름 처리된 사진은 후보 클래스가 남아 있어도 최종 숙도는 표시하지 않습니다.

### 평가 저장 파일

```text
outputs/joint_nine/evaluation_<UTC시각>/
├── metrics.json
├── predictions.json
├── predictions.csv
├── predictions.manifest.json
└── report.md
```

평가의 `predictions.json`에는 정답 과일·숙도·클래스도 저장됩니다.
오답 분석은 이 파일에서 정답과 후보·최종 상태를 함께 비교합니다.
결과 폴더의 `report.md`는 화면에 나온 요약을 다시 확인할 때 사용합니다.

---

## 15. 처리 시간과 FPS

```text
FPS = 정상 처리·모름 판정 장수 × 1000 / 전체 처리 시간(ms)
```

실패가 없다면 `1000 / 평균 처리 시간(ms)`와 같습니다.
실패 시간도 분모에 포함되며 실패 건수는 분자에 포함하지 않습니다.

### 측정 범위

| 포함 | 제외 |
|---|---|
| 이미지 읽기·디코딩 | 모델 로딩 |
| 분류용 전처리 | 사전 워밍업 |
| YOLO 실행 | 에이전트 행동 제안 |
| 특징 거리·모름 판정 | 화면 출력·CSV 저장 |
| 예측 결과 구성 | 프로그램 시작 시간 |

폴더 예측은 모델을 한 번 로딩하고, 먼저 처리 가능한 이미지 한 장으로 워밍업합니다.
이후 사진을 순차 처리합니다.
평가도 측정 전 워밍업을 수행합니다.
현재 통합 코드에서는 CPU와 PyTorch 스레드 4개를 지정합니다.

### 평균과 P95

- 평균: 전체 사진의 평균 처리 시간
- P95: 처리 시간 분포에서 약 95% 지점의 시간
- FPS: 지정된 측정 범위의 처리량

일부 고해상도 사진이 오래 걸리면 평균과 P95가 커집니다.
모델 입력 크기가 224여도 원본 읽기·디코딩 비용은 원본 크기의 영향을 받습니다.

### 팀원 간 비교 방법

동일 이미지 파일·모델 묶음·코드·환경으로 비교합니다.
가능하면 여러 번 실행해 일시적인 CPU 부하와 파일 캐시 영향을 함께 살펴봅니다.
FPS 차이가 크면 결과 CSV의 `total_ms`를 내림차순으로 확인합니다.
CPU·전원 설정·저장장치·라이브러리 버전·다른 프로그램 부하도 영향을 줍니다.

---

## 16. 판정 이후의 행동 제안

예측 결과에는 `src/agent/decision.py`의 규칙으로 다음 행동을 제안합니다.
현재 정책은 추가 신경망이 아니라 예측 결과에 적용하는 규칙입니다.

| 행동 | 의미 |
|---|---|
| `auto_classify` | 자동 분류 제안 |
| `human_review` | 사람 확인 제안 |
| `retake` | 재촬영 제안 |

현재 기본 확률 기준은 0.70입니다.
코드에 지정된 `tomato / overripe` 조합은 사람 확인을 우선합니다.
이 조합은 정책 설정이며 최신 F1 결과로 자동 해제되지는 않습니다.

수락된 예측에서는 9클래스 중 후보 클래스의 확률을 사용합니다.
과일 내부 3개 확률의 합으로 다시 정규화하지 않습니다.
모름 판정은 사람 확인으로 연결하며 숙도와 확신도 표시를 생략합니다.

재촬영 제안은 확률 기준에 따른 권고입니다.
사진의 초점·노출을 별도 검사한 결과는 아닙니다.
이 기능은 다음 행동을 제안하며 실제 선별 장치를 제어하지 않습니다.

---

## 17. 프로젝트 구조와 주요 파일

```text
SmartSort/
├── main.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── metadata/fruit_labels.csv
│   └── predict_list/
├── models/
│   ├── pretrained/yolo26n-cls.pt
│   └── joint_nine/
├── src/
│   ├── joint.py
│   ├── joint_core.py
│   ├── joint_labels.py
│   ├── predict.py
│   ├── performance_history.py
│   ├── rejection.py
│   └── agent/
├── outputs/
│   ├── joint_nine/
│   ├── predictions/
│   └── evaluation/
├── experiments/nine_class_comparison/
├── docs/
├── tests/
└── archive/
    ├── before_joint_nine_20261023/
    └── src_cleanup_20261026/
```

| 파일 | 역할 |
|---|---|
| `main.py` | 메뉴 진입점 |
| `src/joint.py` | 현재 학습·평가·예측·모델 설치 |
| `src/joint_core.py` | YOLO 실행·보정·지표 계산 |
| `src/joint_labels.py` | CSV 숙도·split 반영 |
| `src/predict.py` | 예측 진입점·결과 CSV 및 실행 정보 저장 |
| `src/performance_history.py` | 누적 이력·SVG 생성 |
| `src/rejection.py` | 모름 점수·특징 거리·과일별 기준 보정 |
| `src/agent/` | 행동 제안 규칙 |

현재 필요한 공유 함수는 `src/rejection.py`로 분리했습니다.
이전 방식 전용 Python 파일 15개와 해당 테스트는 정리 전 보관본에 남겼습니다.
현재 예측 명령은 통합 YOLO 전용이며 이전 `--fruit-mode image/filename` 옵션은 제거했습니다.

---

## 18. 개발 이력과 이전 원본

### 초기 설계

초기 프로젝트는 스마트폰 사진에서 색상과 표면 특징을 추출해 숙도를 분류하는 방식으로 시작했습니다.
촬영·라벨링, 전처리, 특징 추출, 모델 비교, 평가, 발표 자료를 6일 계획으로 구성했습니다.

### OpenCV·특징 기반 모델

```text
사진 → 전처리 → K-means 배경 분리 → HSV·색상 비율·밝기 특징
     → Logistic Regression / SVM / Random Forest / KNN / Decision Tree
     → 숙도 분류
```

HSV 통계·색상 비율·표면 특징을 사용하고 과일별 모델 비교를 진행했습니다.
PCA·특징 제거 실험·오분류 분석은 특징과 클래스 분포를 검토하는 방법으로 다뤘습니다.
특징 캐시는 기존 학습·평가 경로에서 반복 계산을 줄이는 데 사용했습니다.

### 종류 인식과 YOLO 비교

과일 종류 전용 분류기와 모름 판정을 추가하고,
과일별 숙도 모델 및 YOLOv8n-cls 비교 실험을 진행했습니다.
이후 한 모델로 과일·숙도를 함께 분류하는 통합 방식을 도입했습니다.

- 2026-10-23: 기본 실행을 통합 9클래스 YOLOv8n-cls로 전환
- 2026-10-26: YOLO26n-cls 채택 및 후속 재학습
- 후속 개선: CSV 숙도·split 갱신, 평가 이력 그래프 자동 갱신

이전 모듈을 불러오는 과거 실험·변환 스크립트는 현재 src 구성에서 실행할 수 없습니다.
해당 소스는 `archive/src_cleanup_20261026/`의 정리 전 스냅샷에 함께 보관했습니다.
과거 방식 재현에는 보관본 소스와 당시 모델·데이터 경로가 필요합니다.

과거 모델별 점수는 당시 데이터·분할·평가 방식의 기록입니다.
원본 실험의 설정과 산출물은 해당 문서·실행 폴더에서 확인합니다.

### 이전 원본 보관

```text
archive/before_joint_nine_20261023/
```

전환 전 코드·데이터·모델·학습 체크포인트를 보관한 위치입니다.
`archive_manifest.json`에는 3,085개 원본 파일의 SHA-256 기록이 있습니다.
보관본의 상세 실행·복원 범위는 폴더 안의 `RESTORE.md`를 따릅니다.
보관본 README는 이전 구현 시점의 설명이므로 현재 메뉴 안내와 다릅니다.

이전 방식을 실행하려면 보관 폴더로 이동한 뒤 해당 환경에서 `main.py`를 실행합니다.
복원할 때는 현재 작업을 별도 보관하고 필요한 원본 경로를 확인합니다.
과거 outputs·experiments는 현재 위치에도 계속 보존되어 있을 수 있습니다.

---

## 19. 협업·점검·문제 해결

### 협업 방법

- 데이터·코드·모델 변경을 구분해 기록합니다.
- 팀원별 작업 브랜치에서 수정하고 PR로 검토합니다.
- 라벨 수정 시 대상 개체·날짜·숙도·split을 함께 공유합니다.
- 모델 전달 시 모델 묶음 4개 파일을 함께 전달합니다.
- 평가·예측 결과 파일과 실행 정보를 함께 남깁니다.

프로젝트의 `.gitignore`에는 실행 산출물과 보관본을 무시하는 규칙이 있습니다.
화면에 파일이 존재해도 팀원에게 Git으로 전달되는지는 별도로 확인합니다.
가상환경·대용량 실행 결과를 무조건 `git add .`로 추가하지 않습니다.

### 관련 기능 점검

```powershell
python -m unittest tests.test_joint tests.test_joint_labels tests.test_performance_history
```

이 명령은 관련 단위 테스트이며 실제 전체 데이터 재학습을 대신하지 않습니다.
테스트용 임시 폴더 권한 오류가 발생하면 쓰기 가능한 임시 경로 설정을 확인합니다.

### 자주 발생하는 문제

| 증상 | 확인할 내용 |
|---|---|
| CSV 수정 후 평가가 같음 | 1번 학습이 성공했는지, 새 manifest가 설치됐는지 확인 |
| train 지정인데 학습에 안 들어감 | 기존 calibration·selection은 유지되는 규칙 확인 |
| 새 사진이 학습에 안 들어감 | 기존 manifest 대상만 갱신하므로 자료 편입 절차 필요 |
| 같은 개체 분할 충돌 | 같은 개체의 모든 날짜에 대한 CSV split 확인 |
| 모델 파일 변경 오류 | `bundle.json`과 모델 묶음의 해시 일치 확인 |
| 사진 변경 오류 | manifest 등록 후 이미지 교체·리사이즈 여부 확인 |
| 숙도 F1과 Accuracy 차이가 큼 | 숙도별 사진 수·오답·고정 클래스 평균 방식 확인 |
| 일부 사진만 매우 느림 | CSV 시간순 정렬 후 원본 크기·형식·읽기 상태 확인 |
| 그래프가 갱신되지 않음 | 2번 완료 여부·갱신 메시지·SVG 파일 시각 확인 |
| 다른 팀원 결과와 다름 | 입력 파일·모델 해시·코드·환경 버전 비교 |

### 앞으로의 개선 항목

- 미숙 토마토의 개체 다양성과 라벨 일관성 점검
- 각 숙도 평가 자료의 충분한 확보
- 검증 성능을 이용한 학습 횟수·모델 선택 비교
- 동일 입력에서의 처리 속도 비교
- 실제 사용 조건에 맞는 입력 사진 범위 정리

이 항목은 향후 개선 방향이며 구현 완료를 의미하지 않습니다.

---

## 20. 데이터 출처와 참고 문서

### 데이터 출처

직접 촬영 자료와 함께 다음 외부 자료를 프로젝트 데이터 수집·실험 과정에서 참고했습니다.
원본 링크와 개별 자료의 이용 조건은 각 배포 페이지에서 확인합니다.

- [Mendeley Data 데이터셋, 버전 2](https://data.mendeley.com/datasets/ptfscwtnyz/2)
- [Kaggle — Fruit Ripeness: Unripe, Ripe and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)
- [Kaggle — Riped and Unriped Tomato Dataset](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset)
(https://www.kaggle.com/datasets/techkhid/riped-and-unriped-tomato-dataset)
- [Roboflow Universe — 기존 README에 기록된 참조 링크](https://universe.roboflow.com/models/object-detection)

Roboflow 링크는 특정 데이터셋 버전을 식별하지 않는 기존 참조입니다.
실제 사용 파일과 원본 버전의 대응은 별도 확인이 필요합니다.

### 프로젝트 내부 문서

| 문서 | 내용 |
|---|---|
| [통합 모델 전환 기록](docs/joint_nine_migration.md) | 2026-10-23 전환 당시 기록 |
| [종류·모름 판정 v2](docs/fruit_unknown_detection_v2.md) | 이전 종류 인식 방식과 실험 |
| [평가 이력 설명](docs/performance_history.md) | 평가 이력 관련 문서 |
| [데이터 개발 기록](docs/dataset_development_log.md) | 데이터 구성·개발 기록 |
| [토마토 증강 실험](docs/tomato_augmentation_experiment.md) | 토마토 관련 과거 실험 |
| [토마토 F1 개선 비교](docs/tomato_f1_improvement_comparison.md) | 토마토 관련 비교 기록 |
| [9클래스 실험](experiments/nine_class_comparison/README.md) | 분리된 모델 비교 실험 |

과거 문서는 작성 당시의 구현·수치를 설명합니다.
현재 기본 실행은 이 README와 `main.py`, `src/joint.py`를 기준으로 확인합니다.
