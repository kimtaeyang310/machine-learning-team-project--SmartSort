# 🍎 SmartSort
### 농산물 숙도 자동 분류기

**Team Project for Machine Learning Classification**

스마트폰으로 직접 촬영한 농산물 이미지를 분석하여  
농산물의 **숙도(Unripe / Ripe / Overripe)** 를 자동으로 판별하는
머신러닝 기반 숙도 분류 프로젝트입니다. 데이터 확보 범위를 고려하여 품질 분류는 수행하지 않습니다.

딥러닝 모델을 사용하지 않고,

**OpenCV 기반 이미지 처리 + Feature Engineering + Scikit-learn 기반 머신러닝 모델**

을 활용하여 농산물의 색상 및 표면 특징을 분석합니다.

---

# 📌 Project Information

| 항목 | 내용 |
|---|---|
| Project Name | `machine-learning-team-project--SmartSort` |
| Project Topic | 농산물 숙도 자동 분류기 |
| Project Type | Machine Learning Mini Project |
| Duration | 6일 |
| Team Size | 2명 |
| Main Task | Image Classification |
| Main Language | Python |

### Main Keywords

- Machine Learning
- Image Processing
- OpenCV
- K-means
- HSV
- Feature Engineering
- Classification
- PCA
- Scikit-learn
- K-Nearest Neighbors (KNN)
- Decision Tree (DT)

---

# 👥 Team

| 이름 | 담당 | 주요 역할 |
|---|---|---|
| 김태양 | **Data & Analysis** | 데이터 수집·정리, 라벨링, 데이터 품질 점검, EDA, 시각화, 발표 자료 정리 |
| 류채연 | **Model & Evaluation** | 이미지 전처리, 특징 추출, 머신러닝 모델 학습·비교, 튜닝, 성능 평가, 오분류 분석 |

> 최종 데이터 검수, Feature 선정, 결과 분석, README 및 발표 자료 작성은 두 팀원이 함께 진행합니다.

---

# 🎯 1. 프로젝트 소개

농산물은 숙성 과정에서 색상이 변화하고,  
시간이 지나면서 갈변이나 표면 색상 변화가 발생할 수 있습니다.

예를 들어 바나나는 숙성됨에 따라 다음과 같은 변화를 보일 수 있습니다.

```text
초록색
  ↓
노란색
  ↓
갈색 반점 증가
```

SmartSort는 이러한 **색상 변화와 표면 특징을 이미지에서 추출**하여  
농산물의 숙도를 자동으로 판별하는 것을 목표로 합니다.

본 프로젝트에서는 숙도 3단계 분류 문제를 다룹니다.

### 숙도 분류

```text
Unripe
  ↓
Ripe
  ↓
Overripe
```

프로젝트의 기본 처리 방식은 다음과 같습니다.

```text
OpenCV Image Processing
          +
Feature Engineering
          +
Machine Learning
```

---

# 🎯 2. 프로젝트 목표

주요 목표는 다음과 같습니다.

- 스마트폰을 이용한 직접 이미지 데이터셋 구축
- 농산물 숙도 3단계 분류
- OpenCV 기반 이미지 전처리
- K-means를 이용한 배경과 농산물 영역 분리
- HSV 기반 색상 특징 추출
- 숙도 분류를 위한 밝기·채도·영역 특징 추출
- Feature Dataset 생성
- 여러 머신러닝 모델 성능 비교
- Macro F1 Score 기반 모델 평가
- Confusion Matrix를 이용한 오분류 분석
- PCA를 활용한 데이터 및 특징 분포 분석
- CPU 환경에서도 빠르게 처리할 수 있는 판정 시스템 구현

---

# 🧩 3. 문제 정의

## 입력

흰색 배경 위에서 촬영한 농산물 1개 이미지와 과일 종류(banana / mandarin / tomato)

```text
Input Image
    │
    └── Banana / Mandarin / Tomato ...
```

---

## 출력 — 숙도 분류

농산물의 숙도를 다음 3단계로 분류합니다.

| Label | Description |
|---|---|
| `unripe` | 덜 익음 |
| `ripe` | 적당히 익음 |
| `overripe` | 너무 익음 |

예상 출력:

```text
Ripeness Prediction : ripe
Confidence          : 89.4%
```

---

## 최종 출력 예시

```text
-----------------------------
SmartSort Prediction
-----------------------------

Ripeness : Ripe

Ripeness Confidence : 89.4%
```

---

# 📊 4. 목표 성능

| 항목 | 목표 |
|---|---|
| Ripeness Classification | Macro F1 Score ≥ 0.85 |
| Processing Speed | 25 FPS 이상 |
| Generalization | 다른 날짜·조명 환경에서도 안정적인 분류 |

---

# 🔄 5. 전체 Machine Learning Pipeline

```text
스마트폰 이미지 촬영
        ↓
데이터 라벨링
        ↓
이미지 전처리
        ↓
K-means 기반 색상 분할
        ↓
배경 제거
        ↓
농산물 영역 Mask 생성
        ↓
HSV 색상 특징 추출
        ↓
색상 비율 특징 추출
        ↓
밝기 / 채도 / 영역 특징 추출
        ↓
Feature Dataset 생성
        ↓
Train / Test Split
        ↓
Machine Learning Model 학습
        ↓
 ┌─────────────────────┐
 │ Ripeness Classifier │
 └─────────────────────┘
        ↓
모델 성능 비교
        ↓
최종 모델 선정
        ↓
Prediction + Confidence 출력
```

---

# 📷 6. Dataset

본 프로젝트에서는 외부 데이터셋에만 의존하지 않고  
**팀원이 직접 스마트폰으로 촬영한 이미지 데이터**를 사용합니다.

## 데이터 수집

같은 품종의 농산물을 일정 기간 보관하면서  
숙도에 따른 외관 변화를 촬영합니다.

예시:

```text
Day 1 → unripe
Day 2 → unripe
Day 3 → ripe
Day 4 → ripe
Day 5 → overripe
Day 6 → overripe
```

최소 **100장 이상의 이미지 확보**를 목표로 합니다.

---

# 📏 7. 촬영 규칙

촬영 환경에 따른 불필요한 차이를 줄이기 위해  
다음과 같은 촬영 규칙을 적용합니다.

- 흰색 배경 사용
- 스마트폰 수직 촬영
- 동일하거나 유사한 촬영 거리 유지
- 동일 품종 사용
- 촬영 날짜 기록
- 조명 조건 기록
- 가능한 동일한 장소에서 촬영
- 파일명에 촬영 정보를 기록

예시:

```text
20261003_tomato_T01_indoor_01.jpg
20261003_tomato_T01_indoor_02.jpg
20261006_tomato_T01_indoor_01.jpg
```

---

# 🏷️ 8. Labeling

이미지에 **숙도 라벨**을 부여합니다. 직접 촬영한 귤·토마토는 개체·날짜 단위로 라벨과 split을 관리하고, 바나나는 폴더명에서 숙도 라벨을 읽습니다.

## Ripeness Label

```text
0 → unripe
1 → ripe
2 → overripe
```

직접 촬영 데이터의 Metadata 예시 (`data/metadata/fruit_labels.csv`):

```csv
fruit,fruit_id,date,ripeness,group_id,split
tomato,T01,2026-10-03,unripe,T01_20261003,train
tomato,T01,2026-10-04,ripe,T01_20261004,train
tomato,T02,2026-10-04,ripe,T02_20261004,test
mandarin,M01,2026-10-06,overripe,M01_20261006,train
```

두 팀원이 이미지를 확인하고 라벨링하며,  
판단이 다른 데이터는 상호 검토 후 최종 라벨을 결정합니다.

---

# ✂️ 9. Train / Test Split

- 직접 촬영한 귤·토마토는 Metadata에 지정한 `train` / `test`를 사용합니다.
- 바나나는 같은 원본의 증강 사진을 하나의 그룹으로 묶어 숙도별 원본 그룹의 약 20%를 Test에 배정합니다.
- 학습과 Test 사이에 같은 그룹이 들어가지 않도록 검사합니다.
- Train을 과일별로 나누고 각 과일 안에서 동일한 3-fold StratifiedGroupKFold로 5개 모델을 비교합니다.

현재 직접 촬영 데이터의 `group_id`는 **개체 ID + 촬영 날짜**입니다. 따라서 같은 개체의 다른 날짜 사진은 서로 다른 그룹으로 취급됩니다. 이 방식은 새 개체에 대한 일반화 성능을 완전히 보장하지 않으며, 개체 단위 검증은 별도의 개선 항목입니다.

Test는 최종 평가에 사용합니다. Test 결과를 반복해서 보며 설정을 바꿨다면 최종 보고용으로 별도의 미사용 데이터를 확보해야 합니다.

---

# 🖼️ 10. Image Preprocessing

머신러닝에 필요한 특징을 추출하기 전에  
OpenCV를 이용하여 이미지를 전처리합니다.

주요 작업:

- 이미지 크기 통일
- 이미지 노이즈 제거
- BGR → HSV 변환
- 배경 제거
- 농산물 영역 Mask 생성
- Morphology 연산
- 필요 시 Blur 적용

---

# 🎨 11. K-means Background Segmentation

흰색 배경과 농산물 영역을 분리하기 위해  
**K-means Clustering**을 활용합니다.

```text
Original Image
      ↓
Pixel Data 변환
      ↓
K-means Clustering
      ↓
K = 2 ~ 3
      ↓
Background / Fruit 분리
      ↓
Fruit Mask 생성
```

배경을 제거한 뒤  
농산물 영역만을 기준으로 특징을 추출합니다.

---

# 🌈 12. Feature Extraction

이미지 전체가 아닌  
**농산물 영역에서만 특징을 추출**합니다.

## Color Features

숙도 분류를 위해 HSV 색공간에서 색상 정보를 추출합니다.

- Mean H
- Mean S
- Mean V
- Standard Deviation H
- Standard Deviation S
- Standard Deviation V
- HSV Histogram
- Color Moments

---

## Color Ratio Features

농산물의 숙도 변화에 따른 특정 색상 비율을 계산합니다.

- Green Pixel Ratio
- Yellow Pixel Ratio
- Brown Pixel Ratio

예시:

| H_mean | S_mean | V_mean | Green Ratio | Yellow Ratio | Brown Ratio | Ripeness |
|---:|---:|---:|---:|---:|---:|---|
| 72.1 | 145.2 | 178.3 | 0.61 | 0.21 | 0.02 | unripe |
| 54.3 | 161.8 | 184.2 | 0.31 | 0.54 | 0.05 | ripe |
| 32.8 | 132.4 | 151.8 | 0.08 | 0.40 | 0.29 | overripe |

---

## Brightness / Saturation / Area Features

숙도에 따른 어두운 영역과 채도 변화를 수치화하며, 품질 라벨을 예측하지 않습니다.

- Dark Ratio: 과일 영역에서 V < 60인 픽셀 비율
- Low Saturation Ratio: 과일 영역에서 S < 40인 픽셀 비율
- Area Ratio: 패딩 전 영상에서 과일 영역 비율
- Fruit Pixel Count: 과일 마스크의 픽셀 수

---

# 🤖 13. Machine Learning Models

`src/train.py`에서 **바나나·귤·토마토를 분리하여 각각 다음 5개 숙도 분류 모델을 비교**합니다. 각 과일은 자기 과일의 학습 사진만 사용하고, 과일별로 최고 모델을 선택합니다.

| 모델 | 전처리 | 주요 초기 설정 |
|---|---|---|
| Logistic Regression | StandardScaler | max_iter=3000, class_weight=balanced |
| SVM | StandardScaler | RBF kernel, C=1.0, gamma=scale, class_weight=balanced |
| Random Forest | 별도 표준화 없음 | n_estimators=300, min_samples_leaf=2, class_weight=balanced |
| K-Nearest Neighbors (KNN) | StandardScaler | n_neighbors=5, weights=uniform |
| Decision Tree (DT) | 별도 표준화 없음 | max_depth=5, min_samples_leaf=2, class_weight=balanced |

KNN은 특징 간 거리로 판단하므로 단위가 다른 특징을 표준화합니다. DT는 과적합을 줄이기 위해 깊이와 말단 노드의 최소 샘플 수를 제한합니다. KNN·DT 설정은 초기 비교용이며 최적화된 값이 아닙니다.

- 과일별로 3-fold StratifiedGroupKFold를 구성하고, 해당 과일의 5개 후보는 동일한 특징과 분할로 비교합니다.
- StandardScaler는 Pipeline 내부에서 각 fold의 학습 데이터에만 맞춥니다.
- 과일별 교차검증 Macro F1 평균이 가장 높은 모델을 선택하고 해당 과일의 Train 전체로 재학습합니다. 바나나·귤·토마토에서 서로 다른 알고리즘이 선택될 수 있습니다.
- Test는 모델 선택이나 튜닝에 사용하지 않으며, 최종 평가는 `src/evaluate.py`에서 수행합니다.
- 모델별 평균·표준편차·fold별 점수는 `outputs/training/by_fruit/model_comparison.csv`에 저장됩니다.
- 통합 비교 CSV에는 과일마다 5행씩 총 15행과 `fruit`, `selected` 열이 저장됩니다.
- 과일별 상세 비교·누적 학습 기록·그래프는 `outputs/training/by_fruit/banana/`, `mandarin/`, `tomato/`에 저장됩니다.
- 기존 통합 모델과 과거 학습 결과는 보존합니다. 과일별 모델의 결과는 새로운 학습 경로에 저장됩니다.
- 과일별 데이터에 숙도 3개가 모두 있어야 하며 각 검증 fold에도 3개 숙도가 필요합니다. 부족하면 분할 기준을 임의로 완화하지 않고 오류를 표시합니다.

실행 (프로젝트 최상위 폴더):

```bash
python -m src.train
python -m src.evaluate
python -m src.predict "data/raw/이미지경로.jpg" --fruit tomato
```

학습을 실행하면 `models/ripeness_by_fruit.joblib`에 과일별 분류기 3개와 공통 전처리 설정을 하나의 묶음으로 저장합니다. 다음 실행부터 이 파일과 `outputs/training/by_fruit/`의 최신 결과를 갱신하며 실행별 비교 기록은 보관합니다. 기존 `models/ripeness_model.joblib`은 자동으로 대체하거나 사용하지 않습니다.

새 이미지 예측에는 `--fruit banana`, `--fruit mandarin`, `--fruit tomato` 중 하나가 필요합니다. 과일 종류를 자동 인식하는 기능은 없으며, 사용자가 잘못 지정하면 다른 과일 모델로 예측할 수 있습니다. `python main.py` 메뉴에서도 과일 종류를 입력합니다.

평가는 저장된 Test 목록의 `fruit` 열에 따라 모델을 선택합니다. 전체·과일별 점수는 기존 `outputs/evaluation/evaluation_history.csv`와 그래프에 계속 누적하므로 같은 Test 목록이면 과거 통합 모델 결과와 비교할 수 있습니다. 이는 **과일 종류가 주어진 조건**에서의 숙도 성능이며, 과일 종류 자동 인식 성능을 포함하지 않습니다.

---

## 과일별 모델 첫 실행 결과 (2026-10-13)

학습 2,086장·Test 491장의 기존 분할을 유지하고, 학습 내부 교차검증으로만 모델을 선택했습니다.

| 과일 | 선택 모델 | CV Macro F1 | Test Macro F1 | Test Accuracy |
|---|---|---:|---:|---:|
| 바나나 | Random Forest | 0.9354 | 0.9048 | 0.9044 |
| 귤 | Random Forest | 0.9267 | 0.8413 | 0.8571 |
| 토마토 | SVM | 0.7592 | 0.3500 | 0.5273 |

전체 Test Macro F1은 0.8538, Accuracy는 0.8554입니다. 직전 통합 KNN 대비 귤은 개선됐으나 바나나·토마토는 하락했습니다. 과일별 분리가 성능 향상을 보장하지 않으며, 토마토는 데이터·특징·검증 분할의 추가 분석이 필요합니다. 이 결과를 근거로 Test에서 가장 잘 나온 후보를 다시 선택하지는 않았습니다.

---

# 🍌 14. Ripeness Classification

숙도 분류 모델은 주로 **색상 기반 Feature**를 사용합니다.

```text
HSV Features
      +
Color Ratio
      +
Color Moments
      ↓
Machine Learning Model
      ↓
Unripe / Ripe / Overripe
```

---

# 🔬 15. Feature Ablation

각 특징이 모델 성능에 얼마나 영향을 미치는지 확인하기 위해  
Feature Ablation 실험을 진행합니다.

예시:

```text
Experiment 1
HSV Features

        ↓

Experiment 2
HSV Features
+ Color Ratio

        ↓

Experiment 3
HSV Features
+ Color Ratio
+ Brightness / Saturation / Area Features
```

이를 통해 숙도 분류에
어떤 Feature가 가장 중요한지 확인합니다.

---

# 📊 16. Model Evaluation

모델 성능은 다음 지표를 사용해 평가합니다.

- Accuracy
- Precision
- Recall
- F1 Score
- Macro F1 Score
- Confusion Matrix

특히 클래스별 데이터 수의 차이를 고려하여  
**Macro F1 Score를 주요 평가 지표**로 사용합니다.

목표:

```text
Ripeness Macro F1 >= 0.85
```

---

# 📉 17. PCA & Error Analysis

모델의 결과를 점수만 확인하는 것이 아니라  
오분류된 데이터를 직접 분석합니다.

```text
Feature Dataset
      ↓
Standard Scaling
      ↓
PCA
      ↓
2D Feature Space
```

PCA 결과를 이용하여 다음 내용을 확인합니다.

- 숙도별 데이터 분포
- 클래스 간 겹치는 영역
- 이상치
- 오분류 데이터
- Feature 분리 가능성

---

# ⚡ 18. Performance

분류 정확도뿐만 아니라  
이미지 한 장을 처리하는 속도도 측정합니다.

측정 항목:

- Image Load Time
- Preprocessing Time
- Segmentation Time
- Feature Extraction Time
- Prediction Time
- Total Processing Time
- FPS

FPS 계산:

```text
FPS = 1 / 이미지 1장 평균 처리 시간
```

목표:

```text
25 FPS 이상
```

---

# 👥 19. Team Roles

## 🅰️ 김태양 — Data & Analysis

### 주요 담당

- 스마트폰 이미지 데이터 수집
- 촬영 규칙 관리
- 파일명 및 데이터 폴더 관리
- Ripeness Labeling
- 라벨 품질 검토
- Metadata 관리
- Train / Test Group 분할
- 데이터 클래스 분포 확인
- EDA
- 데이터 시각화
- 모델 결과 시각화 지원
- 발표 자료 정리

### 주요 산출물

```text
data/
metadata/
labels.csv
EDA Notebook
Dataset Statistics
Visualization
Presentation Material
```

---

## 🅱️ 류채연 — Model & Evaluation

### 주요 담당

- OpenCV 이미지 전처리
- K-means 기반 배경 제거
- Fruit Mask 생성
- HSV Feature Extraction
- Color Ratio Feature Extraction
- Brightness / Saturation / Area Feature Extraction
- Feature Dataset 생성
- 숙도 분류 모델 구현
- Logistic Regression / SVM / Random Forest / KNN / DT 비교
- Hyperparameter Tuning
- Macro F1 Score 평가
- Confusion Matrix 분석
- PCA 기반 오분류 분석
- 최종 Prediction Pipeline 구성

### 주요 산출물

```text
src/preprocessing.py
src/feature_extraction.py
src/train.py
src/evaluate.py
src/predict.py
models/
Evaluation Results
```

---

## 🤝 공동 작업

- 데이터 최종 검수
- Feature 선정
- 모델 비교
- 최종 모델 선정
- 결과 분석
- README 작성
- 프로젝트 보고서 작성
- 발표 자료 제작
- 최종 발표 준비

---

# 📅 20. 6-Day Project Plan

| Day | 주요 작업 |
|---|---|
| Day 1 | 프로젝트 설계, 데이터 촬영 기준 확정, 초기 데이터 수집 |
| Day 2 | 데이터 수집 및 라벨링, Metadata 작성, EDA |
| Day 3 | 이미지 전처리, K-means 배경 제거, Mask 생성 |
| Day 4 | HSV / Color Ratio / Brightness / Saturation / Area Feature 추출, Feature Dataset 생성 |
| Day 5 | 머신러닝 모델 학습·비교, Hyperparameter Tuning, 평가 |
| Day 6 | PCA·오분류 분석, 최종 모델 선정, Prediction Pipeline, README·PPT 정리 |

---

# ✅ 21. Project Progress

## Phase 01 — Project Setup

- [x] 프로젝트 주제 선정
- [x] GitHub Repository 생성
- [x] 팀원 구성
- [x] 초기 역할 분담
- [x] README 초안 작성
- [x] requirements.txt 작성
- [x] 프로젝트 폴더 구조 생성

---

## Phase 02 — Dataset

- [x] 촬영 대상 농산물 선정
- [x] 촬영 환경 구성
- [ ] 이미지 데이터 수집
- [ ] 클래스별 100장 이상 데이터 확보
- [ ] 숙도 라벨링
- [ ] 라벨 품질 검토
- [ ] Metadata 생성
- [ ] Train / Test 데이터 분리

---

## Phase 03 — Image Processing

- [ ] 이미지 크기 통일
- [ ] HSV 변환
- [ ] K-means 색상 분할
- [ ] 배경 제거
- [ ] Fruit Mask 생성

---

## Phase 04 — Feature Engineering

- [ ] HSV Feature 추출
- [ ] HSV Histogram 생성
- [ ] Color Moments 추출
- [ ] Green Ratio 계산
- [ ] Yellow Ratio 계산
- [ ] Brown Ratio 계산
- [ ] Brightness / Saturation / Area Feature 추출
- [ ] Feature Dataset 생성

---

## Phase 05 — Machine Learning

- [x] Logistic Regression 구현
- [x] SVM 구현
- [x] Random Forest 구현
- [x] KNN 구현
- [x] Decision Tree 구현
- [x] 과일별 5개 모델 실제 데이터 학습·성능 비교
- [ ] Hyperparameter Tuning
- [ ] Feature Ablation

---

## Phase 06 — Evaluation

- [ ] Ripeness Classification 평가
- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] Macro F1 Score
- [ ] Confusion Matrix
- [ ] PCA
- [ ] 오분류 분석
- [ ] FPS 측정

---

## Phase 07 — Final

- [ ] 최종 모델 선정
- [ ] Prediction Pipeline 구현
- [ ] README 최종 수정
- [ ] 결과 그래프 정리
- [ ] 발표 자료 제작
- [ ] 최종 발표

---

# 🚀 22. Expected Result

최종적으로 이미지 한 장을 입력하면  
미숙·적숙·과숙 중 하나의 숙도를 예측합니다.

```text
             Input Image
                  ↓
            Preprocessing
                  ↓
          Fruit Segmentation
                  ↓
          Feature Extraction
                  ↓
        Selected Fruit Ripeness Model
                  ↓
       Unripe / Ripe / Overripe
                  ↓
          Ripeness Prediction
```

예상 결과:

```text
--------------------------------
SmartSort Prediction
--------------------------------

Ripeness   : Ripe

Ripeness Confidence : 89.4%
```

수치는 출력 형식 예시입니다. 예측 확률은 `predict_proba`를 지원하는 모델에서만 표시되며, 현재 SVM 설정은 확률을 출력하지 않습니다. 이 값은 보정된 정답 확률을 의미하지 않습니다.

일괄 예측 기능을 추가할 경우 사용할 CSV 형식 예시입니다. 현재 예측 명령은 이미지 한 장의 결과를 화면에 출력합니다.

```csv
filename,ripeness,ripeness_confidence
banana_001.jpg,ripe,0.894
banana_002.jpg,unripe,0.921
banana_003.jpg,overripe,0.861
```

---

# 🛠️ 23. Development Environment

- Python
- VS Code
- Jupyter Notebook
- Git
- GitHub

### 주요 라이브러리

```text
numpy
pandas
scikit-learn
opencv-python
joblib
```

설치:

```bash
pip install -r requirements.txt
```

---

# 🌿 24. Git Collaboration

팀원별 Feature Branch를 생성하여 작업합니다.

```text
main
│
├── feature/taeyang
│
└── feature/chaeyeon
```

작업 시작 전:

```bash
git checkout main
git pull origin main
```

개인 브랜치로 이동:

```bash
git checkout feature/taeyang
```

작업 후:

```bash
git add .
git commit -m "Add feature extraction"
git push origin feature/taeyang
```

GitHub에서 Pull Request를 생성한 후  
코드를 확인하고 `main` 브랜치에 병합합니다.

---

# 📁 25. 프로젝트 구조

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── dataset.py
│   ├── train.py
│   ├── model_bundle.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
├── outputs/
├── tests/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🍎 SmartSort

> **Machine Learning Based Agricultural Ripeness Classification System**

농산물 이미지에서 색상과 표면 특징을 추출하여  
농산물의 **숙도를 자동으로 판별**합니다.

```text
Image
  ↓
Feature Extraction
  ↓
Machine Learning
  ↓
Ripeness
  ↓
Smart Sorting
```
