# 🍎 SmartSort
### 농산물 숙도·품질 자동 선별기

**Team Project for Machine Learning Classification**

스마트폰으로 직접 촬영한 농산물 이미지를 분석하여  
농산물의 **숙도(Unripe / Ripe / Overripe)** 와  
**품질 상태(Normal / Defective)** 를 자동으로 판별하는  
머신러닝 기반 자동 선별 프로젝트입니다.

딥러닝 모델을 사용하지 않고,

**OpenCV 기반 이미지 처리 + Feature Engineering + Scikit-learn 기반 머신러닝 모델**

을 활용하여 농산물의 색상 및 표면 특징을 분석합니다.

---

# 📌 Project Information

| 항목 | 내용 |
|---|---|
| Project Name | `machine-learning-team-project--SmartSort` |
| Project Topic | 농산물 숙도·품질 자동 선별기 |
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
- LightGBM
- XGBoost

---

# 👥 Team

| 이름 | 담당 | 주요 역할 |
|---|---|---|
| 류채연 | **Data & Analysis** | 데이터 수집·정리, 라벨링, 데이터 품질 점검, EDA, 시각화, 발표 자료 정리 |
| 김태양 | **Model & Evaluation** | 이미지 전처리, 특징 추출, 머신러닝 모델 학습·비교, 튜닝, 성능 평가, 오분류 분석 |

> 최종 데이터 검수, Feature 선정, 결과 분석, README 및 발표 자료 작성은 두 팀원이 함께 진행합니다.

---

# 🎯 1. 프로젝트 소개

농산물은 숙성 과정에서 색상이 변화하고,  
시간이 지나면서 표면에 반점이나 흠집 등의 품질 변화가 발생할 수 있습니다.

예를 들어 바나나는 숙성됨에 따라 다음과 같은 변화를 보일 수 있습니다.

```text
초록색
  ↓
노란색
  ↓
갈색 반점 증가
```

SmartSort는 이러한 **색상 변화와 표면 특징을 이미지에서 추출**하여  
농산물의 상태를 자동으로 판별하는 것을 목표로 합니다.

본 프로젝트에서는 두 가지 분류 문제를 다룹니다.

### 숙도 분류

```text
Unripe
  ↓
Ripe
  ↓
Overripe
```

### 품질 상태 분류

```text
Normal
   /
Defective
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
- 농산물 품질 상태 2단계 분류
- OpenCV 기반 이미지 전처리
- K-means를 이용한 배경과 농산물 영역 분리
- HSV 기반 색상 특징 추출
- 농산물 표면의 흠집 및 반점 특징 추출
- Feature Dataset 생성
- 여러 머신러닝 모델 성능 비교
- Macro F1 Score 기반 모델 평가
- Confusion Matrix를 이용한 오분류 분석
- PCA를 활용한 데이터 및 특징 분포 분석
- CPU 환경에서도 빠르게 처리할 수 있는 판정 시스템 구현

---

# 🧩 3. 문제 정의

## 입력

흰색 배경 위에서 촬영한 농산물 1개 이미지

```text
Input Image
    │
    └── Banana / Mandarin / Tomato ...
```

---

## 출력 1 — 숙도 분류

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

## 출력 2 — 품질 상태 분류

농산물의 표면 상태를 기준으로 다음 2단계로 분류합니다.

| Label | Description |
|---|---|
| `normal` | 정상 |
| `defective` | 흠집·반점 등 이상 상태 |

예상 출력:

```text
Quality Prediction : normal
Confidence         : 91.2%
```

---

## 최종 출력 예시

```text
-----------------------------
SmartSort Prediction
-----------------------------

Ripeness : Ripe
Quality  : Normal

Ripeness Confidence : 89.4%
Quality Confidence  : 91.2%
```

---

# 📊 4. 목표 성능

| 항목 | 목표 |
|---|---|
| Ripeness Classification | Macro F1 Score ≥ 0.85 |
| Quality Classification | Macro F1 Score ≥ 0.85 |
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
표면 / 반점 특징 추출
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
        +
 ┌─────────────────────┐
 │ Quality Classifier  │
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
숙도와 표면 상태 변화를 촬영합니다.

예시:

```text
Day 1 → unripe / normal
Day 2 → unripe / normal
Day 3 → ripe / normal
Day 4 → ripe / defective
Day 5 → overripe / defective
Day 6 → overripe / defective
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
banana_day01_normal_001.jpg
banana_day01_normal_002.jpg
banana_day04_defective_001.jpg
```

---

# 🏷️ 8. Labeling

하나의 이미지에 **숙도 라벨과 품질 라벨을 각각 부여**합니다.

## Ripeness Label

```text
0 → unripe
1 → ripe
2 → overripe
```

## Quality Label

```text
0 → normal
1 → defective
```

예시 Metadata:

```csv
filename,ripeness,quality,date,light
banana_001.jpg,unripe,normal,2026-10-03,normal
banana_002.jpg,ripe,normal,2026-10-04,normal
banana_003.jpg,ripe,defective,2026-10-05,normal
banana_004.jpg,overripe,defective,2026-10-06,normal
```

두 팀원이 이미지를 확인하고 라벨링하며,  
판단이 다른 데이터는 상호 검토 후 최종 라벨을 결정합니다.

---

# ✂️ 9. Train / Test Split

동일한 농산물 또는 같은 날 촬영한 이미지가  
Train과 Test에 동시에 포함되지 않도록 주의합니다.

단순한 Random Split뿐 아니라  
**촬영 날짜 또는 농산물 단위 Group Split**을 고려합니다.

예시:

```text
Train
 ├── Day 1
 ├── Day 2
 ├── Day 3
 └── Day 4

Test
 ├── Day 5
 └── Day 6
```

이를 통해 비슷한 이미지가 Train과 Test에 동시에 포함되면서 발생할 수 있는  
**Data Leakage**를 줄입니다.

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

## Defect Features

품질 상태를 판단하기 위해  
농산물 표면의 흠집 및 반점 특징을 추출합니다.

- Dark Region Ratio
- Dark Pixel Ratio
- Defect Area
- Largest Defect Area
- Defect Count
- Connected Components Count

필요한 경우 다음 방법도 추가로 실험합니다.

- Threshold
- Connected Components
- DBSCAN
- LBP

---

# 🤖 13. Machine Learning Models

Feature Dataset을 이용하여 여러 머신러닝 모델을 학습하고 비교합니다.

## 기본 비교 모델

- Logistic Regression
- Support Vector Machine
- Random Forest

## 추가 실험 모델

프로젝트 진행 상황에 따라 다음 모델도 실험합니다.

- K-Nearest Neighbors
- Decision Tree
- LightGBM
- XGBoost

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

# 🔍 15. Quality Classification

품질 상태 분류 모델은  
색상뿐만 아니라 표면의 반점과 흠집 정보를 중요하게 사용합니다.

```text
Defect Features
      +
Dark Region Ratio
      +
Connected Components
      +
Color Features
      ↓
Machine Learning Model
      ↓
Normal / Defective
```

---

# 🔬 16. Feature Ablation

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
+ Defect Features
```

이를 통해 숙도 분류와 품질 분류에  
어떤 Feature가 가장 중요한지 확인합니다.

---

# 📊 17. Model Evaluation

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
Quality Macro F1  >= 0.85
```

---

# 📉 18. PCA & Error Analysis

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
- 품질 상태별 데이터 분포
- 클래스 간 겹치는 영역
- 이상치
- 오분류 데이터
- Feature 분리 가능성

---

# ⚡ 19. Performance

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

# 👥 20. Team Roles

## 🅰️ 류채연 — Data & Analysis

### 주요 담당

- 스마트폰 이미지 데이터 수집
- 촬영 규칙 관리
- 파일명 및 데이터 폴더 관리
- Ripeness Labeling
- Quality Labeling
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

## 🅱️ 김태양 — Model & Evaluation

### 주요 담당

- OpenCV 이미지 전처리
- K-means 기반 배경 제거
- Fruit Mask 생성
- HSV Feature Extraction
- Color Ratio Feature Extraction
- Defect Feature Extraction
- Feature Dataset 생성
- 숙도 분류 모델 구현
- 품질 분류 모델 구현
- Logistic Regression / SVM / Random Forest 비교
- LightGBM / XGBoost 추가 실험
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

# 📅 21. 6-Day Project Plan

| Day | 주요 작업 |
|---|---|
| Day 1 | 프로젝트 설계, 데이터 촬영 기준 확정, 초기 데이터 수집 |
| Day 2 | 데이터 수집 및 라벨링, Metadata 작성, EDA |
| Day 3 | 이미지 전처리, K-means 배경 제거, Mask 생성 |
| Day 4 | HSV / Color Ratio / Defect Feature 추출, Feature Dataset 생성 |
| Day 5 | 머신러닝 모델 학습·비교, Hyperparameter Tuning, 평가 |
| Day 6 | PCA·오분류 분석, 최종 모델 선정, Prediction Pipeline, README·PPT 정리 |

---

# ✅ 22. Project Progress

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
- [ ] 품질 상태 라벨링
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
- [ ] Defect Feature 추출
- [ ] Feature Dataset 생성

---

## Phase 05 — Machine Learning

- [ ] Logistic Regression
- [ ] SVM
- [ ] Random Forest
- [ ] LightGBM
- [ ] XGBoost
- [ ] Hyperparameter Tuning
- [ ] Feature Ablation

---

## Phase 06 — Evaluation

- [ ] Ripeness Classification 평가
- [ ] Quality Classification 평가
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

# 🚀 23. Expected Result

최종적으로 이미지 한 장을 입력하면  
숙도와 품질 상태를 함께 예측하는 것을 목표로 합니다.

```text
             Input Image
                  ↓
            Preprocessing
                  ↓
          Fruit Segmentation
                  ↓
          Feature Extraction
                  ↓
      ┌───────────┴───────────┐
      ↓                       ↓
Ripeness Model           Quality Model
      ↓                       ↓
Unripe / Ripe /        Normal /
Overripe               Defective
      │                       │
      └───────────┬───────────┘
                  ↓
        Final Prediction
```

예상 결과:

```text
--------------------------------
SmartSort Prediction
--------------------------------

Ripeness   : Ripe
Quality    : Normal

Ripeness Confidence : 89.4%
Quality Confidence  : 91.2%
```

여러 이미지를 처리할 경우 결과를 CSV 형태로 저장합니다.

```csv
filename,ripeness,ripeness_confidence,quality,quality_confidence
banana_001.jpg,ripe,0.894,normal,0.912
banana_002.jpg,unripe,0.921,normal,0.934
banana_003.jpg,overripe,0.861,defective,0.887
```

---

# 🛠️ 24. Development Environment

- Python
- VS Code
- Jupyter Notebook
- Git
- GitHub

### 주요 라이브러리

```text
numpy
pandas
matplotlib
scikit-learn
opencv-python
lightgbm
xgboost
```

설치:

```bash
pip install -r requirements.txt
```

---

# 🌿 25. Git Collaboration

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

# 📁 26. 프로젝트 구조

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

> **Machine Learning Based Agricultural Ripeness & Quality Classification System**

농산물 이미지에서 색상과 표면 특징을 추출하여  
농산물의 **숙도와 품질 상태를 자동으로 판별**합니다.

```text
Image
  ↓
Feature Extraction
  ↓
Machine Learning
  ↓
Ripeness + Quality
  ↓
Smart Sorting
```