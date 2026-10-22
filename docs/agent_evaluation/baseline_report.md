# 재현 가능한 에이전트 기준선 평가

- 평가 이미지: 58장 / 분할: {'test': 58}
- 모델 예측 정확도: 39/58 (67.2%)
- 행동 분포: {'auto_classify': 54, 'retake': 2, 'human_review': 2}
- 자동 분류 커버리지: 54/58 (93.1%)
- 자동 분류 오분류: 15/54 (27.8%)
- 비자동 행동으로 라우팅된 모델 오분류: 4건

## 과일·숙도별 모델 지표

| 과일 | 숙도 | 표본 수 | 정밀도 | 재현율 |
|---|---|---:|---:|---:|
| banana | unripe | 4 | 50.0% | 75.0% |
| banana | ripe | 12 | 90.0% | 75.0% |
| banana | overripe | 14 | 100.0% | 100.0% |
| mandarin | unripe | 3 | 23.1% | 100.0% |
| mandarin | ripe | 10 | N/A | 0.0% |
| mandarin | overripe | 0 | N/A | N/A |
| tomato | unripe | 2 | 100.0% | 100.0% |
| tomato | ripe | 10 | 72.7% | 80.0% |
| tomato | overripe | 3 | 0.0% | 0.0% |

## 혼동 행렬 (실제 행 -> 예측 열)

| 과일 | 실제 숙도 | 미숙 예측 | 적숙 예측 | 과숙 예측 |
|---|---|---:|---:|---:|
| banana | unripe | 3 | 1 | 0 |
| banana | ripe | 3 | 9 | 0 |
| banana | overripe | 0 | 0 | 14 |
| mandarin | unripe | 3 | 0 | 0 |
| mandarin | ripe | 10 | 0 | 0 |
| mandarin | overripe | 0 | 0 | 0 |
| tomato | unripe | 2 | 0 | 0 |
| tomato | ripe | 0 | 8 | 2 |
| tomato | overripe | 0 | 3 | 0 |

## 자동 분류 오분류 이미지

| 파일 | 실제 | 예측 | 확률 | 과일 개체 그룹 |
|---|---|---|---:|---|
| 20261008_mandarin_M10_indoor_02.jpg | ripe | unripe | 92.6% | M10_20261008 |
| 20261008_mandarin_M10_indoor_03.jpg | ripe | unripe | 97.1% | M10_20261008 |
| 20261008_mandarin_M10_indoor_04.jpg | ripe | unripe | 73.1% | M10_20261008 |
| 20261008_mandarin_M10_indoor_05.jpg | ripe | unripe | 98.0% | M10_20261008 |
| 20261008_mandarin_M10_indoor_dim_06.jpg | ripe | unripe | 99.0% | M10_20261008 |
| 20261008_mandarin_M10_indoor_dim_07.jpg | ripe | unripe | 99.5% | M10_20261008 |
| 20261008_mandarin_M10_indoor_dim_08.jpg | ripe | unripe | 99.5% | M10_20261008 |
| 20261008_mandarin_M10_indoor_dim_09.jpg | ripe | unripe | 91.8% | M10_20261008 |
| 20261008_mandarin_M10_indoor_dim_10.jpg | ripe | unripe | 98.2% | M10_20261008 |
| 20261008_tomato_T09_indoor_dim_08.jpg | overripe | ripe | 81.5% | T09_20261008 |
| 20261008_tomato_T09_indoor_dim_09.jpg | overripe | ripe | 88.0% | T09_20261008 |
| 20261008_tomato_T09_indoor_dim_10.jpg | overripe | ripe | 85.6% | T09_20261008 |
| banana25.jpg | ripe | unripe | 99.4% | mendeley_banana_ripe_202 |
| banana26.jpg | ripe | unripe | 98.9% | mendeley_banana_ripe_202 |
| banana30.jpg | unripe | ripe | 72.6% | mendeley_banana_semi_ripe_187 |

## 비자동 행동으로 라우팅된 오분류

| 파일 | 실제 | 예측 | 행동 | 확률 |
|---|---|---|---|---:|
| 20261008_mandarin_M10_indoor_01.jpg | ripe | unripe | retake | 61.7% |
| 20261008_tomato_T10_indoor_dim_06.jpg | ripe | overripe | human_review | 86.1% |
| 20261008_tomato_T10_indoor_dim_07.jpg | ripe | overripe | human_review | 93.1% |
| banana15.jpg | ripe | unripe | retake | 59.0% |

## 표본 독립성

| 과일 | 이미지 수 | 서로 다른 개체 그룹 수 |
|---|---:|---:|
| banana | 30 | 26 |
| mandarin | 13 | 2 |
| tomato | 15 | 4 |
