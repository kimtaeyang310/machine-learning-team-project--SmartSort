# 성능 기록과 비교

기존 실행 방법을 그대로 사용합니다. 실행 시각은 한국 시간(UTC+09:00)입니다.

- `python -m src.train`: 학습이 완료되면 모델별 교차검증 결과를 `outputs/training/runs/날짜_시간/model_comparison.csv`에 보관합니다. `outputs/training/training_history.csv`와 `training_history.svg`도 갱신합니다. 기존 최상위 학습 파일은 최신 결과로 유지됩니다.
- `python -m src.evaluate`: 성공한 평가마다 `outputs/evaluation/evaluation_history.csv` 하나에 기록을 누적하고 `evaluation_history.svg`를 갱신합니다. 실행별 폴더, 사진별 예측·오분류·특징, 혼동행렬 CSV는 새로 저장하지 않습니다. 혼동행렬과 처리 실패 원인은 화면에서 확인합니다.
- `python -m src.performance_history`: 모델 실행 없이 누적 CSV 및 기존 평가 폴더에서 그래프를 다시 만듭니다. 기존 기록은 중복 없이 가져옵니다.


SVG는 브라우저에서 열 수 있습니다. 가로축은 실행 시각, 세로축은 0~1 점수입니다. 점 위에 마우스를 올리면 정확한 점수가 표시됩니다. 평가 그래프에는 Accuracy와 Macro F1, 학습 그래프에는 모델별 교차검증 Macro F1이 표시됩니다. CSV는 Excel에서 열 수 있는 UTF-8 BOM 형식입니다.

평가 CSV에는 그래프에 필요한 다섯 열만 저장합니다.

| 열 | 의미 |
|---|---|
| evaluated_at | 한국 시간 기준 실행 시각 |
| scope | 전체(overall) 또는 과일 종류 |
| dataset_id | 평가 사진 목록·정답·그룹 구분값 |
| accuracy | 정확도 |
| macro_f1 | 숙도별 F1의 평균 |

사진 목록이나 정답이 바뀌면 별도 그래프로 나뉩니다. 식별자는 이미지 파일 내용, 수동 마스크, 전처리 코드의 변경까지 검증하지는 않습니다.


기존 평가 폴더의 점수는 자동으로 가져오며 과거 상세 파일은 삭제하지 않습니다. 기존 폴더명 시각은 한국 시간으로 해석합니다. 학습 이력은 이 기능을 추가한 뒤 성공한 실행부터 쌓입니다. 과거에 덮어쓴 학습 결과는 복원하지 않습니다.

학습 교차검증 점수와 최종 Test 점수는 서로 다른 그래프입니다. Test 점수로 반복 튜닝했다면 해당 데이터는 개발에 영향을 준 것이므로, 최종 보고 성능은 별도의 미사용 Test로 확인합니다. 결과 파일은 저장소의 기존 `.gitignore` 규칙에 따라 로컬에 보관됩니다.
