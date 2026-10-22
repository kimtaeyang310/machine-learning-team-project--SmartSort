# 에이전트 기준선과 추가 검증 표본 계획

## 기준선 고정

- `baseline_predictions.csv`는 제공된 58장 실행 로그를 정답 라벨과 대조해 만든 고정 스냅샷입니다.
- 원본 출력은 `source_run_output.txt`, 파일 해시와 출처 한계는 `baseline_manifest.json`에 보존했습니다.
- 귤·토마토 정답은 `data/metadata/fruit_labels.csv`, 바나나 정답은 원본 라벨 폴더와 정확히 일치하는 이미지 파일 해시로 연결했습니다.
- 모든 58장은 현재 데이터 분할에서 `test`입니다. 이 자료는 기준선으로 보존하고 임계값을 조정하는 데 재사용하지 않습니다.
- 재평가 명령: `python scripts/evaluate_agent_baseline.py`
- 보고서를 다시 저장하려면 `python scripts/evaluate_agent_baseline.py --output docs/agent_evaluation/baseline_report.md`를 실행합니다.
- 이번 로그에는 실행 시각, 모델 가중치 해시, 라이브러리 버전이 없어 완전한 실행환경 재현은 보장하지 않습니다. 다음 예측부터는 결과 CSV와 함께 모델 파일 SHA-256, 코드 커밋, 실행 환경을 기록해야 합니다.

## 다음 실행 결과 보존

프로젝트 루트에서 예측 CSV와 manifest를 함께 생성합니다.

```powershell
python -m src.predict data/predict_list --results-csv docs/agent_evaluation/next_run.csv
```

새 표본의 정답을 `validation_capture_template.csv` 형식으로 별도 CSV에 입력합니다. `ground_truth_status`는 사람이 라벨을 확인한 행만 `confirmed` 또는 `verified`로 설정합니다. 예측값을 보지 않고 정답을 확정하며, `image_name`이 예측 결과 CSV와 일치해야 합니다. 두 파일을 매칭해 평가하려면 다음처럼 실행합니다.

```powershell
python scripts/evaluate_agent_baseline.py docs/agent_evaluation/next_run.csv --labels-csv docs/agent_evaluation/validation_labels.csv --output docs/agent_evaluation/next_run_report.md
```

검증 표본 CSV와 결과는 새 실행마다 별도 이름으로 보존합니다. 기준선 CSV는 수정하지 않습니다.

## 이번 기준선에서 확인된 표본 공백

이번 test는 이미지 58장이지만 개체 그룹은 32개입니다. 여러 장이 같은 개체를 다른 조건에서 촬영한 것이므로 이미지 수를 독립 표본 수로 보면 안 됩니다.

- 귤: 13장, 2개체(M09, M10). 적숙 표본은 M10 한 개체의 사진 10장뿐이며 모두 미숙으로 오분류됐습니다.
- 토마토: 15장, 4개체(T09, T10, T15, T16). 과숙은 T09 한 개체의 사진 3장뿐이며 모두 적숙으로 오분류됐습니다.
- 바나나: 30장, 26개 원본 그룹. 오류는 미숙·적숙 사이에서 나타났습니다.

## 새 검증 표본 수집 원칙

1. 현재 test 그룹은 계속 잠급니다. 새 표본은 `validation`으로 관리하고 기존 test 개체·원본을 복사하거나 변형해 넣지 않습니다.
2. 귤 적숙과 토마토 과숙은 기존 실패 그룹과 다른 과일 개체 및 촬영일에서 확보합니다. 바나나는 미숙·적숙 경계 오류를 확인할 독립 원본 그룹을 추가합니다.
3. 숙도 정답은 모델 결과를 보기 전에 사람이 확인해 기록합니다. 불확실한 라벨은 추정하지 말고 재검토 상태로 둡니다.
4. 같은 과일 개체의 여러 사진은 하나의 `group_id`에 묶습니다. 같은 그룹이 train, validation, test에 나뉘지 않게 합니다.
5. 가능하면 조명·배경·거리 등 촬영 조건을 함께 기록하고, 사진 재촬영을 평가할 때는 최초 사진과 재촬영 사진을 같은 그룹으로 연결합니다.
6. 표본 수의 임의 목표를 두지 않습니다. 수집을 시작하기 전에 팀이 필요한 신뢰구간 폭 또는 허용 오차를 정하고, 독립 개체 그룹 기준으로 그 기준을 충족하는지 판단합니다.

## 정책 변경 전 비교 기준

새 validation 예측을 고정한 뒤 현재 정책을 기준선으로 두고 후보 정책을 오프라인 비교합니다. 각 후보에서 과일·숙도별 정밀도와 재현율, 자동 분류 커버리지, 자동 분류 오분류율, 사람 확인으로 보낸 실제 오류 비율을 함께 봅니다. 재촬영 권고는 동일 개체의 실제 재촬영 전후 결과가 개선되는지 별도로 평가합니다.

정책 변경은 사전에 합의한 운영 목표를 만족하고, 독립 test에서 최종 확인된 경우에만 반영합니다. 현재 70% 기준을 단순히 높이는 방식은 이번 데이터에서 높은 확률의 오분류가 다수 있었으므로 근거가 되지 않습니다.

## 다음 실행 때 남길 정보

- 이미지별 결과 CSV: 이미지 경로, 과일·개체 그룹, 라벨 분할, 정답(검증 전용), 예측, 클래스별 확률, 에이전트 행동·이유
- 실행 manifest: 실행 시각, 코드 커밋, 모델 번들 SHA-256, 선택된 과일별 모델/가중치 식별자, Python·Ultralytics 버전, 장치
- 사람 확인이 끝난 후에는 예측값과 확정 정답을 별도 필드로 보존
