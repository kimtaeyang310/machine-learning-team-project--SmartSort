# SmartSort 에이전트 기준선 안내

## 에이전트가 하는 일

SmartSort의 분류 모델이 과일의 숙도를 예측한 뒤, 에이전트 정책이 그 결과를 받아 다음 행동을 제안합니다. 현재 에이전트는 LLM이 아니라 `src/agent/decision.py`에 정의된 명시적 규칙으로 동작합니다.

| 우선순위 | 조건 | 제안 행동 |
|---:|---|---|
| 1 | 예측 결과가 취약 클래스인 토마토 과숙 | `human_review` — 사람 확인 요청 |
| 2 | 예측 확률을 얻지 못함 | `human_review` — 자동 분류 보류 |
| 3 | 예측된 숙도의 확률이 70% 미만 | `retake` — 재촬영 권고 |
| 4 | 위 조건에 해당하지 않음 | `auto_classify` — 자동 분류 제안 |

취약 클래스 규칙이 확률 규칙보다 먼저 적용됩니다. 따라서 토마토 과숙으로 예측되면 확률이 낮아도 사람 확인을 요청합니다. 반면 실제 과숙을 모델이 다른 숙도로 예측한 경우에는 이 예외가 발동하지 않을 수 있습니다.

## 한 장의 예측 처리 흐름

```text
이미지와 과일 종류 확인
        ↓
과일별 기존 분류 모델이 숙도와 확률을 계산
        ↓
decide_next_action()이 정책을 적용
        ↓
예측 결과에 agent_decision을 추가하고 화면에 출력
```

`agent_decision`은 다음 정보를 담습니다.

- `action`: `auto_classify`, `retake`, `human_review` 중 하나
- `reason`: 해당 행동을 고른 이유
- `suggested_label`, `confidence`: 모델 예측과 그 예측의 확률
- `requires_confirmation`: 사람의 확인이 필요한지
- `final_label`, `finalized_by`: 최종 확정 전에는 비어 있는 값

현재 폴더 예측은 행동과 이유를 출력할 뿐, 실제 재촬영을 수행하거나 사람의 승인을 받거나 재고를 변경하지 않습니다. `finalize_decision()`은 사용자 확정값을 결과 객체에 반영하는 함수이며, 현재 예측 UI나 파일·DB 저장 흐름에는 연결되어 있지 않습니다.

## 기준선의 의미와 한계

`baseline_predictions.csv`는 제공된 58장 실행 로그의 예측·행동을 정답 라벨과 대조해 고정한 스냅샷입니다. 결과는 39/58 정답(67.2%), 자동 분류 54장 중 15장 오분류(27.8%)입니다. 이 수치는 현재 모델과 정책을 평가한 값이지, 에이전트가 모델 정확도를 개선했다는 뜻은 아닙니다.

이번 표본은 모두 test 분할입니다. 기준선을 유지하기 위해 임계값을 맞추거나 예외 규칙을 고르는 데 재사용하지 않습니다. 70%는 검증으로 최적화된 운영 기준이 아니라 현재 정책의 비교 기준입니다.

모델의 클래스 확률은 그 자체로 정답 가능성을 보증하지 않습니다. 이번 기준선에서도 고확률 오분류가 나왔습니다. 또한 `retake`는 확률이 낮다는 이유로 제안하는 휴리스틱이며, 사진 품질을 직접 판정하거나 재촬영 후 개선을 보장하지 않습니다.

## 현재 코드 위치

- 정책 규칙과 사용자 확정 결과 형식: `src/agent/decision.py`
- 모델 예측 뒤 정책 호출 및 선택적 결과 CSV 저장: `src/predict.py`
- 고정 기준선과 지표 계산: `docs/agent_evaluation/baseline_predictions.csv`, `scripts/evaluate_agent_baseline.py`
- 기준선 원본 로그와 출처 정보: `docs/agent_evaluation/source_run_output.txt`, `docs/agent_evaluation/baseline_manifest.json`

다음 실행 결과를 보존하려면 프로젝트 루트에서 실행합니다.

```powershell
python -m src.predict data/predict_list --results-csv docs/agent_evaluation/next_run.csv
```

사람이 모델 결과를 보기 전에 확인한 정답 라벨을 추가한 후, 다음 명령으로 보고서를 생성할 수 있습니다.

```powershell
python scripts/evaluate_agent_baseline.py docs/agent_evaluation/next_run.csv --labels-csv docs/agent_evaluation/validation_labels.csv --output docs/agent_evaluation/next_run_report.md
```

## 평가 지표 읽는 법

- **정밀도:** 해당 숙도로 예측한 사진 중 실제로 그 숙도인 비율
- **재현율:** 실제 해당 숙도 사진 중 모델이 찾아낸 비율
- **자동 분류 커버리지:** 전체 입력 중 `auto_classify` 제안을 받은 비율
- **자동 분류 오분류율:** 자동 분류 제안 중 정답 라벨과 다른 비율
- **비자동 라우팅 오류:** 재촬영 또는 사람 확인으로 넘긴 예측 중 모델 예측이 틀린 건수. 이는 사람이 실제로 오류를 수정했다는 뜻은 아닙니다.

여러 장의 사진이 같은 과일 개체에서 나왔으면 독립 표본 하나로 보지 않습니다. 검증 표본은 서로 다른 과일 개체와 촬영일을 포함하고, 같은 개체의 사진이 train·validation·test 사이에 나뉘지 않도록 `group_id`로 관리합니다.
