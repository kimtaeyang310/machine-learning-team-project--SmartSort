# Python 함수 구성 점검

2026-10-30 기준 Git에서 관리하는 Python 파일 66개를 점검했습니다.
기존 함수·클래스 구조는 유지하고, 역할이 섞인 부분과 중복 처리를 중심으로 14개 파일을 리팩터링했습니다.
앞서 추가한 한국어 주석도 유지했습니다.

## 변경 내용

| 대상 | 분리한 역할 |
|---|---|
| `src/predict.py` | CSV 저장, 실행 명세 구성, 명령행 실행 |
| `src/assistant/tools.py` | 실시간·저장 결과의 공통 집계, 저장된 처리 시간 집계 |
| `src/assistant/ollama_client.py` | 과일 별칭·지원 여부·전체 분석 범위 검증 |
| `src/video_tracking.py` | 한 프레임의 추적·투표, 개체별 통계, 영상 입출력 |
| `manual_mask.py` | 클릭 좌표 변환과 이진 마스크 생성 |
| `scripts/register_mandarin_diversity.py` | 원본 가족 그룹화, 재현 가능한 분할·등록 행 구성 |
| `scripts/experiment_1030_mandarin.py` | 특징 추출, 기하 정합 기반 원본 그룹화 |
| `scripts/train_two_stage.py` | 종류·숙도 모델별 학습 자료 구성 |
| `src/joint.py`, `scripts/convert_fruit_classifier.py`, `scripts/experiment_t01_train.py`, `scripts/select_crop_policy.py` | 명령행 인자 해석을 `main(argv=None)` 또는 `cli(argv=None)`로 분리 |
| `tmp/pdfs/build_proposal.py` | PDF 생성 진입 함수와 12개 페이지 렌더링 함수, 작업 내부의 캔버스 상태 |
| `tmp/pdfs/retarget.py` | 문구 교체를 명시적 실행 함수로 분리 |

대표적으로 `save_prediction_artifacts`는 90줄에서 17줄,
`summarize_recent`는 101줄에서 31줄,
`process_tracked_video`는 128줄에서 78줄로 줄었습니다.
함수 길이 자체보다 입출력·계산·상태 관리의 책임을 나누는 데 초점을 맞췄습니다.

## 유지한 부분

- 이미 역할이 분명한 분류·거부 정책·라벨 관리 함수와 테스트 구조.
- UI 구성, 대화 상태 전이, 실험 진행처럼 순서 자체가 중요한 실행 함수. 길다는 이유만으로 일괄 분할하지 않았습니다.
- 모델 설정, 임계값, 난수 seed, 데이터 분할 비율, 저장 열 이름과 결과 형식.
- 과거 비교 실험의 환경 초기화 순서. 해당 실험과 변환 도구에는 현재 `src`에 없는 이전 모듈 참조가 있으므로 이번 작업에서 실행 호환성까지 복구한 것은 아닙니다.

## 검증

- 기존 테스트 38개와 새 회귀 테스트 10개, 총 48개 통과.
- 수정 전·후 실시간/CSV 집계 결과 비교.
- 비서 요청 20가지 조합에서 응답, 대화 기록, 확인 대기 상태, 작업 호출 비교.
- CSV 바이트 및 생성 시각을 제외한 실행 명세 비교.
- PDF 페이지 함수를 펼친 구문 트리가 기존 그리기 명령과 동일한지 확인.
- 기존 Python 파일 66개의 문법 검사.

학습·배포·데이터 등록 및 실제 PDF 생성은 실행하지 않았습니다.
영상 검증은 합성 이미지와 모의 분류기를 사용하며 실제 모델 성능을 새로 측정한 것은 아닙니다.

테스트 명령:

```powershell
& C:/Users/user1/anaconda3/envs/teamTC/python.exe -m unittest discover -s tests -v
```
