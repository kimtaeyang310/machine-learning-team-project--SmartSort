# 외부 검증 데이터셋 재조사 기록

## 목적과 현재 상태

기존 58장 test와 분리해서 귤 `ripe`, 바나나의 숙도 전반, 토마토 `overripe` 취약 구간을 확인하기 위한 공개 데이터 후보를 다시 조사했습니다. 기존 기준선 CSV·보고서·모델 정책은 수정하지 않았습니다.

세 과일을 모두 포함하고 각 과일의 프로젝트 숙도 라벨까지 제공한다고 확인된 단일 공개 데이터셋은 찾지 못했습니다. 현재는 아래 과일별 자료를 조합하는 방안이 가장 구체적입니다. 실제 외부 예측 결과는 데이터 파일을 내려받아 확인하고 모델을 실행하기 전까지 만들어지지 않습니다.

## 과일별 후보와 프로젝트 라벨 대응

| 과일 | 후보 | 공개 정보 | 대응 및 판단 |
|---|---|---|---|
| 귤 | [Mandarinas_Madurez (Roboflow Universe)](https://universe.roboflow.com/cesars-workspace-1vy3h/mandarinas_madurez) | 634장, `Mandarina_madura`와 `Mandarina_verde` 두 클래스, CC BY 4.0으로 표시. | `Mandarina_madura`→`ripe`, `Mandarina_verde`→`unripe`. 귤 `ripe` 취약 구간을 직접 확인할 후보입니다. 2단계만 있으므로 과숙 평가는 할 수 없습니다. 공개 다운로드/내보내기 화면에서 실제 파일 접근과 주석 형식을 받은 뒤 확정합니다. |
| 바나나 | [Annotated Banana Ripeness Stage Dataset (Mendeley Data v2)](https://data.mendeley.com/datasets/56c75hvm7r/2) | 819장, `unripe`/`ripe`/`overripe`, YOLO 바운딩박스 주석, CC BY 4.0, 페이지에 Download All 제공. | 라벨은 프로젝트와 직접 대응합니다. `unripe`→`unripe`, `ripe`→`ripe`, `overripe`→`overripe`. 이미지 단위 분류를 위해서는 이미지별 상자 라벨을 읽어 과일 대상이 명확한 이미지인지 확인해야 합니다. 같은 바나나의 연속 사진/복제본은 독립 표본으로 세지 않습니다. |
| 토마토 | [Tomato dataset (Roboflow Universe)](https://universe.roboflow.com/tomato-6l30u/tomato-hat4j) | 토마토 전용 object detection 자료, 2,194장, CC BY 4.0으로 표시. 페이지상 8개 클래스: `ripe`, `overripe`, `damaged`, `irregular`, `over-ripe`, `rotten`, `semi-ripe`, `unripe`. | `ripe`→`ripe`, `unripe`→`unripe`, `semi-ripe`는 프로젝트에 동일 클래스가 없어 제외합니다. `overripe`와 `over-ripe`는 표기상 동일해 보이지만, export한 주석과 샘플을 대조한 뒤에만 하나의 프로젝트 `overripe`로 합칩니다. `rotten`, `damaged`, `irregular`는 과숙과 동의어가 아니므로 제외합니다. 단일 과일 전용 후보입니다. |

## 다운로드 전에 확인할 사항

- Roboflow 귤 자료의 공개 페이지는 클래스와 라이선스를 보여 주지만, 이 조사에서 이미지 아카이브를 실제로 내려받아 본 것은 아닙니다. 플랫폼에서 공개 export가 가능한지와 파일 구성을 확인한 뒤 진행합니다.
- 바나나 Mendeley 자료는 다운로드 버튼이 공개되어 있지만, 게시된 클래스 정의와 YOLO 주석을 내려받은 파일에서 대조해야 합니다. 게시 설명에 표본 분할 수와 비율 표기가 일치하지 않는 부분이 있으므로 원본 파일의 실제 개수를 다시 셉니다.
- 토마토 후보는 과일 전용이지만 object detection 형식입니다. export한 뒤 이미지별 라벨과 바운딩박스를 확인하고, 단일 토마토가 분명하게 들어간 이미지만 분류 평가에 사용합니다.
- Roboflow 페이지에 같은 뜻처럼 보이는 `overripe`와 `over-ripe`가 별도 클래스로 보입니다. 원본 주석을 확인하기 전에는 합치지 않습니다. `rotten`은 부패, `overripe`는 과숙이므로 서로 바꿔 쓰지 않습니다.
- 모든 후보에 대해 이미지 손상, 반복 이미지, 변형/증강본, 개체 식별 가능 여부를 다운로드 후 확인합니다. 서로 같은 원본에서 나온 이미지가 있으면 묶어서 기록합니다.

## 별도 보존 방식

외부 표본은 `data/external_validation/` 아래의 과일별 폴더에 둡니다. 예측 CSV, 실행 manifest, 라벨 CSV, 보고서도 과일별 별도 파일로 저장하고 기존 `baseline_predictions.csv`와 병합하지 않습니다. 라벨 CSV의 `ground_truth_status`는 출처 annotation을 확인한 표본에만 `verified`로 설정합니다. 개체 식별을 할 수 없는 표본의 `group_id`는 비워 두거나 `unknown`으로 기록합니다.

현재 예측 스크립트는 파일명에서 `banana`, `mandarin`, `tomato` 토큰을 읽어 과일별 모델을 고릅니다. 실행용 파일명은 해당 토큰을 포함하도록 만들되 원래 이름은 별도 `source_image_name`으로 보존해야 합니다. 검증 기준에 맞게 후보 이미지 전체를 무작정 이름 변경·혼합하지 말고, 모델이 처리할 단일 과일 이미지를 우선 선별합니다.

## 이 조사에서 외부 성능을 아직 산출하지 않은 이유

데이터 페이지의 요약과 라벨 정의만 확인한 단계입니다. 다운로드 아카이브 내부의 실제 클래스 수, 주석의 정확성, 중복, 모델이 요구하는 입력 형태와의 호환성을 확인하기 전에는 이미지 수나 정확도·정밀도·재현율을 결과처럼 기록하지 않습니다. 확보 후에는 과일별 실행 결과를 각각 평가하고, 토마토에서 `overripe`와 `over-ripe`를 통합한다면 통합 전 원본 클래스 수와 통합 근거를 함께 남깁니다.

## 출처

- [Mandarinas_Madurez, Roboflow Universe](https://universe.roboflow.com/cesars-workspace-1vy3h/mandarinas_madurez): 634 images, ripe/green classes, CC BY 4.0 표시.
- [Annotated Banana Ripeness Stage Dataset, Mendeley Data v2](https://data.mendeley.com/datasets/56c75hvm7r/2): 819 images, three ripeness classes, CC BY 4.0.
- [Tomato dataset, Roboflow Universe](https://universe.roboflow.com/tomato-6l30u/tomato-hat4j): tomato-only, 2,194 images, eight visible classes, CC BY 4.0 표시.
