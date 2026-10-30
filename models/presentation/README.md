# 선별 테스트 과일 시연용 모델

- 구조: YOLO26n-cls 통합 9클래스, CPU 추론.
- 모델은 적숙 귤 자료를 추가한 다양성 실험의 고정 후보를 복사했습니다. 재학습하거나 임계값을 수정하지 않았습니다.
- `model.pt`, `policy.joblib`, `split_manifest.json`의 SHA-256을 `bundle.json`에서 확인합니다.
- 예전 실험 outputs 폴더는 추론에 필요하지 않습니다. bundle의 source는 출처 기록이며 실행 입력이 아닙니다.
- `config/presentation.json`, `data/presentation/`, src 코드와 함께 공유합니다.
- 사진용 기본 모델 `models/joint_nine`과 구분합니다. 이 폴더를 기본 모델 위에 덮어쓰지 않습니다.

현재 시연 사진은 성공한 테스트 예시를 선별한 자료이므로, 시연 결과가 새로운 정확도 평가를 뜻하지 않습니다.
