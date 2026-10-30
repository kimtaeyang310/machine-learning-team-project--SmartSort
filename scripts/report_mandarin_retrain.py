"""Create a concise report for the M06-M09 retraining run."""
# M06~M09 관련 재학습 실행 결과를 읽어 비교 보고서로 요약합니다.
from pathlib import Path
import json
from collections import Counter
from src.joint import ROOT


def main():
    run=Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    result=json.loads((run/'result.json').read_text(encoding='utf-8'))
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    tests=json.loads((run/'test_metrics.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    a,b=selection['baseline_same_selection'],selection['retrained']
    t1,t2=tests['previous_on_new_test'],tests['retrained_on_new_test']
    lines=['# M06~M09 분할 변경 및 영상 모델 재학습','',
        '## 변경 사항','',
        '| 개체 | 숙도 라벨 | 이전 | 변경 후 | 사진 수 |','|---|---|---|---|---:|',
        '| M06 | 적숙 유지 | test | train | 10 |',
        '| M07 | 미숙 유지 | test | train | 10 |',
        '| M08 | 미숙 유지 | test | train | 10 |',
        '| M09 | 적숙 유지 | test | selection | 10 |','',
        '- 네 개체 모두 CSV split=train으로 관리하고, 선택용 M09는 phase=selection으로 명시했습니다.',
        '- 선택용 M09는 신경망 학습과 기준 특징 구성에 사용하지 않았습니다. 다음 학습에서도 phase 지정이 유지됩니다.',
        '- M10 시연 원본은 기존처럼 추가 학습·보정·선택에서 제외했습니다.',
        '- CSV phase 필드를 선택 항목으로 추가했습니다. 기존 빈 값의 동작은 유지하며 split/phase 모순과 같은 개체의 분할 중복을 검사합니다.','',
        '## 학습','',
        f"- 이전 영상용 크롭 모델에서 총 {sum(config['training_counts'].values())}장으로 6 epochs 추가 학습했습니다.",
        '- 동일한 18% 크롭 전처리를 유지했습니다.',
        '- Calibration에서 임계값 후보를 만들고 Selection에서 오수락 최소, 9클래스 Macro F1 최대, 거부 최소 순으로 선택했습니다. 바나나는 숙도별, 귤·토마토는 과일별 기준을 사용합니다.',
        '- 임계값 탐색은 사전에 정한 5~95% 분위수와 0 수락 기준 후보에 대해 최대 3회 좌표 탐색했습니다. 테스트 점수는 선택에 사용하지 않았습니다.',
        f"- 새 영상 기본 모델 적용: {result['deployed']}",
        '- 선택 자료에서는 새 후보가 개선됐지만 시연 영상에서 M10 귤이 여전히 거부되고 빨간 토마토도 거부되는 퇴행을 확인했습니다. 발표용 기본 모델은 이전 버전으로 복원하고 새 후보는 이 실행 폴더의 bundle에 보존했습니다. CSV 분할 변경은 유지합니다.',
        '- 이전 영상 모델과 CSV는 이 실행 폴더의 previous_video_bundle 및 fruit_labels.before.csv에 보존했습니다.',
        '- 사진용 models/joint_nine 모델은 그대로 유지했습니다. 다음 사진 모델 학습은 변경된 CSV를 반영합니다.','',
        '## 동일한 변경 후 Selection에서 비교','',
        '| 항목 | 이전 영상 모델 | 재학습 모델 |','|---|---:|---:|',
        f"| 9클래스 Macro F1 | {a['nine_class_macro_f1']:.4f} | {b['nine_class_macro_f1']:.4f} |",
        f"| 종류·숙도 동시 정답률 | {a['joint_accuracy']:.2%} | {b['joint_accuracy']:.2%} |",
        f"| 지원 과일 거부·검출 실패율 | {a['known_rejection_or_missing_rate']:.2%} | {b['known_rejection_or_missing_rate']:.2%} |",
        f"| 미지 과일 오수락률 | {a['unknown_false_acceptance_rate']:.2%} | {b['unknown_false_acceptance_rate']:.2%} |",
        f"| 귤 숙도 Macro F1 | {a['per_fruit']['mandarin']['macro_f1']:.4f} | {b['per_fruit']['mandarin']['macro_f1']:.4f} |",'',
        '선택용에는 적숙 귤 M09의 사진 10장이 새로 포함되었습니다. 과거의 적숙 0장 Selection 점수와 직접 비교하지 않고 같은 새 분할에서 이전/새 모델을 비교했습니다.','',
        '## 남은 Test에서 회귀 비교','',
        '| 항목 | 이전 영상 모델 | 재학습 모델 |','|---|---:|---:|',
        f"| 고정 9클래스 Macro F1 | {t1['nine_class_macro_f1']:.4f} | {t2['nine_class_macro_f1']:.4f} |",
        f"| 종류·숙도 동시 정답률 | {t1['joint_accuracy']:.2%} | {t2['joint_accuracy']:.2%} |",
        f"| 지원 과일 거부·검출 실패율 | {t1['known_rejection_or_missing_rate']:.2%} | {t2['known_rejection_or_missing_rate']:.2%} |",
        f"| 미지 과일 오수락률 | {t1['unknown_false_acceptance_rate']:.2%} | {t2['unknown_false_acceptance_rate']:.2%} |",'',
        '- M06~M09를 이동했고 M10은 시연 중복으로 제외하므로, 이번 귤 Test에는 과숙 20장만 남습니다. 미숙·적숙 귤의 독립 테스트 성능은 확인할 수 없습니다.',
        '- 고정 9클래스 Macro F1은 없는 두 귤 클래스를 0으로 포함합니다. 같은 변경 후 Test에서의 이전/새 모델 비교는 가능하지만 이전 분할의 0.9197과 직접 비교할 수 없습니다.',
        f"- 귤 Selection F1은 개선됐지만 토마토 Selection F1은 {a['per_fruit']['tomato']['macro_f1']:.4f}→{b['per_fruit']['tomato']['macro_f1']:.4f}, 토마토 Test F1은 {t1['per_fruit']['tomato']['macro_f1']:.4f}→{t2['per_fruit']['tomato']['macro_f1']:.4f}로 하락했습니다. 전체 Test 정답률은 개선됐지만 Macro F1은 하락한 절충 결과입니다.",
        '- 이동한 개체들은 과거 테스트 결과를 이미 확인한 자료입니다. 새 데이터의 맹검 검증이 아닌 변경된 분할에 대한 회귀 평가입니다.',
        '- 적숙/미숙의 새로운 개체로 별도 최종 테스트를 확보해야 합니다.',
        '- 이전에 발견한 과숙 자료의 동일 원본 의심 쌍은 이번 요청 범위에서 수정하지 않았습니다. 기준 선택 자료 독립성의 한계가 남아 있습니다.','',
        '## 실행','',
        '메인 메뉴 6번 또는 `python -m src.video --display`는 복원한 기존 영상 모델을 실행합니다.',
        '재학습 후보의 결과 영상은 `outputs/video_experiment/fruit_conveyor_retrained.mp4`에 저장했습니다. 기존 최종 영상은 덮어쓰지 않았습니다.',
        f'재학습 후보를 직접 실행하려면 `python -m src.video --model-dir outputs/crop_calibration/{run.name}/bundle --output outputs/video_experiment/fruit_conveyor_retrained.mp4 --display`를 사용합니다.',
        '재학습 작업은 `python -m scripts.retrain_mandarin_split`로 재현할 수 있습니다. 추가 실행은 새로운 실행 폴더와 모델 백업을 생성합니다.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(run/'report.md')


if __name__=='__main__':
    main()
