"""Write the crop adaptation report from frozen experiment artifacts."""
from pathlib import Path
import json
from collections import Counter
from src.joint import ROOT


def main():
    run = Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    tests = json.loads((run/'test_metrics.json').read_text(encoding='utf-8'))
    audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    config = json.loads((run/'finetune_config.json').read_text(encoding='utf-8'))
    old,new = tests['baseline'],tests['crop_calibrated']
    lines = ['# 과일 영역 크롭 모델 보정 결과','',
        '## 적용한 변경','',
        f"- 기존 모델에서 크롭 학습 사진 {sum(config['training_counts'].values())}장으로 6 epochs 추가 학습했습니다.",
        '- 입력 사진을 최대 320픽셀로 맞춘 뒤 기존 영상과 같은 HSV 영역 추출·18% 여백 크롭을 적용했습니다.',
        '- Train의 크롭 특징만으로 모름 판정 기준 특징을 구성했습니다.',
        '- 먼저 Calibration의 과일별 95% 종류 정답·수락 목표를 시도했지만 Selection의 미지 과일 6/40장을 오수락했습니다.',
        '- 최종 기준은 Calibration의 5~95% 분위수 후보 중 Selection에서 미지 과일 오수락을 우선 최소화하는 조합으로 선택했습니다. 알려진 과일 95% 수락 목표는 최종 기준에서 보장하지 않습니다.',
        '- 포도를 바나나 과숙으로 오인하는 문제를 바나나 전체에 적용하지 않도록 바나나는 미숙·적숙·과숙별 기준을 분리했습니다. 귤과 토마토는 과일별 기준을 사용합니다.',
        '- Selection에서 미지 과일 오수락 최소 → 9클래스 Macro F1 최대 → 거부 최소 순으로 판정 방식을 선택했습니다.',
        f"- 선택 방식: {selection['chosen_method']}; 영상 모델 적용: {selection['promote_video_policy']}",
        '- 원본 사진용 models/joint_nine 모델 묶음은 변경하지 않았습니다. 영상 전용 models/joint_nine_crop을 별도로 생성했습니다.','',
        '## 같은 크롭 입력에서의 테스트 비교','',
        '각 비교는 동일한 테스트 사진에 동일한 영역 추출을 적용했습니다. 기존 모델의 원본 사진 평가 점수와 직접 비교하는 표가 아닙니다.','',
        '| 항목 | 기존 모델·기준 | 크롭 추가 학습·보정 |','|---|---:|---:|',
        f"| 9클래스 Macro F1 | {old['nine_class_macro_f1']:.4f} | {new['nine_class_macro_f1']:.4f} |",
        f"| 종류·숙도 동시 정답률 | {old['joint_accuracy']:.2%} | {new['joint_accuracy']:.2%} |",
        f"| 지원 과일 거부 또는 검출 실패율 | {old['known_rejection_or_missing_rate']:.2%} | {new['known_rejection_or_missing_rate']:.2%} |",
        f"| 미지 과일 오수락률 | {old['unknown_false_acceptance_rate']:.2%} | {new['unknown_false_acceptance_rate']:.2%} |",
        f"| 검출 실패 장수 | {old['missing_detection_count']} | {new['missing_detection_count']} |",'',
        f"테스트 자료: 지원 과일 {new['known_count']}장, 미지 과일 {new['unknown_count']}장.",'',
        '| 과일 | 테스트 장수 | 기존 Macro F1 | 보정 Macro F1 |','|---|---:|---:|---:|']
    for fruit,name in [('banana','바나나'),('mandarin','귤'),('tomato','토마토')]:
        a,b=old['per_fruit'][fruit],new['per_fruit'][fruit]
        lines.append(f"| {name} | {b['count']} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} |")
    lines += ['', '## 검증 범위와 남은 문제','',
        '- 기존 manifest의 개체별 train/calibration/selection/test 분할을 유지했습니다. 각 후보의 테스트 점수는 기준 선택과 적용 여부를 결정한 뒤 계산했습니다. 다만 앞선 후보들의 테스트 결과도 이미 확인했으므로 최종 결과는 새 맹검 평가가 아닌 탐색적 회귀 평가입니다.',
        '- 시연 사진과 매우 유사한 원본을 픽셀 유사도로 찾고 해당 개체 전체를 이번 추가 학습·보정·평가에서 제외했습니다. 제외 목록은 selection.json에 있습니다.',
        '- 초기 8/255 유사도 기준이 회색 배경의 서로 다른 과일까지 묶는 것을 확인해 2/255로 수정했습니다. 픽셀 일치 검사 변경이며 정답 성능을 기준으로 정하지 않았습니다.',
        '- 기존 모델 자체가 시연 사진 원본으로 이미 학습되었을 가능성은 없애지 못합니다. 시연 개선을 독립적인 일반화 성능으로 주장하면 안 됩니다.',
        '- 기존에 확인한 테스트 자료를 재사용한 회귀 평가입니다. 새 촬영 영상에 대한 독립 검증과 수동 박스 정답에 대한 검출 정확도 평가는 수행하지 않았습니다.',
        '- 검출 실패는 알려진 과일에 대한 실패로 포함했습니다. 가장 큰 영역 한 개를 선택해 사진별 라벨과 비교하므로 개별 다중 과일 평가가 아닙니다.',
        '- 색상·배경 기반 검출의 한계는 그대로입니다. 저채도 과숙 과일, 복잡한 배경, 겹친 과일, 화면 경계에 일부만 나타난 과일은 오인식·거부될 수 있습니다.',
        '- 프레임별 결과는 독립 표본이 아니며 동영상 461프레임을 테스트 사진 461장처럼 계산하지 않습니다.',
        '- 모름 임계값은 종류·숙도 정확도의 확률이나 식품 안전성 보장 수치가 아닙니다.','',
        '## 재현','', '프로젝트 루트의 teamTC 환경에서 다음 순서로 실행합니다.','',
        '```powershell','python -m scripts.calibrate_crops','python -m scripts.finetune_crops','python -m scripts.select_crop_policy','python -m scripts.select_strict_crop_policy','python -m scripts.select_class_crop_policy','python -m scripts.report_crop_calibration','python -m src.video --output outputs/video_experiment/fruit_conveyor_calibrated.mp4','```','',
        '새 실험이 기존 영상 모델보다 낫다고 판단되면 installed.json에 생성 경로가 기록됩니다. 이미 models/joint_nine_crop이 있으면 새 후보는 실험 폴더의 bundle에 저장되어 자동 덮어쓰기를 피합니다. 이 경우 --model-dir로 해당 후보를 지정합니다.','',
        '원본 모델로 영상 분석하려면:','',
        '```powershell','python -m src.video --model-dir models/joint_nine --output outputs/video_experiment/fruit_conveyor_baseline.mp4','```']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(run/'report.md')


if __name__ == '__main__':
    main()
