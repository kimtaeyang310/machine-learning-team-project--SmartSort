"""Diagnose the frozen two-stage policies after selection and write results."""
from pathlib import Path
import json
import cv2
from src.joint import ROOT
from src.joint_core import dump
from src.two_stage import TwoStageClassifier
from src.video import detect_regions,crop_region


def main():
    run=Path((ROOT/'outputs/two_stage_experiment/latest_run.txt').read_text(encoding='utf-8'))
    cap=cv2.VideoCapture(str(ROOT/'outputs/video_experiment/fruit_conveyor.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,90)
    ok,frame=cap.read();cap.release()
    if not ok:
        raise ValueError('Cannot read demonstration frame')
    box=max(detect_regions(frame),key=lambda b:b[0])
    results={}
    for point in ('strict','balanced'):
        classifier=TwoStageClassifier(run/point)
        results[point]=classifier.predict_frame(crop_region(frame,box))
    # Oracle diagnostic only: do not change the actual gate or displayed result.
    oracle_p,_=classifier.stages['mandarin'].raw(crop_region(frame,box))
    from src.two_stage import STAGES
    oracle={'assumption':'fruit is given as mandarin; bypass only for diagnosis, never production',
            'prediction':STAGES[int(oracle_p.argmax())],
            'stage_probabilities':dict(zip(STAGES,map(float,oracle_p)))}
    dump(run/'demo_diagnostic.json',{'frame':90,'box':list(box),'used_for_selection':False,'results':results,'oracle_mandarin_stage':oracle})
    comparison=json.loads((run/'comparison.json').read_text(encoding='utf-8'))
    decisions=json.loads((run/'decision.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    baseline=comparison['baseline_selection']
    rows=[('기존 통합 모델',baseline)]+[(name,comparison['two_stage'][name]['selection']) for name in ('strict','balanced')]
    lines=['# 종류·숙도 분리 모델 실험','',
        '## 구조','',
        '```text','과일 후보 영역 크롭','  → 종류 전용 모델: 바나나 / 귤 / 토마토','  → 종류 신호로 지원 과일 여부 판정',
        '     → 거부: 모름, 숙도 모델 실행 생략','     → 수락: 해당 과일의 숙도 전용 모델 실행','          → 미숙 / 적숙 / 과숙 + 필요 시 재촬영·확인 권고','```','',
        '- 종류 전용 3클래스 YOLO 1개와 과일별 숙도 3클래스 YOLO 3개를 각각 6 epochs 학습했습니다.',
        '- 각 모델은 기존 crop 모델의 특징 추출 가중치에서 시작하되 분류 출력층은 해당 3클래스로 교체되어 학습됩니다.',
        '- Train 사진, 18% 크롭, seed와 입력 크기는 기존 실험과 동일합니다. M06·M07·M08은 학습, M09는 적숙 기준 선택용입니다.',
        '- M10을 비롯한 시연 중복 개체는 학습·보정·선택에서 제외했습니다.',
        '- 모델 선택은 고정 6 epochs의 last.pt이며, API 로더를 위한 val 3장은 학습 사진 복사본으로 점수는 무시했습니다.',
        '- 종류 모델의 특징 거리, 종류 점수, 점수 차이를 Calibration/Selection에서 비교했습니다. 숙도 확률은 지원 과일 거부 기준에 넣지 않았습니다.',
        '- 종류가 확실하지만 숙도가 불확실하면 과일 종류를 유지한 채 재촬영·확인 권고로 표시합니다.','',
        '## 동일한 변경 후 Selection 비교','',
        '| 방식 | 9클래스 Macro F1 | 종류·숙도 정답률 | 지원 과일 거부·검출 실패 | 미지 과일 오수락 |',
        '|---|---:|---:|---:|---:|']
    for name,m in rows:
        lines.append(f"| {name} | {m['nine_class_macro_f1']:.4f} | {m['joint_accuracy']:.2%} | {m['known_rejection_or_missing_rate']:.2%} | {round(m['unknown_false_acceptance_rate']*40)}/40 |")
    lines += ['',
        '- strict는 Selection에서 미지 과일 오수락 0%, balanced는 5% 이내 후보 중 Macro F1을 최대화하고 거부를 줄이는 기준으로 선택했습니다.',
        '- 후보와 두 운영점은 시연 사진 및 Test 점수를 보기 전에 결정했습니다. 기본 발표용 모델은 변경하지 않았습니다.','',
        '## 단계별 진단','',
        '| 운영점 | 종류 정답률: 거부 전·검출 실패 포함 | 종류가 맞은 사진의 숙도 정답률 | 귤 Macro F1 |',
        '|---|---:|---:|---:|']
    for name in ('strict','balanced'):
        m=comparison['two_stage'][name]['selection']
        lines.append(f"| {name} | {m['kind_accuracy_before_gate_including_missing']:.2%} | {m['stage_accuracy_on_correct_kind']:.2%} | {m['per_fruit']['mandarin']['macro_f1']:.4f} |")
    lines += ['', '## 시연 M10 귤 결과','',
        '실제 이동 영상의 90번 프레임에서 과일 영역을 검출한 뒤 평가했습니다. 결과를 보고 임계값을 다시 조정하지 않았습니다.','',
        '| 운영점 | 종류 후보 | 종류 최고 점수 | 최종 종류 | 최종 숙도 | 안내 |',
        '|---|---|---:|---|---|---|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['candidate_fruit']} | {r['kind_confidence']:.4f} | {r['fruit']} | {r['ripeness']} | {r['agent_decision']['action']} |")
    lines += ['',
        f"검사 목적으로만 '귤임을 이미 안다'고 가정하고 숙도 모델을 따로 실행한 결과는 {oracle['prediction']}이며 점수는 {oracle['stage_probabilities']}입니다. 실제 영상은 종류 게이트를 우회하지 않으며, 이 진단을 보고 기준을 변경하지 않았습니다.",
        '종류 점수와 숙도 점수는 서로 다른 모델에서 나옵니다. 숙도 점수는 선택된 과일에 조건부인 값이며 이전 통합 9클래스 점수와 직접 비교할 수 없습니다. 점수는 보정된 실제 정답 확률이 아닙니다.','',
        '## 변경 후 Test의 회귀 비교','',
        '| 방식 | 종류·숙도 정답률 | 지원 과일 거부·검출 실패 | 미지 과일 오수락 |',
        '|---|---:|---:|---:|']
    for name,m in [('기존 통합 모델',comparison['baseline_test'])]+[(n,comparison['two_stage'][n]['test']) for n in ('strict','balanced')]:
        lines.append(f"| {name} | {m['joint_accuracy']:.2%} | {m['known_rejection_or_missing_rate']:.2%} | {round(m['unknown_false_acceptance_rate']*40)}/40 |")
    lines += ['',
        '## 한계와 실행','',
        '- 기존에 확인한 자료를 반복 사용한 회귀 실험입니다. 새 개체·새 영상에 대한 맹검 성능이 아닙니다.',
        '- 남은 귤 Test에는 과숙만 있어 미숙·적숙 귤의 독립 성능을 검증할 수 없습니다.',
        '- 과숙의 서로 다른 ID 사이에 동일 원본 의심 쌍이 남아 있어 기준 선택 독립성에도 한계가 있습니다.',
        '- 색상·배경 기반 후보 검출의 한계는 그대로입니다. 위치 검출 모델을 학습한 실험이 아닙니다.',
        '- 지원 과일은 종류 모델과 숙도 모델을 차례로 두 번 실행하므로 단일 모델보다 느릴 수 있습니다. 네 모델은 영상당 한 번씩 로딩합니다.',
        '- 기본 모델은 보존하고 strict/balanced 폴더에 실험용 모델 묶음을 저장했습니다.','',
        '```powershell',
        f'python -m src.video --model-dir outputs/two_stage_experiment/{run.name}/strict --output outputs/video_experiment/fruit_conveyor_two_stage_strict.mp4',
        f'python -m src.video --model-dir outputs/two_stage_experiment/{run.name}/balanced --output outputs/video_experiment/fruit_conveyor_two_stage_balanced.mp4',
        '```','',
        '학습·기준 선택 재현: `python -m scripts.train_two_stage`, `python -m scripts.select_two_stage`, `python -m scripts.report_two_stage`.']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(run/'report.md')
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
