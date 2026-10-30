"""Summarize the frozen-network rejection experiment and the actual trade-offs."""
# 고정 모델의 거부 규칙 실험에서 지원 과일 오거부와 미지원 과일 오수락을 함께 비교합니다.
from pathlib import Path
import json
from src.joint import ROOT


def main():
    run=Path((ROOT/'outputs/rejection_experiment/latest_run.txt').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    initial=json.loads((run/'decision.json').read_text(encoding='utf-8'))
    choices=json.loads((run/'rejection_reduction_decision.json').read_text(encoding='utf-8'))['operating_points']
    test=json.loads((run/'rejection_reduction_test.json').read_text(encoding='utf-8'))
    demo=json.loads((run/'demo_diagnostic.json').read_text(encoding='utf-8'))
    base=initial['baseline']['selection']
    accepted=[r for r in demo['all_rules'] if r['accepted']]
    min_false=min((r['selection_unknown_false_acceptance_rate'] for r in accepted),default=None)
    lines=['# 特徴距離・分類点数・点差の複合判定実験'.replace('特徴距離・分類点数・点差の複合判定実験','특징 거리·분류 점수·점수 차이의 복합 판정 실험'),'',
        '## 결론','',
        '고정된 기존 영상 모델에서 397개 규칙을 비교했습니다. 전체 거부율을 줄이는 절충안은 찾았지만, 시연 귤 M10을 안전하게 수락하는 규칙은 찾지 못했습니다. 기본 모델·모름 정책은 변경하지 않았습니다.','',
        '## 고정한 항목과 후보','',
        '- 발표용으로 복원한 기존 crop 모델을 사용했습니다. 직전 재학습 후보가 아닙니다.',
        '- 네트워크 가중치, 특징 기준 벡터, 기존 과일/숙도별 거리 기준값, 18% 크롭 전처리는 고정했습니다.',
        '- CSV의 현재 분할(M06·M07·M08 학습, M09 적숙 선택용)을 그대로 적용했습니다.',
        '- 비교한 신호: 9클래스 최고 점수, 1·2위 점수 차이, 후보 과일의 숙도 3개 점수 합, 기존 특징 거리.',
        '- 후보 방식: 거리 단독, 분류 점수 단독, 점수 차이 단독, 거리와 점수 동시 조건, 강한 분류 점수로 거리 거부를 구제하는 규칙.',
        '- 보정 자료에서 정답 종류를 예측한 사진의 점수 분위수와 사전 고정한 수치로 후보를 구성했습니다.',
        '- 397개 후보는 Selection의 지원 과일 436장과 미지 과일 40장에서 비교했습니다. 검출 실패도 실패로 포함했습니다.',
        '- 시연 사진은 후보/운영점 선택에 사용하지 않았고 결정 파일을 저장한 뒤 진단했습니다. Test도 선택에 사용하지 않았습니다.',
        '- 분류 점수는 실제 정답 확률이나 보정된 신뢰도가 아닙니다.','',
        '## 거부 감소 목표의 Selection 비교','',
        '정답률과 9클래스 Macro F1이 기존보다 낮아지지 않는 후보 중, 미지 과일 오수락 허용 범위별로 지원 과일 거부를 최소화했습니다. 단순 F1 최대 규칙은 오히려 거부를 늘렸기 때문에 별도로 기록했습니다.','',
        '| 규칙 | 지원 과일 거부·검출 실패 | 미지 과일 오수락 | 종류·숙도 정답률 | 9클래스 Macro F1 |','|---|---:|---:|---:|---:|',
        f"| 기존 거리 단독 | {base['known_rejection_or_missing_rate']:.2%} | 0/40 | {base['joint_accuracy']:.2%} | {base['nine_class_macro_f1']:.4f} |"]
    for name,title in [('low_rejection_0','오수락 0% 제한'),('low_rejection_5','오수락 5% 이내 제한'),('low_rejection_10','오수락 10% 이내 제한')]:
        m=choices[name]['selection']
        lines.append(f"| {title} | {m['known_rejection_or_missing_rate']:.2%} | {round(m['unknown_false_acceptance_rate']*40)}/40 | {m['joint_accuracy']:.2%} | {m['nine_class_macro_f1']:.4f} |")
    f1point=initial['operating_points']['strict']['selection']
    lines += ['',
        '- 0% 및 5% 제한에서는 거부를 줄이면서 정답률·Macro F1도 유지하는 새 후보가 없어 기존 거리 규칙이 선택됐습니다.',
        '- 10% 제한의 절충안은 거리 기준을 1.5배로 넓히되 최고 9클래스 점수 0.5 이상, 후보 과일의 숙도 점수 합 0.95 이상을 동시에 요구합니다.',
        '- 이 절충안은 지원 과일의 거부·검출 실패를 60/436장→25/436장으로 줄였지만 미지 과일 4/40장을 지원 과일로 받아들였습니다. 기본 정책으로 적용하지 않았습니다.',
        f"- F1만 최대화한 0% 오수락 후보는 F1 {f1point['nine_class_macro_f1']:.4f}지만 지원 과일 거부가 {f1point['known_rejection_or_missing_rate']:.2%}로 늘어, 거부 감소 해결책으로 채택하지 않았습니다.",'',
        '## 시연 귤 M10 진단','',
        f"- 영상 90번 프레임의 후보: {demo['candidate_class']} (귤·적숙).",
        f"- 최고 9클래스 점수: {demo['joint_confidence']:.4f}, 1·2위 차이: {demo['joint_margin']:.4f}.",
        f"- 귤 숙도 3개 점수 합: {demo['fruit_scores']['mandarin']:.4f}.",
        f"- 특징 거리: {demo['embedding_score']:.4f}, 기존 귤 기준: 약 0.1318.",
        '- 선택한 0%·5%·10% 오수락 운영점 모두 M10을 모름으로 처리했습니다.',
        f"- 전체 실험 규칙 중 M10을 받아들이는 규칙은 {len(accepted)}개였습니다. 이 중 Selection 미지 과일 오수락이 가장 적은 경우도 {round(min_false*40)}/40장({min_false:.1%})이었습니다.",
        '- 이것은 실험한 규칙 범위에서의 결과이며 모든 가능한 판정법의 한계를 증명하는 수치는 아닙니다. 해당 사진을 통과시키는 것만 목표로 기준을 풀면 잘못된 수락이 크게 늘어남을 보여줍니다.','',
        '## 변경 후 Test의 회귀 비교','',
        '| 규칙 | 지원 과일 거부·검출 실패 | 미지 과일 오수락 | 종류·숙도 정답률 |','|---|---:|---:|---:|']
    for name,title in [('low_rejection_0','기존 규칙'),('low_rejection_10','10% 허용 실험 후보')]:
        m=test[name]
        lines.append(f"| {title} | {m['known_rejection_or_missing_rate']:.2%} | {round(m['unknown_false_acceptance_rate']*40)}/40 | {m['joint_accuracy']:.2%} |")
    lines += ['',
        '- 선택에서 정한 오수락 허용 범위가 새 자료에서도 유지된다는 보장은 없습니다. 위 Test는 이미 확인한 회귀 자료입니다.',
        '- 남은 귤 Test에는 과숙만 있으며 미숙·적숙 귤의 새 독립 검증을 수행한 것은 아닙니다.',
        '- 이전에 확인한 과숙 이미지의 동일 원본 의심 쌍은 남아 있으므로 Selection 독립성에도 한계가 있습니다.',
        '- 최종 영상에는 잘림·축소·압축과 부분 가림이 추가됩니다. 정지 사진 결과와 프레임 결과가 같다고 보장하지 않습니다.','',
        '## 결과물','',
        '- `strict`, `balanced`: F1 우선 실험용 모델 묶음. 기본 모델로 적용하지 않았습니다.',
        '- `low_rejection_0`, `low_rejection_5`, `low_rejection_10`: 거부 감소를 우선한 별도 실험용 모델 묶음.',
        '- `candidates.json`: 397개 규칙의 Selection 결과.',
        '- `demo_diagnostic.json`: 선택 완료 후 시연 귤 점검 결과.',
        '- `outputs/video_experiment/fruit_conveyor_gate_experiment.mp4`: 10% 허용 절충안의 영상 결과. 기본 발표 영상은 보존했습니다.','',
        '## 권장 후속 방향','',
        '거부 기준 조정만으로 M10 문제를 해결하는 것은 현재 실험에서 검증되지 않았습니다. 다음에는 종류 판정과 숙도 판정을 분리하거나, 품종·색·표면이 다른 귤 개체를 추가해 특징 표현 자체를 보완해야 합니다. 발표 화면에는 확정 결과와 확인 필요 후보를 구분해 표시하는 방법을 사용할 수 있지만, 모름 결과를 정답으로 강제 변경하지는 않습니다.','',
        '## 실행','',
        '```powershell',
        'python -m scripts.experiment_rejection_signals',
        'python -m scripts.finish_rejection_experiment',
        f'python -m src.video --model-dir outputs/rejection_experiment/{run.name}/low_rejection_10 --output outputs/video_experiment/fruit_conveyor_gate_experiment.mp4',
        '```']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(run/'report.md')
    print('Regression unknown false accepts:',round(test['low_rejection_10']['unknown_false_acceptance_rate']*40),'/40')


if __name__=='__main__':
    main()
