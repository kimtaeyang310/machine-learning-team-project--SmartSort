"""Diagnose held-out M10 after policy selection and report diversity effects."""
# 정책 선택 후 보류 그룹 M10과 귤 자료 다양성 추가 효과를 분석해 보고서로 남깁니다.
from pathlib import Path
import copy
import json
import cv2
import numpy as np
from src.joint import ROOT,JointClassifier
from src.joint_core import dump,CLASSES
from src.video import detect_regions,crop_region


def main():
    run=Path((ROOT/'outputs/mandarin_diversity/latest_run.txt').read_text(encoding='utf-8'))
    comparison=json.loads((run/'comparison.json').read_text(encoding='utf-8'))
    registration=json.loads((run/'registration.json').read_text(encoding='utf-8'))
    config=json.loads((run/'training_config.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    cap=cv2.VideoCapture(str(ROOT/'outputs/video_experiment/fruit_conveyor.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,90);ok,frame=cap.read();cap.release()
    if not ok:raise ValueError('Cannot read demonstration frame')
    box=max(detect_regions(frame),key=lambda b:b[0])
    demo={}
    for arm,directory in [('baseline',run/'frozen_baseline'),('control',run/'control'),('diversity',run/'diversity')]:
        model=JointClassifier(directory)
        demo[arm]=model.predict_frame(crop_region(frame,box))
    dump(run/'demo_diagnostic.json',{'frame':90,'box':list(box),'used_for_selection':False,'predictions':demo})
    # Raw classifier accuracy is useful for separating representation improvement
    # from rejection improvements; missing proposals remain failures.
    raw={}
    for arm in comparison:
        data=np.load(run/(arm+'_features.npz'));p=data['probabilities']
        raw[arm]={}
        for cohort in ('old','new'):
            ids=[i for i,r in enumerate(audit) if r['phase']=='test' and not r['excluded_demo_group'] and r['fruit'] in ('banana','mandarin','tomato') and bool(r.get('new_1029'))==(cohort=='new')]
            raw[arm][cohort]=float(np.mean([np.isfinite(p[i]).all() and CLASSES[int(p[i].argmax())]==audit[i]['fruit']+'_'+audit[i]['ripeness'] for i in ids]))
    dump(run/'raw_classification_test.json',raw)
    names={'baseline':'기존 기본 모델','control':'기존 자료 재학습 + 새 보정 자료','diversity':'새 학습 자료 + 새 보정 자료'}
    lines=['# 적숙 귤 자료 추가·다양성 확보 실험','',
        '## 등록과 분할','',
        f"- data/raw/1029mandarin의 {registration['images']}장 전체를 사용자 정의에 따라 mandarin/ripe로 등록했습니다.",
        f"- 파일명 기준 원본 스크린샷 계열은 {registration['source_filename_families']}개입니다. 회전·이동·뒤집기·노이즈 변형을 같은 계열로 묶었습니다. 사진 1,373장을 서로 다른 개체 1,373개로 세지 않았습니다.",
        '- 서로 다른 원본 이름 간에도 매우 유사한 사진은 합치도록 보수적인 dHash/RGB 검사도 했지만 설정 기준에 해당하는 추가 쌍은 없었습니다. 원본 웹 출처/실제 개체가 독립이라는 뜻은 아닙니다.',
        '- 새 CSV 날짜 2026-10-29는 등록 날짜입니다. 파일명 속 2018년 스크린샷 시각을 실제 과일 촬영 날짜로 확인한 것은 아닙니다.',
        '- 사용자 정의 라벨을 적용한 실험이며 외관으로 숙도를 재판정하거나 실제 성숙 상태를 검증하지 않았습니다.','',
        '| 용도 | 원본 파일 그룹 | 사진 수 |','|---|---:|---:|']
    for phase in ('train','calibration','selection','test'):
        lines.append(f"| {phase} | {registration['new_source_clusters_by_phase'][phase]} | {registration['new_images_by_phase'][phase]} |")
    lines += ['',
        '- 모든 변형본을 원본 그룹과 같은 분할에 두었습니다. 원본 그룹의 분할 중복을 검사했습니다.',
        '- 새 이미지 전부 읽기와 과일 후보 크롭에 성공했습니다. 기존 파일과 SHA-256이 일치하는 신규 파일은 없었습니다. 다른 인코딩/각도/스크린샷의 동일 원본까지 완전히 배제한 것은 아닙니다.',
        '- 새 자료 라벨은 data/metadata/fruit_labels.csv에, 파일 목록·해시는 data/metadata/mandarin_1029_manifest.json에 저장했습니다. 이후 메인 메뉴 학습에서도 등록 파일이 반영됩니다.',
        '- 원본 이미지 파일은 이동·이름 변경·덮어쓰기하지 않았습니다. 기존 CSV 사본은 이 실행 폴더의 fruit_labels.before.csv에 있습니다.','',
        '## 비교 설계','',
        '- 모델 구조는 기존 통합 9클래스 crop 모델로 고정했습니다. 직전의 종류/숙도 분리 모델로 구조까지 함께 바꾸지 않았습니다.',
        '- A: 기존 기본 모델과 정책을 그대로 유지한 기준선.',
        '- B: 동일한 초기 가중치에서 기존 학습 자료 1,276장만 6 epochs 추가 학습한 대조군. 새 보정·선택 자료는 C와 동일하게 사용했습니다.',
        '- C: 기존 1,276장에 새 학습 822장을 더한 2,098장으로 같은 6 epochs 추가 학습. 기존/새 보정 자료로 특징 기준과 모름 기준을 재계산했습니다.',
        f"- 기존에 저장된 B 대조 모델 재사용: {config['control_reused']}. 초기 가중치 해시, 기존 표본 순서·라벨·개체·분할·해시와 epoch 수가 같은지 확인했습니다.",
        '- 기준 벡터는 각 모델의 학습 자료에서만 구성했습니다. B는 기존 학습 자료, C는 기존+새 학습 자료를 사용합니다.',
        '- B/C의 보정·선택 데이터, 거리 기반 판정 방식과 탐색 규칙은 같습니다. Calibration 분위수 후보를 만들고 Selection에서 미지 과일 오수락 최소 → Macro F1 최대 → 거부 최소 순으로 선택했습니다.',
        '- 후보 정책을 저장한 뒤 기존/새 Test와 M10을 확인했습니다. M10 및 기존 시연 중복 개체는 추가 학습·보정·선택에서 제외했습니다.',
        '- epochs는 같지만 C의 자료가 많아 최적화 업데이트 횟수도 늘었습니다. 학습 스텝까지 고정한 순수한 인과 실험은 아닙니다.','',
        '## 기존 Selection에서 비교','',
        '| 방식 | 9클래스 Macro F1 | 종류·숙도 정답률 | 지원 과일 거부·검출 실패 | 미지 과일 오수락 |',
        '|---|---:|---:|---:|---:|']
    for arm in comparison:
        m=comparison[arm]['selection']['old']
        lines.append(f"| {names[arm]} | {m['nine_class_macro_f1']:.4f} | {m['joint_accuracy']:.2%} | {m['known_rejection_or_missing_rate']:.2%} | {round(m['unknown_false_acceptance_rate']*40)}/40 |")
    lines += ['', '## 새 Test: 학습에 쓰지 않은 원본 그룹 19개·131장','',
        '| 방식 | 거부 전 종류·숙도 정답률 | 모름 판정 포함 정답률 | 거부율 |',
        '|---|---:|---:|---:|']
    for arm in comparison:
        m=comparison[arm]['test']['new']
        lines.append(f"| {names[arm]} | {raw[arm]['new']:.2%} | {m['joint_accuracy']:.2%} | {m['known_rejection_or_missing_rate']:.2%} |")
    lines += ['',
        '새 Test는 모두 적숙 귤이므로 이 표는 다른 과일·숙도 전체 정확도가 아닙니다. 실제 개체/원본 출처 독립성은 검증되지 않았으며 기존 원본의 변형 계열만 분리한 평가입니다.','',
        '## 기존 Test에서 회귀 비교','',
        '| 방식 | 종류·숙도 정답률 | 지원 과일 거부·검출 실패 | 미지 과일 오수락 |',
        '|---|---:|---:|---:|']
    for arm in comparison:
        m=comparison[arm]['test']['old']
        lines.append(f"| {names[arm]} | {m['joint_accuracy']:.2%} | {m['known_rejection_or_missing_rate']:.2%} | {round(m['unknown_false_acceptance_rate']*40)}/40 |")
    lines += ['',
        '기존 Test는 이미 여러 번 확인한 회귀 자료입니다. M06~M09 이동과 M10 시연 제외 때문에 귤의 미숙·적숙 Test는 남아 있지 않습니다.','',
        '## 시연 M10 결과','',
        '| 방식 | 최종 상태 | 후보 | 특징 거리 | 수락 기준 |',
        '|---|---|---|---:|---:|']
    for arm,r in demo.items():
        lines.append(f"| {names[arm]} | {r['status']} | {r['candidate_class']} | {r['score']:.4f} | {r['threshold']:.4f} |")
    lines += ['',
        'M10 결과는 정책 선택 후 진단했으며, 통과시키기 위해 기준을 다시 풀거나 정답 라벨을 강제로 넣지 않았습니다.','',
        '## 한계와 저장 위치','',
        '- 새 자료는 주로 주황색·밝은 배경의 스크린샷/외부 이미지로 보이며, M10의 황록색·회색 배경과 촬영 조건은 여전히 다릅니다.',
        '- 원본 파일 그룹 198개가 실제 서로 다른 과일 198개라는 근거는 없습니다. 같은 원본의 다른 스크린샷이나 같은 과일의 다른 사진이 남아 있을 수 있습니다.',
        '- 기존 과숙 자료의 동일 원본 의심 쌍은 이번 실험에서 변경하지 않았습니다. 미지 과일도 기존 포도·레몬의 제한된 표본입니다.',
        '- 새 모델은 이 실행 폴더의 diversity, 대조 모델은 control, 기존 모델 사본은 frozen_baseline에 보존했습니다. 기존 기본 모델은 자동 교체하지 않았습니다.',
        '- 전체 표본·분할·크롭 경로: audit.json, 라벨 등록: registration.json, 정책 선택: decision.json, 상세 비교: comparison.json.',
        '- 새 후보의 추적 영상: outputs/video_experiment/fruit_conveyor_diversity_tracked.mp4.','',
        '```powershell',
        f'python -m src.video --tracking --model-dir outputs/mandarin_diversity/{run.name}/diversity --output outputs/video_experiment/fruit_conveyor_diversity_tracked.mp4 --display',
        '```']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(run/'report.md')
    print(json.dumps({arm:{k:r[k] for k in ('status','candidate_class','score','threshold')} for arm,r in demo.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
