"""Stricter development-only gate selection after 95% acceptance over-accepted unknowns.
Test metrics have already been inspected for the first candidate; final test
numbers are explicitly exploratory regression results, not a fresh blind test.
"""
# 미지원 과일을 과도하게 수락한 기존 기준보다 엄격한 crop 정책을 개발 자료로 선택합니다.
# 이미 확인한 Test 결과는 새로운 독립 검증이 아니라 탐색적 회귀 평가로 해석합니다.
from pathlib import Path
from itertools import product
import json
import shutil
import joblib
import numpy as np
from src.joint import ROOT, MODEL_DIR, JointClassifier
from src.joint_core import CLASSES, FRUITS, dump, sha
from src.rejection import unfamiliarity_scores
from scripts.select_crop_policy import evaluate


def main():
    run = Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    for name in ('selection.json','test_metrics.json','crop_policy.joblib'):
        shutil.copy2(run/name,run/('initial_95_'+name))
    audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    data = np.load(run/'crop_features.npz')
    p,e = data['probabilities'],data['embeddings']
    valid = np.isfinite(p).all(axis=1)
    training = [i for i,r in enumerate(audit) if valid[i] and not r['excluded_demo_group'] and r['phase']=='train' and r['fruit'] in FRUITS]
    reference = e[training]
    scores = {m:np.full(len(audit),np.inf) for m in ('msp','embedding')}
    for m,v in unfamiliarity_scores(p[valid],e[valid],reference).items():
        scores[m][valid] = v
    old_selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    levels = [.05,.10,.25,.50,.65,.80,.90,.95]
    options = {}
    for fruit in FRUITS:
        indices = [i for i,r in enumerate(audit) if valid[i] and not r['excluded_demo_group'] and r['phase']=='calibration' and r['fruit']==fruit and CLASSES[int(p[i].argmax())].split('_',1)[0]==fruit]
        if len(indices)<10:
            raise ValueError('Insufficient calibration examples')
        options[fruit] = sorted(set([float(np.quantile(scores['embedding'][indices],q,method='higher'))+1e-6 for q in levels]))
    candidates = []
    # Evaluate no test samples while searching: selection phase only.
    for thresholds in product(*(options[f] for f in FRUITS)):
        policy = {'reference':reference,'method':'embedding','thresholds':{'embedding':dict(zip(FRUITS,thresholds))}}
        metric = evaluate(audit,p,scores,policy,'selection')
        candidates.append((metric,policy))
    key = lambda m:(m['unknown_false_acceptance_rate'],-m['nine_class_macro_f1'],m['known_rejection_or_missing_rate'])
    metric,policy = min(candidates,key=lambda pair:key(pair[0]))
    promote = key(metric)<key(old_selection['baseline'])
    dump(run/'strict_selection_search.json',{'quantile_grid':levels,'candidate_count':len(candidates),
        'candidates':[{'thresholds':p['thresholds'],'selection':m} for m,p in candidates],
        'test_used_in_threshold_search':False})
    selection = {**old_selection,'candidates':{'embedding':metric},'thresholds':policy['thresholds'],
        'chosen_method':'embedding','promote_video_policy':promote,
        'calibration_acceptance_target':'selected from 5%-95% quantile grid to prioritize unknown rejection; 95% known acceptance is not guaranteed',
        'test_caveat':'initial 95%-gate test was already inspected; final result is exploratory regression, not blind validation',
        'selection_rule':'minimize unknown false acceptance, maximize macro F1, minimize rejection; thresholds drawn from calibration only; evaluated on selection only'}
    dump(run/'selection.json',selection)
    joblib.dump(policy,run/'crop_policy.joblib')
    initial_test = json.loads((run/'initial_95_test_metrics.json').read_text(encoding='utf-8'))
    dump(run/'test_metrics.json',{'baseline':initial_test['baseline'],'crop_calibrated':evaluate(audit,p,scores,policy,'test'),
        'note':selection['test_caveat']})
    if promote:
        destination = ROOT/'models/joint_nine_crop'
        if destination.exists():
            destination = run/'strict_bundle'
        if destination.exists():
            raise ValueError('Strict candidate bundle already exists; start a new experiment')
        destination.mkdir(parents=True)
        shutil.copy2(run/'finetune/weights/last.pt',destination/'model.pt')
        shutil.copy2(run/'crop_policy.joblib',destination/'policy.joblib')
        shutil.copy2(MODEL_DIR/'split_manifest.json',destination/'split_manifest.json')
        dump(destination/'bundle.json',{'format':1,'imgsz':224,'model_name':'yolo26n_cls_joint_nine_crop',
             'source':str(run),'adaptation':'six-epoch crop fine-tuning; development-selected strict embedding thresholds',
             'preprocessing':'max-side 320; HSV foreground; largest region; 18% crop padding',
             'sha256':{name:sha(destination/name) for name in ('model.pt','policy.joblib','split_manifest.json')}})
        JointClassifier(destination)
        dump(run/'installed.json',{'model_dir':str(destination)})
    print(json.dumps({'selection':metric,'promoted':promote,'thresholds':policy['thresholds']},ensure_ascii=False,indent=2),flush=True)
    print((run/'test_metrics.json').read_text(encoding='utf-8'),flush=True)


if __name__ == '__main__':
    main()
