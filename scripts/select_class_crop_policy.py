"""Select banana stage-specific gates using calibration/selection only."""
# 바나나의 숙도별 판정 기준을 calibration·selection 자료로 선택합니다.
# 과일 공통 임계값보다 세밀한 클래스별 임계값을 비교하는 단계입니다.
from pathlib import Path
from itertools import product
import json
import shutil
import joblib
import numpy as np
from src.joint import ROOT, JointClassifier
from src.joint_core import CLASSES, dump, sha
from src.rejection import unfamiliarity_scores
from scripts.select_crop_policy import evaluate


def main():
    run = Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    policy = joblib.load(run/'crop_policy.joblib')
    old_selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    cache = np.load(run/'crop_features.npz')
    p,e = cache['probabilities'],cache['embeddings']
    valid = np.isfinite(p).all(axis=1)
    values = unfamiliarity_scores(p[valid],e[valid],policy['reference'])['embedding']
    scores = {'embedding':np.full(len(audit),np.inf)}
    scores['embedding'][valid] = values
    labels = ['banana_'+s for s in ('unripe','ripe','overripe')]
    options = {}
    levels = [.05,.10,.25,.50,.65,.80,.90,.95]
    for label in labels:
        indices = [i for i,r in enumerate(audit) if valid[i] and not r['excluded_demo_group'] and r['phase']=='calibration' and r['fruit']+'_'+r['ripeness']==label and CLASSES[int(p[i].argmax())]==label]
        if len(indices)<10:
            raise ValueError(f'Insufficient calibration: {label}')
        options[label] = sorted(set([policy['thresholds']['embedding']['banana']] + [float(np.quantile(scores['embedding'][indices],q,method='higher'))+1e-6 for q in levels]))
    best = None
    key = lambda m:(m['unknown_false_acceptance_rate'],-m['nine_class_macro_f1'],m['known_rejection_or_missing_rate'])
    candidates = []
    for thresholds in product(*(options[label] for label in labels)):
        candidate = {**policy,'class_thresholds':dict(zip(labels,thresholds))}
        metric = evaluate(audit,p,scores,candidate,'selection')
        candidates.append({'class_thresholds':candidate['class_thresholds'],'selection':metric})
        if best is None or key(metric)<key(best[0]):
            best = metric,candidate
    metric,candidate = best
    if key(metric) >= key(old_selection['candidates']['embedding']):
        print('No development improvement; kept fruit gates')
        return
    for name in ('selection.json','test_metrics.json','crop_policy.joblib'):
        shutil.copy2(run/name,run/('strict_fruit_'+name))
    selection = {**old_selection,'candidates':{'embedding':metric},'class_thresholds':candidate['class_thresholds'],
        'calibration_acceptance_target':'calibration quantile grid; banana stage-specific, mandarin/tomato fruit-specific; no fixed acceptance guarantee',
        'selection_rule':'minimize selection unknown false acceptance, maximize selection 9-class macro F1, minimize known rejection; all thresholds from calibration',
        'test_caveat':'test viewed for previous candidates; final result is exploratory regression and requires new independent data'}
    dump(run/'class_gate_search.json',{'quantile_grid':levels,'candidates':candidates,'test_used_in_threshold_search':False})
    dump(run/'selection.json',selection)
    joblib.dump(candidate,run/'crop_policy.joblib')
    old_tests = json.loads((run/'strict_fruit_test_metrics.json').read_text(encoding='utf-8'))
    dump(run/'test_metrics.json',{'baseline':old_tests['baseline'],'crop_calibrated':evaluate(audit,p,scores,candidate,'test'),'note':selection['test_caveat']})
    destination = Path(json.loads((run/'installed.json').read_text(encoding='utf-8'))['model_dir'])
    # Retain the immediately previous policy and bundle metadata before replacing.
    for name in ('policy.joblib','bundle.json'):
        shutil.copy2(destination/name,run/('strict_fruit_bundle_'+name))
    shutil.copy2(run/'crop_policy.joblib',destination/'policy.joblib')
    bundle = json.loads((destination/'bundle.json').read_text(encoding='utf-8'))
    bundle['adaptation'] = 'six-epoch crop fine-tuning; calibration/selection-based banana stage-specific embedding gates'
    bundle['sha256']['policy.joblib'] = sha(destination/'policy.joblib')
    dump(destination/'bundle.json',bundle)
    JointClassifier(destination)
    print(json.dumps({'selection':metric,'class_thresholds':candidate['class_thresholds']},ensure_ascii=False,indent=2),flush=True)
    print((run/'test_metrics.json').read_text(encoding='utf-8'),flush=True)


if __name__ == '__main__':
    main()
