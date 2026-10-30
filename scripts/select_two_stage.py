"""Freeze a two-stage policy on calibration/selection, then score regression tests."""
# 종류 판정과 숙도를 분리한 모델의 정책을 검증 자료로 고정한 뒤 회귀 평가합니다.
from pathlib import Path
from itertools import product
import copy
import json
import shutil
import numpy as np
import joblib
from sklearn.metrics import f1_score
from src.joint import ROOT
from src.joint_core import dump,sha,CLASSES
from src.two_stage import FRUITS,STAGES,kind_gate,TwoStageClassifier
from src.rejection import unfamiliarity_scores
from scripts.select_crop_policy import evaluate as evaluate_joint


def evaluate(audit,p,stage,distances,policy,phase):
    accepted,_,_=kind_gate(p,None,policy,distances=distances)
    ids=[i for i,r in enumerate(audit) if r['phase']==phase and not r['excluded_demo_group']]
    known=[i for i in ids if audit[i]['fruit'] in FRUITS]
    unknown=[i for i in ids if audit[i]['fruit'] not in FRUITS]
    predictions={};kinds={};missing=0
    for i in ids:
        if not np.isfinite(p[i]).all() or not np.isfinite(stage[i]).all():
            predictions[i]='no_detection';kinds[i]='no_detection';missing+=1
        else:
            kinds[i]=FRUITS[int(p[i].argmax())]
            predictions[i]=kinds[i]+'_'+STAGES[int(stage[i].argmax())] if accepted[i] else 'unknown'
    truth=[audit[i]['fruit']+'_'+audit[i]['ripeness'] for i in known]
    routed_correct=[i for i in known if kinds[i]==audit[i]['fruit']]
    result={'known_count':len(known),'unknown_count':len(unknown),
        'nine_class_macro_f1':float(f1_score(truth,[predictions[i] for i in known],labels=CLASSES,average='macro',zero_division=0)),
        'supported_class_macro_f1':float(f1_score(truth,[predictions[i] for i in known],labels=sorted(set(truth)),average='macro',zero_division=0)),
        'joint_accuracy':float(np.mean([predictions[i]==t for i,t in zip(known,truth)])),
        'kind_accuracy_before_gate_including_missing':float(np.mean([kinds[i]==audit[i]['fruit'] for i in known])),
        'stage_accuracy_on_correct_kind':float(np.mean([STAGES[int(stage[i].argmax())]==audit[i]['ripeness'] for i in routed_correct])) if routed_correct else None,
        'known_rejection_or_missing_rate':float(np.mean([not accepted[i] for i in known])),
        'unknown_false_acceptance_rate':float(np.mean([accepted[i] for i in unknown])) if unknown else None,
        'missing_detection_count':missing,'class_support':{label:truth.count(label) for label in CLASSES},'per_fruit':{}}
    for fruit in FRUITS:
        selected=[i for i in known if audit[i]['fruit']==fruit]
        result['per_fruit'][fruit]={'count':len(selected),
            'macro_f1':float(f1_score([audit[i]['fruit']+'_'+audit[i]['ripeness'] for i in selected],[predictions[i] for i in selected],labels=[fruit+'_'+s for s in STAGES],average='macro',zero_division=0)),
            'rejection_or_missing_rate':float(np.mean([not accepted[i] for i in selected])) if selected else None}
    return result


def main():
    run=Path((ROOT/'outputs/two_stage_experiment/latest_run.txt').read_text(encoding='utf-8'))
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    cache=np.load(run/'features.npz')
    p,e,stage=cache['kind_probabilities'],cache['embeddings'],cache['stage_probabilities']
    valid=np.isfinite(p).all(axis=1)
    train=[i for i,r in enumerate(audit) if valid[i] and r['phase']=='train' and not r['excluded_demo_group'] and r['fruit'] in FRUITS]
    reference=e[train]
    distances=np.full(len(audit),np.inf)
    distances[valid]=unfamiliarity_scores(p[valid],e[valid],reference)['embedding']
    options={}
    calibration=[]
    levels=[.05,.10,.25,.50,.65,.80,.90,.95]
    for fruit in FRUITS:
        ids=[i for i,r in enumerate(audit) if valid[i] and not r['excluded_demo_group'] and r['phase']=='calibration' and r['fruit']==fruit]
        correct=[i for i in ids if FRUITS[int(p[i].argmax())]==fruit]
        if len(correct)<10:
            raise ValueError(f'Too few correctly classified calibration kinds: {fruit}')
        options[fruit]=sorted(set([0.]+[float(np.quantile(distances[correct],q,method='higher'))+1e-6 for q in levels]))
        calibration.append({'fruit':fruit,'count':len(ids),'kind_correct':len(correct)})
    dump(run/'gate_search_config.json',{'distance_quantiles':levels,'calibration':calibration,
        'strict_budget':0.,'balanced_budget':.05,
        'rule':'maximize selection nine-class Macro F1; minimize known rejection then unknown acceptance under each budget',
        'demo_used_for_selection':False,'test_used_for_selection':False})
    candidates=[]
    for thresholds in product(*(options[f] for f in FRUITS)):
        policy={'reference':reference,'thresholds':dict(zip(FRUITS,thresholds)),'gate':{'kind':'distance'}}
        metrics=evaluate(audit,p,stage,distances,policy,'selection')
        candidates.append({'thresholds':policy['thresholds'],'gate':policy['gate'],'selection':metrics})
    thresholds={f:options[f][-1] for f in FRUITS}
    gates=[]
    for confidence,margin in product([.5,.7,.85,.95,.99],[0.,.2,.5]):
        gates.append({'kind':'confidence','min_confidence':confidence,'min_margin':margin})
        for cap in [1.,1.5,2.]:
            gates.append({'kind':'both','min_confidence':confidence,'min_margin':margin,'distance_multiplier':cap})
        for cap in [2.,4.]:
            gates.append({'kind':'rescue','min_confidence':confidence,'min_margin':margin,'distance_multiplier':cap})
    for gate in gates:
        policy={'reference':reference,'thresholds':thresholds,'gate':gate}
        candidates.append({'thresholds':thresholds,'gate':gate,'selection':evaluate(audit,p,stage,distances,policy,'selection')})
    dump(run/'gate_candidates.json',candidates)
    selected={}
    for name,budget in [('strict',0.),('balanced',.05)]:
        allowed=[r for r in candidates if r['selection']['unknown_false_acceptance_rate']<=budget+1e-12]
        selected[name]=max(allowed,key=lambda r:(r['selection']['nine_class_macro_f1'],-r['selection']['known_rejection_or_missing_rate'],-r['selection']['unknown_false_acceptance_rate']))
    # Commit the policies before scoring Test or predicting demonstration images.
    dump(run/'decision.json',{'operating_points':selected,'default_model_changed':False,'demo_used_for_selection':False,'test_used_for_selection':False})
    manifest=[{k:r[k] for k in ('path','fruit','ripeness','group','source_group','phase','sha256')} for r in audit]
    for name,entry in selected.items():
        destination=run/name;destination.mkdir()
        policy={'reference':reference,'thresholds':entry['thresholds'],'gate':entry['gate']}
        joblib.dump(policy,destination/'policy.joblib')
        for head in ['kind']+[f+'_stage' for f in FRUITS]:
            shutil.copy2(run/head/'weights/last.pt',destination/(head+'.pt'))
        dump(destination/'split_manifest.json',manifest)
        files=['kind.pt']+[f+'_stage.pt' for f in FRUITS]+['policy.joblib','split_manifest.json']
        dump(destination/'bundle.json',{'format':1,'architecture':'two_stage','imgsz':224,'model_name':'yolo26n_kind_then_fruit_stage',
            'source':str(run),'kind_labels':FRUITS,'stage_labels':STAGES,'gate':entry['gate'],
            'training':'6 fixed epochs for kind model and each separate per-fruit stage model',
            'sha256':{file:sha(destination/file) for file in files}})
        TwoStageClassifier(destination)
    old=joblib.load(ROOT/'models/joint_nine_crop/policy.joblib')
    old_cache=np.load(config['baseline_cache'])
    old_p,old_e=old_cache['probabilities'],old_cache['embeddings']
    old_valid=np.isfinite(old_p).all(axis=1)
    old_scores={'embedding':np.full(len(audit),np.inf)}
    old_scores['embedding'][old_valid]=unfamiliarity_scores(old_p[old_valid],old_e[old_valid],old['reference'])['embedding']
    comparison={'baseline_selection':evaluate_joint(audit,old_p,old_scores,old,'selection'),
        'baseline_test':evaluate_joint(audit,old_p,old_scores,old,'test'),
        'two_stage':{name:{'selection':entry['selection'],
            'test':evaluate(audit,p,stage,distances,{'reference':reference,'thresholds':entry['thresholds'],'gate':entry['gate']},'test')}
            for name,entry in selected.items()},
        'limitations':'Previously inspected regression data; no unripe/ripe mandarin remains in test; suspected shared-source overripe images remain.'}
    dump(run/'comparison.json',comparison)
    dump(run/'progress.json',{'phase':'selection_and_regression_complete','candidate_count':len(candidates)})
    print('Selected policies',json.dumps(selected,ensure_ascii=False,indent=2),flush=True)
    print('TWO_STAGE_SELECTED',run,flush=True)


if __name__=='__main__':
    main()
