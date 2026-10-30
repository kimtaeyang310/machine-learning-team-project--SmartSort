"""Fixed-network comparison of distance, confidence, margin and hybrid rejection."""
# 가중치를 고정하고 거리·최대 확률·상위 확률 차이·혼합 규칙의 거부 성능을 비교합니다.
from pathlib import Path
from itertools import product
import copy
import json
import shutil
import joblib
import numpy as np
from src.joint import ROOT,JointClassifier,stamp
from src.joint_core import CLASSES,dump,sha
from src.joint_labels import refresh_labels
from src.rejection import unfamiliarity_scores
from scripts.select_crop_policy import evaluate


def main():
    run=ROOT/'outputs/rejection_experiment'/('run_'+stamp())
    run.mkdir(parents=True)
    directory=ROOT/'models/joint_nine_crop'
    bundle=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
    source=Path(bundle['source'])
    if sha(source/'finetune/weights/last.pt')!=bundle['sha256']['model.pt']:
        raise ValueError('Feature cache does not match frozen default model')
    old=joblib.load(directory/'policy.joblib')
    audit=json.loads((source/'audit.json').read_text(encoding='utf-8'))
    audit,changes=refresh_labels(audit,ROOT/'data/metadata/fruit_labels.csv')
    cache=np.load(source/'crop_features.npz')
    p,e=cache['probabilities'],cache['embeddings']
    valid=np.isfinite(p).all(axis=1)
    scores={'embedding':np.full(len(audit),np.inf)}
    scores['embedding'][valid]=unfamiliarity_scores(p[valid],e[valid],old['reference'])['embedding']
    calibration=[i for i,r in enumerate(audit) if valid[i] and r['phase']=='calibration' and not r['excluded_demo_group']
                 and CLASSES[int(p[i].argmax())].split('_',1)[0]==r['fruit']]
    ordered=np.sort(p[calibration],axis=1)
    confidence_candidates=sorted(set([.5,.7,.85,.95,.99]+[round(float(np.quantile(ordered[:,-1],q)),6) for q in (.05,.25)]))
    margin_candidates=sorted(set([0.,.2,.5]+[round(float(np.quantile(ordered[:,-1]-ordered[:,-2],.05)),6)]))
    configs=[{'kind':'distance'}]
    configs += [{'kind':'confidence','min_confidence':c} for c in confidence_candidates]
    configs += [{'kind':'margin','min_margin':m} for m in margin_candidates]
    for cap,confidence,margin,fruit_confidence in product([1.5,2.,3.,4.],confidence_candidates,margin_candidates,[.8,.95,.99]):
        configs.append({'kind':'rescue','distance_multiplier':cap,'min_confidence':confidence,
                        'min_margin':margin,'min_fruit_confidence':fruit_confidence})
    for confidence,margin in product(confidence_candidates,margin_candidates):
        configs.append({'kind':'rescue','distance_multiplier':None,'min_confidence':confidence,'min_margin':margin,'min_fruit_confidence':.99})
    for cap,confidence in product([1.,1.5,2.],confidence_candidates):
        configs.append({'kind':'gated_distance','distance_multiplier':cap,'min_confidence':confidence,'min_fruit_confidence':.95})
    config={'model_sha256':bundle['sha256']['model.pt'],'policy_sha256':bundle['sha256']['policy.joblib'],
        'source_cache':str(source),'csv_changes':changes,'candidate_count':len(configs),
        'confidence_candidates':confidence_candidates,'margin_candidates':margin_candidates,
        'rules':'Prefer largest selection 9-class Macro F1, then lowest rejection, then lowest unknown acceptance, then simplest config. Report separate 0% and <=5% selection unknown-acceptance budgets.',
        'budgets':[0.,.05],'demo_used_for_selection':False,'test_used_for_selection':False,
        'frozen':'network, feature reference, distance thresholds, crop preprocessing and current CSV membership are fixed'}
    dump(run/'config.json',config)
    dump(run/'audit.json',audit)
    # Selection is the only data scored during the search.
    candidates=[]
    for index,acceptance in enumerate(configs):
        policy={**old,'acceptance':acceptance}
        metrics=evaluate(audit,p,scores,policy,'selection')
        candidates.append({'id':index,'acceptance':acceptance,'selection':metrics})
        if (index+1)%100==0:
            print('Compared gate candidates',index+1,'/',len(configs),flush=True)
    dump(run/'candidates.json',candidates)
    baseline=candidates[0]
    results={}
    for name,budget in [('strict',0.),('balanced',.05)]:
        allowed=[r for r in candidates if r['selection']['unknown_false_acceptance_rate']<=budget+1e-12]
        best=max(allowed,key=lambda r:(r['selection']['nine_class_macro_f1'],
            -r['selection']['known_rejection_or_missing_rate'],-r['selection']['unknown_false_acceptance_rate'],
            -len(r['acceptance']),-r['id']))
        results[name]=best
    # Commit the chosen rules before viewing demo predictions or test scores.
    dump(run/'decision.json',{'baseline':baseline,'operating_points':results,'default_model_changed':False,
        'demo_used_for_selection':False,'test_used_for_selection':False})
    for name,best in results.items():
        candidate=run/name;candidate.mkdir()
        policy={**old,'acceptance':best['acceptance']}
        joblib.dump(policy,candidate/'policy.joblib')
        shutil.copy2(directory/'model.pt',candidate/'model.pt')
        manifest=[{k:r[k] for k in ('path','fruit','ripeness','group','source_group','phase','sha256')} for r in audit]
        dump(candidate/'split_manifest.json',manifest)
        dump(candidate/'bundle.json',{**bundle,'source':str(run),'adaptation':'fixed crop network; experimental confidence/margin hybrid rejection',
            'sha256':{file:sha(candidate/file) for file in ('model.pt','policy.joblib','split_manifest.json')}})
        JointClassifier(candidate)
    dump(run/'test_comparison.json',{'baseline':evaluate(audit,p,scores,old,'test'),
         **{name:evaluate(audit,p,scores,{**old,'acceptance':best['acceptance']},'test') for name,best in results.items()},
         'note':'Previously inspected regression test; no unripe/ripe mandarin remains after authorized split move and demo exclusion.'})
    assert sha(directory/'model.pt')==bundle['sha256']['model.pt']
    assert sha(directory/'policy.joblib')==bundle['sha256']['policy.joblib']
    (run.parent/'latest_run.txt').write_text(str(run),encoding='utf-8')
    print('REJECTION_RUN',str(run),flush=True)
    for name,row in [('baseline',baseline)]+list(results.items()):
        print(name,json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
