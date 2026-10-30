"""Select diversity/control policies without test or demonstration-based tuning."""
from pathlib import Path
import copy
import json
import shutil
import joblib
import numpy as np
from src.joint import ROOT,JointClassifier
from src.joint_core import dump,sha
from scripts.retrain_mandarin_split import select_policy,policy_scores
from scripts.select_crop_policy import evaluate


def subset(audit,cohort):
    result=copy.deepcopy(audit)
    for r in result:
        if cohort=='old' and r.get('new_1029'):r['excluded_demo_group']=True
        if cohort=='new' and not r.get('new_1029'):r['excluded_demo_group']=True
    return result


def main():
    run=Path((ROOT/'outputs/mandarin_diversity/latest_run.txt').read_text(encoding='utf-8'))
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    old=subset(audit,'old');new=subset(audit,'new')
    policies={'baseline':joblib.load(run/'frozen_baseline/policy.joblib')}
    selections={};caches={};scores_by_arm={}
    for name in ('baseline','control','diversity'):
        data=np.load(run/(name+'_features.npz'))
        p,e=data['probabilities'],data['embeddings'];caches[name]=(p,e)
        if name!='baseline':
            policy,scores,history=select_policy(audit,p,e)
            policies[name]=policy
            dump(run/(name+'_threshold_search.json'),history)
        else:scores=policy_scores(p,e,policies[name])
        scores_by_arm[name]=scores
        selections[name]={cohort:evaluate(rows,p,scores,policies[name],'selection') for cohort,rows in [('combined',audit),('old',old),('new',new)]}
        print('Selected',name,'combined F1',selections[name]['combined']['nine_class_macro_f1'],flush=True)
    # Save all policies and selection decisions before examining either test set.
    decision={'selection':selections,'default_model_changed':False,'demo_used_for_selection':False,'test_used_for_selection':False,
        'rule':'same embedding gate, calibration quantile candidates; minimize selection unknown acceptance then maximize macro F1 and minimize rejection',
        'control':'old-only 6 epochs with new calibration; diversity adds new training/reference crops; same calibration/selection data'}
    dump(run/'decision.json',decision)
    for name in ('control','diversity'):
        directory=run/name;directory.mkdir()
        checkpoint=run/'control_model.pt' if name=='control' else run/'training/weights/last.pt'
        shutil.copy2(checkpoint,directory/'model.pt')
        joblib.dump(policies[name],directory/'policy.joblib')
        manifest=[{k:r[k] for k in ('path','fruit','ripeness','group','source_group','phase','sha256')} for r in audit if r.get('active',True)]
        dump(directory/'split_manifest.json',manifest)
        dump(directory/'bundle.json',{'format':1,'imgsz':224,'model_name':'yolo26n_cls_joint_nine_crop_'+name,
            'source':str(run),'adaptation':'additional ripe mandarin source families' if name=='diversity' else 'old-only training with new calibration',
            'arm':name,'preprocessing':'max-side 320; HSV largest region; 18% padding',
            'sha256':{file:sha(directory/file) for file in ('model.pt','policy.joblib','split_manifest.json')}})
        JointClassifier(directory)
    comparison={}
    for name in ('baseline','control','diversity'):
        p,e=caches[name];scores=scores_by_arm[name]
        comparison[name]={'selection':selections[name],
            'test':{cohort:evaluate(rows,p,scores,policies[name],'test') for cohort,rows in [('combined',audit),('old',old),('new',new)]}}
    dump(run/'comparison.json',comparison)
    dump(run/'progress.json',{'phase':'selection_and_test_complete'})
    for name in comparison:
        print(name,'old test',comparison[name]['test']['old']['joint_accuracy'],'new test',comparison[name]['test']['new']['joint_accuracy'],flush=True)
    print('DIVERSITY_SELECTED',run,flush=True)


if __name__=='__main__':main()
