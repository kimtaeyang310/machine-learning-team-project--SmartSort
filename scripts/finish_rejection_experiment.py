"""Add rejection-reduction operating points without selecting on demo/test."""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
import joblib
from src.joint import ROOT,JointClassifier
from src.joint_core import CLASSES,dump,sha
from src.rejection import unfamiliarity_scores,acceptance_mask
from scripts.select_crop_policy import evaluate
from src.video import detect_regions,crop_region


def main():
    run=Path((ROOT/'outputs/rejection_experiment/latest_run.txt').read_text(encoding='utf-8'))
    candidates=json.loads((run/'candidates.json').read_text(encoding='utf-8'))
    baseline=candidates[0]['selection']
    points={}
    for name,budget in [('low_rejection_0',0.),('low_rejection_5',.05),('low_rejection_10',.10)]:
        eligible=[r for r in candidates if r['selection']['unknown_false_acceptance_rate']<=budget+1e-12
            and r['selection']['nine_class_macro_f1']>=baseline['nine_class_macro_f1']
            and r['selection']['joint_accuracy']>=baseline['joint_accuracy']]
        points[name]=min(eligible,key=lambda r:(r['selection']['known_rejection_or_missing_rate'],
            -r['selection']['nine_class_macro_f1'],r['selection']['unknown_false_acceptance_rate'],len(r['acceptance']),r['id']))
    dump(run/'rejection_reduction_decision.json',{'operating_points':points,
        'rule':'Minimize known rejection subject to no decrease in selection joint accuracy or nine-class macro F1, under separate selection unknown acceptance budgets 0%,5%,10%.',
        'demo_used_for_selection':False,'test_used_for_selection':False,'default_model_changed':False})
    directory=ROOT/'models/joint_nine_crop'
    bundle=json.loads((directory/'bundle.json').read_text(encoding='utf-8'))
    old=joblib.load(directory/'policy.joblib')
    source=Path(bundle['source'])
    cache=np.load(source/'crop_features.npz')
    p,e=cache['probabilities'],cache['embeddings']
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    valid=np.isfinite(p).all(axis=1)
    scores={'embedding':np.full(len(audit),np.inf)}
    scores['embedding'][valid]=unfamiliarity_scores(p[valid],e[valid],old['reference'])['embedding']
    test={}
    for name,row in points.items():
        policy={**old,'acceptance':row['acceptance']}
        candidate=run/name;candidate.mkdir()
        joblib.dump(policy,candidate/'policy.joblib')
        shutil.copy2(directory/'model.pt',candidate/'model.pt')
        shutil.copy2(run/'strict/split_manifest.json',candidate/'split_manifest.json')
        dump(candidate/'bundle.json',{**bundle,'source':str(run),'adaptation':'experimental gate only; network unchanged',
            'acceptance_rule':row['acceptance'],
            'sha256':{file:sha(candidate/file) for file in ('model.pt','policy.joblib','split_manifest.json')}})
        test[name]=evaluate(audit,p,scores,policy,'test')
    dump(run/'rejection_reduction_test.json',test)
    # Single frozen-model prediction for the demo; inspect all predefined rules
    # only after all operating-point decisions have been saved.
    cap=cv2.VideoCapture(str(ROOT/'outputs/video_experiment/fruit_conveyor.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,90)
    ok,frame=cap.read();cap.release()
    if not ok:
        raise ValueError('Cannot read demo frame')
    box=max(detect_regions(frame),key=lambda b:b[0])
    model=JointClassifier(directory)
    dp,de=model.engine.raw(crop_region(frame,box))
    ds=unfamiliarity_scores(dp[None],de[None],old['reference'])
    demo=[]
    for row in candidates:
        accepted=bool(acceptance_mask(dp[None],ds,{**old,'acceptance':row['acceptance']},CLASSES)[0])
        demo.append({'candidate_id':row['id'],'accepted':accepted,
            'selection_unknown_false_acceptance_rate':row['selection']['unknown_false_acceptance_rate'],
            'selection_macro_f1':row['selection']['nine_class_macro_f1'],
            'acceptance_rule':row['acceptance']})
    active={name:next(x['accepted'] for x in demo if x['candidate_id']==row['id']) for name,row in points.items()}
    ordered=np.sort(dp)
    fruit_scores={fruit:float(sum(dp[i] for i,label in enumerate(CLASSES) if label.startswith(fruit+'_'))) for fruit in ('banana','mandarin','tomato')}
    dump(run/'demo_diagnostic.json',{'frame':90,'box':list(box),'candidate_class':CLASSES[int(dp.argmax())],
        'joint_confidence':float(ordered[-1]),'joint_margin':float(ordered[-1]-ordered[-2]),
        'fruit_scores':fruit_scores,'embedding_score':float(ds['embedding'][0]),
        'accepted_by_chosen_points':active,'all_rules':demo,'used_for_selection':False})
    print('Operating points',json.dumps({name:{'rule':r['acceptance'],'selection':r['selection'],'demo_accepted':active[name]} for name,r in points.items()},ensure_ascii=False,indent=2),flush=True)
    accepted=[r for r in demo if r['accepted']]
    print('Demo',CLASSES[int(dp.argmax())],float(ordered[-1]),float(ordered[-1]-ordered[-2]),flush=True)
    print('Demo-accepting rules',len(accepted),'minimum selection unknown acceptance',min([r['selection_unknown_false_acceptance_rate'] for r in accepted],default=None),flush=True)


if __name__=='__main__':
    main()
