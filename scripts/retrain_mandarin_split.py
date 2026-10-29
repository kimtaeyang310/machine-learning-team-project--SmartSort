"""Apply the authorized M06-M09 split change and retrain the video crop model."""
from pathlib import Path
from collections import Counter
from itertools import product
import csv
import copy
import hashlib
import json
import shutil
from time import perf_counter
import cv2
import numpy as np
import joblib
import torch
from ultralytics import YOLO
from src.joint import ROOT,MODEL_DIR,JointClassifier,stamp
from src.joint_labels import refresh_labels
from src.joint_core import CLASSES,FRUITS,NinePredictor,dump,sha
from src.rejection import unfamiliarity_scores
from scripts.select_crop_policy import evaluate


def policy_scores(p,e,policy):
    valid=np.isfinite(p).all(axis=1)
    scores={'embedding':np.full(len(p),np.inf)}
    scores['embedding'][valid]=unfamiliarity_scores(p[valid],e[valid],policy['reference'])['embedding']
    return scores


def select_policy(audit,p,e):
    valid=np.isfinite(p).all(axis=1)
    train=[i for i,r in enumerate(audit) if valid[i] and not r['excluded_demo_group'] and r['phase']=='train' and r['fruit'] in FRUITS]
    ref=e[train]
    policy={'reference':ref,'method':'embedding','thresholds':{'embedding':{f:0. for f in FRUITS}},'class_thresholds':{}}
    scores=policy_scores(p,e,policy)
    levels=[.05,.10,.25,.50,.65,.80,.90,.95]
    keys=['banana_overripe','banana_unripe','banana_ripe','mandarin','tomato']
    options={}
    for key in keys:
        ids=[i for i,r in enumerate(audit) if valid[i] and not r['excluded_demo_group'] and r['phase']=='calibration'
             and ((r['fruit']+'_'+r['ripeness']==key and CLASSES[int(p[i].argmax())]==key) if key.startswith('banana_') else (r['fruit']==key and CLASSES[int(p[i].argmax())].split('_')[0]==key))]
        if len(ids)<10:
            raise ValueError(f'Insufficient calibration examples: {key}, {len(ids)}')
        options[key]=sorted(set([0.]+[float(np.quantile(scores['embedding'][ids],q,method='higher'))+1e-6 for q in levels]))
        if key.startswith('banana_'):
            policy['class_thresholds'][key]=options[key][-1]
        else:
            policy['thresholds']['embedding'][key]=options[key][-1]
    policy['thresholds']['embedding']['banana']=max(policy['class_thresholds'].values())
    key_metric=lambda m:(m['unknown_false_acceptance_rate'],-m['nine_class_macro_f1'],m['known_rejection_or_missing_rate'])
    history=[]
    for pass_index in range(3):
        changed=False
        for key in keys:
            best_metric=evaluate(audit,p,scores,policy,'selection')
            best=policy
            for threshold in options[key]:
                candidate=copy.deepcopy(policy)
                if key.startswith('banana_'):
                    candidate['class_thresholds'][key]=threshold
                else:
                    candidate['thresholds']['embedding'][key]=threshold
                metric=evaluate(audit,p,scores,candidate,'selection')
                history.append({'pass':pass_index,'key':key,'threshold':threshold,'metrics':metric})
                if key_metric(metric)<key_metric(best_metric):
                    best_metric,best=metric,candidate
            changed=changed or best is not policy
            policy=best
        if not changed:
            break
    return policy,scores,history


def main():
    started=perf_counter()
    old_dir=ROOT/'models/joint_nine_crop'
    old_bundle=json.loads((old_dir/'bundle.json').read_text(encoding='utf-8'))
    previous=Path(old_bundle['source'])
    run=ROOT/'outputs/crop_calibration'/('split_retrain_'+stamp())
    run.mkdir(parents=True)
    labels_path=ROOT/'data/metadata/fruit_labels.csv'
    shutil.copy2(labels_path,run/'fruit_labels.before.csv')
    shutil.copytree(old_dir,run/'previous_video_bundle')
    audit=json.loads((previous/'audit.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader(labels_path.open(encoding='utf-8-sig')))
    changes=[]
    for row in rows:
        if row['fruit']=='mandarin' and row['fruit_id'] in {'M06','M07','M08','M09'}:
            before=dict(row)
            row['split']='train'
            row['phase']='selection' if row['fruit_id']=='M09' else 'train'
            changes.append({'before':before,'after':dict(row)})
    if len(changes)!=4:
        raise ValueError(f'Expected four CSV rows, found {len(changes)}')
    fields=list(rows[0])
    if 'phase' not in fields:
        fields.append('phase')
    temporary=run/'fruit_labels.updated.csv'
    with temporary.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    updated,label_changes=refresh_labels(audit,temporary)
    for old,new in zip(audit,updated):
        old.update(ripeness=new['ripeness'],phase=new['phase'])
    if any(r['phase']!='selection' for r in audit if r['group']=='mandarin:M09'):
        raise ValueError('M09 is not held out from training')
    for group in ('mandarin:M06','mandarin:M07','mandarin:M08'):
        if any(r['phase']!='train' for r in audit if r['group']==group):
            raise ValueError('Requested training group misplaced')
    dump(run/'csv_changes.json',changes)
    dump(run/'label_changes.json',label_changes)
    # Validate every source and cached crop before changing the authoritative CSV.
    for r in audit:
        if sha(ROOT/r['path'])!=r['sha256']:
            raise ValueError(f"Source changed: {r['path']}")
        if not r['excluded_demo_group'] and r['box'] and not (ROOT/r.get('crop_path','missing')).is_file():
            raise ValueError(f"Missing cached crop: {r['path']}")
    shutil.copy2(temporary,labels_path)
    shutil.copy2(labels_path,run/'fruit_labels.csv')
    manifest=[{k:r[k] for k in ('path','fruit','ripeness','group','source_group','phase','sha256')} for r in audit]
    dump(run/'split_manifest.json',manifest)
    dump(run/'audit.json',audit)
    dataset=run/'dataset'
    for split in ('train','val'):
        for label in CLASSES:
            (dataset/split/label).mkdir(parents=True)
    counts=Counter();seen=set()
    for r in audit:
        if r['phase']!='train' or r['fruit'] not in FRUITS or r['excluded_demo_group'] or not r['box']:
            continue
        label=r['fruit']+'_'+r['ripeness']
        source=ROOT/r['crop_path']
        shutil.copy2(source,dataset/'train'/label/source.name)
        counts[label]+=1
        if label not in seen:
            shutil.copy2(source,dataset/'val'/label/source.name);seen.add(label)
    if seen!=set(CLASSES):
        raise ValueError('Missing training class')
    config={'epochs':6,'imgsz':224,'batch':16,'device':'cpu','seed':42,'initial_checkpoint':str(run/'previous_video_bundle/model.pt'),
        'training_counts':dict(counts),'split_assignment':{'M06':'train','M07':'train','M08':'train','M09':'selection'},
        'selection':'fixed 6 epochs last.pt; thresholds from calibration; choose on selection only; no test-driven model selection',
        'crop':'unchanged 18% padding; same cached crops with source hashes reverified',
        'caveat':'M06-M09 were previously inspected test individuals; new metrics use changed test membership and are regression results, not fresh blind evaluation.'}
    dump(run/'config.json',config)
    print('CSV applied. Training crops',sum(counts.values()),'M09 held out.',flush=True)
    torch.set_num_threads(4)
    model=YOLO(str(run/'previous_video_bundle/model.pt'))
    model.train(data=str(dataset),project=str(run),name='training',epochs=6,imgsz=224,batch=16,device='cpu',workers=0,
        seed=42,deterministic=True,patience=0,val=False,plots=False,save=True,exist_ok=True,
        auto_augment=None,erasing=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,verbose=False)
    checkpoint=run/'training/weights/last.pt'
    encoder=NinePredictor(checkpoint,224)
    old_cache=np.load(previous/'crop_features.npz')
    old_p,old_e=old_cache['probabilities'],old_cache['embeddings']
    p,e=np.full_like(old_p,np.nan),np.full_like(old_e,np.nan)
    for i,r in enumerate(audit):
        if r['excluded_demo_group'] or not r['box']:
            continue
        image=cv2.imdecode(np.fromfile(str(ROOT/r['crop_path']),np.uint8),cv2.IMREAD_COLOR)
        p[i],e[i]=encoder.raw(image)
        if (i+1)%400==0:
            print('Encoded retrained crops',i+1,flush=True)
    np.savez_compressed(run/'crop_features.npz',probabilities=p,embeddings=e)
    policy,scores,history=select_policy(audit,p,e)
    old_policy=joblib.load(run/'previous_video_bundle/policy.joblib')
    old_scores=policy_scores(old_p,old_e,old_policy)
    before=evaluate(audit,old_p,old_scores,old_policy,'selection')
    after=evaluate(audit,p,scores,policy,'selection')
    objective=lambda m:(m['unknown_false_acceptance_rate'],-m['nine_class_macro_f1'],m['known_rejection_or_missing_rate'])
    deploy=objective(after)<=objective(before)
    selection={'baseline_same_selection':before,'retrained':after,'deploy':deploy,'test_used_for_selection':False,
        'thresholds':policy['thresholds'],'class_thresholds':policy['class_thresholds'],
        'search':'three-pass coordinate search; calibration quantiles 5-95% plus zero-reject cutoff; selection unknown acceptance, macro F1, rejection objective'}
    dump(run/'selection.json',selection)
    dump(run/'threshold_search.json',history)
    joblib.dump(policy,run/'policy.joblib')
    # Record the frozen selection/deploy decision before test scoring.
    dump(run/'test_metrics.json',{'previous_on_new_test':evaluate(audit,old_p,old_scores,old_policy,'test'),
        'retrained_on_new_test':evaluate(audit,p,scores,policy,'test'),
        'note':'Changed test membership; not directly comparable to earlier reported metrics; no ripe or unripe mandarin remains after demo exclusion.'})
    candidate=run/'bundle';candidate.mkdir()
    for source,name in [(checkpoint,'model.pt'),(run/'policy.joblib','policy.joblib'),(run/'split_manifest.json','split_manifest.json')]:
        shutil.copy2(source,candidate/name)
    dump(candidate/'bundle.json',{'format':1,'imgsz':224,'model_name':'yolo26n_cls_joint_nine_crop',
        'source':str(run),'adaptation':'M06/M07/M08 train, M09 ripe selection; six-epoch crop continuation and development-only gate calibration',
        'preprocessing':'max-side 320; HSV foreground; largest region; 18% crop padding',
        'sha256':{name:sha(candidate/name) for name in ('model.pt','policy.joblib','split_manifest.json')}})
    JointClassifier(candidate)
    if deploy:
        for name in ('model.pt','policy.joblib','split_manifest.json','bundle.json'):
            shutil.copy2(candidate/name,old_dir/name)
        JointClassifier(old_dir)
    dump(run/'result.json',{'run':str(run),'deployed':deploy,'elapsed_seconds':perf_counter()-started,
        'photo_model_preserved':True,'previous_video_model_backup':str(run/'previous_video_bundle')})
    (ROOT/'outputs/crop_calibration/latest_run.txt').write_text(str(run),encoding='utf-8')
    print('RETRAIN_RUN',run,flush=True)
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    print((run/'test_metrics.json').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    main()
