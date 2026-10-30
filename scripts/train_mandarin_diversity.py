"""Fixed-architecture diversity experiment with old-only training control."""
from pathlib import Path
from collections import Counter
import json
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.joint import ROOT
from src.joint_core import dump,sha,CLASSES,FRUITS,NinePredictor


def train_head(dataset,run,name,weights):
    model=YOLO(str(weights))
    model.train(data=str(dataset),project=str(run),name=name,epochs=6,imgsz=224,batch=16,device='cpu',workers=0,
        seed=42,deterministic=True,patience=0,val=False,plots=False,save=True,exist_ok=True,
        auto_augment=None,erasing=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,verbose=False)
    return run/name/'weights/last.pt'


def main():
    run=Path((ROOT/'outputs/mandarin_diversity/latest_run.txt').read_text(encoding='utf-8'))
    registration=json.loads((run/'registration.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    base_dir=Path(registration['model_source'])
    if (run/'frozen_baseline').exists():raise ValueError('Run already started; do not silently retrain')
    shutil.copytree(base_dir,run/'frozen_baseline')
    base_bundle=json.loads((base_dir/'bundle.json').read_text(encoding='utf-8'))
    if sha(base_dir/'model.pt')!=base_bundle['sha256']['model.pt']:raise ValueError('Changed baseline')
    old=[r for r in audit if not r.get('new_1029')]
    candidate_train=[r for r in audit if r['phase']=='train' and not r['excluded_demo_group'] and r['box'] and r['fruit'] in FRUITS]
    source=Path(registration['baseline_feature_source'])
    # Previous old-only continuation is reusable only if the exact sample order,
    # labels, phases, hashes, initial weights and epoch protocol still match.
    control_run=ROOT/'outputs/crop_calibration/split_retrain_20261028T111029169317Z'
    control_audit=json.loads((control_run/'audit.json').read_text(encoding='utf-8'))
    control_config=json.loads((control_run/'config.json').read_text(encoding='utf-8'))
    comparable=(len(old)==len(control_audit) and all(all(a[k]==b[k] for k in ('path','group','ripeness','phase','sha256')) for a,b in zip(old,control_audit))
        and sha(control_run/'previous_video_bundle/model.pt')==base_bundle['sha256']['model.pt']
        and all(control_config.get(k)==v for k,v in {'epochs':6,'seed':42,'imgsz':224,'batch':16,'device':'cpu'}.items()))
    for r in audit:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError(f"Changed source: {r['path']}")
    dataset=run/'dataset'
    for split in ('train','val'):
        for label in CLASSES:(dataset/split/label).mkdir(parents=True)
    seen=set();counts=Counter()
    for r in candidate_train:
        label=r['fruit']+'_'+r['ripeness'];path=ROOT/r['crop_path']
        shutil.copy2(path,dataset/'train'/label/path.name);counts[label]+=1
        if label not in seen:
            shutil.copy2(path,dataset/'val'/label/path.name);seen.add(label)
    if seen!=set(CLASSES):raise ValueError('Missing training class')
    config={'epochs':6,'imgsz':224,'batch':16,'seed':42,'initial_checkpoint':str(run/'frozen_baseline/model.pt'),
        'initial_sha256':base_bundle['sha256']['model.pt'],'training_counts':dict(counts),'control_reused':bool(comparable),
        'control_run':str(control_run),'architecture':'unchanged joint 9-class YOLO crop model',
        'control':'same initial weights, old-only training 6 epochs; both adapted arms will calibrate with identical combined calibration sources',
        'selection':'last.pt fixed 6 epochs; train-only loader val ignored; test and M10 not used',
        'limitation':'Equal epochs but expanded dataset has more optimizer updates. Source families are not verified individual fruits. All new labels come from user definition.'}
    dump(run/'training_config.json',config)
    torch.set_num_threads(4)
    if comparable:
        control_checkpoint=control_run/'training/weights/last.pt'
        print('Verified cached matched old-only 6-epoch control',flush=True)
    else:
        old_data=run/'old_only_dataset'
        for split in ('train','val'):
            for label in CLASSES:(old_data/split/label).mkdir(parents=True)
        seen=set()
        for r in old:
            if r['phase']!='train' or r['excluded_demo_group'] or not r['box'] or r['fruit'] not in FRUITS:continue
            label=r['fruit']+'_'+r['ripeness'];path=ROOT/r['crop_path']
            shutil.copy2(path,old_data/'train'/label/path.name)
            if label not in seen:shutil.copy2(path,old_data/'val'/label/path.name);seen.add(label)
        control_checkpoint=train_head(old_data,run,'old_only_training',run/'frozen_baseline/model.pt')
    shutil.copy2(control_checkpoint,run/'control_model.pt')
    print('TRAIN_DIVERSITY_CROPS',len(candidate_train),dict(counts),flush=True)
    checkpoint=train_head(dataset,run,'training',run/'frozen_baseline/model.pt')
    dump(run/'progress.json',{'phase':'training_complete'})
    base_cache=np.load(source/'crop_features.npz')
    base_p,base_e=base_cache['probabilities'],base_cache['embeddings']
    control_cache=np.load(control_run/'crop_features.npz') if comparable else None
    cp,ce=(control_cache['probabilities'],control_cache['embeddings']) if comparable else (None,None)
    arms={'baseline':run/'frozen_baseline/model.pt','control':run/'control_model.pt','diversity':checkpoint}
    for name,weights in arms.items():
        encoder=NinePredictor(weights,224)
        probabilities=np.full((len(audit),9),np.nan)
        embeddings=np.full((len(audit),base_e.shape[1]),np.nan,dtype=np.float32)
        if name=='baseline':probabilities[:len(old)],embeddings[:len(old)]=base_p,base_e
        if name=='control' and comparable:probabilities[:len(old)],embeddings[:len(old)]=cp,ce
        for i,r in enumerate(audit):
            if r['excluded_demo_group'] or not r['box']:continue
            if np.isfinite(probabilities[i]).all():continue
            image=cv2.imdecode(np.fromfile(str(ROOT/r['crop_path']),np.uint8),cv2.IMREAD_COLOR)
            probabilities[i],embeddings[i]=encoder.raw(image)
            if (i+1)%400==0:print('Encoded',name,i+1,'/',len(audit),flush=True)
        np.savez_compressed(run/(name+'_features.npz'),probabilities=probabilities,embeddings=embeddings)
        dump(run/'progress.json',{'phase':'features','completed_arm':name})
    dump(run/'progress.json',{'phase':'features_complete'})
    print('DIVERSITY_FEATURES_COMPLETE',run,flush=True)


if __name__=='__main__':main()
