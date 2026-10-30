"""Train kind-3 + three independent stage-3 models on the fixed crop split."""
# 종류 3클래스 모델 하나와 과일별 숙도 3클래스 모델 세 개를 학습합니다.
# 후속 정책 선택에 사용할 종류 확률·숙도 확률·특징 벡터도 저장합니다.
from pathlib import Path
from collections import Counter
import json
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.joint import ROOT,stamp
from src.joint_labels import refresh_labels
from src.joint_core import dump,sha
from src.two_stage import ClassifierHead,FRUITS,STAGES


def prepare_head_datasets(audit, run):
    """종류·숙도 모델별 학습 폴더를 만들고 클래스별 사진 수를 반환합니다."""
    heads={'kind':FRUITS,**{fruit+'_stage':STAGES for fruit in FRUITS}}
    counts={name:Counter() for name in heads}
    for name,labels in heads.items():
        for split in ('train','val'):
            for label in labels:
                (run/'datasets'/name/split/label).mkdir(parents=True)
        seen=set()
        for r in audit:
            if r['phase']!='train' or r['excluded_demo_group'] or not r['box'] or r['fruit'] not in FRUITS:
                continue
            if name!='kind' and name!=r['fruit']+'_stage':
                continue
            label=r['fruit'] if name=='kind' else r['ripeness']
            path=ROOT/r['crop_path']
            shutil.copy2(path,run/'datasets'/name/'train'/label/path.name)
            counts[name][label]+=1
            if label not in seen:
                shutil.copy2(path,run/'datasets'/name/'val'/label/path.name);seen.add(label)
        if seen!=set(labels):
            raise ValueError(f'Missing training class: {name}')
    return heads, counts


def main():
    run=ROOT/'outputs/two_stage_experiment'/('run_'+stamp())
    run.mkdir(parents=True)
    current=ROOT/'models/joint_nine_crop'
    baseline=json.loads((current/'bundle.json').read_text(encoding='utf-8'))
    source=Path(baseline['source'])
    audit=json.loads((source/'audit.json').read_text(encoding='utf-8'))
    audit,changes=refresh_labels(audit,ROOT/'data/metadata/fruit_labels.csv')
    groups={}
    for r in audit:
        groups.setdefault(r['group'],set()).add(r['phase'])
        if sha(ROOT/r['path'])!=r['sha256']:
            raise ValueError(f"Source changed: {r['path']}")
    if any(len(v)>1 for v in groups.values()):
        raise ValueError('Cross-phase group leakage')
    dump(run/'audit.json',audit)
    dump(run/'label_changes.json',changes)
    heads, counts = prepare_head_datasets(audit, run)
    config={'epochs_per_head':6,'imgsz':224,'batch':16,'device':'cpu','seed':42,
        'architecture':'kind-3 YOLO followed by a separate stage-3 YOLO for the selected fruit',
        'initial_checkpoint':str(current/'model.pt'),'initial_sha256':baseline['sha256']['model.pt'],
        'training_counts':{name:dict(c) for name,c in counts.items()},
        'selection':'fixed last.pt at 6 epochs; train-only loader val is ignored; no test-based model selection',
        'demo_used_for_training_or_selection':False,'baseline_cache':str(source/'crop_features.npz'),
        'current_default_bundle':baseline,'crop':'unchanged 18% padding',
        'caveat':'Previously inspected data; same-source overripe near-duplicate concerns remain; no ripe/unripe mandarin test after split change'}
    dump(run/'config.json',config)
    (run.parent/'latest_run.txt').write_text(str(run),encoding='utf-8')
    torch.set_num_threads(4)
    for name in heads:
        print('TRAINING_HEAD',name,dict(counts[name]),flush=True)
        model=YOLO(str(current/'model.pt'))
        model.train(data=str(run/'datasets'/name),project=str(run),name=name,epochs=6,imgsz=224,batch=16,
            device='cpu',workers=0,seed=42,deterministic=True,patience=0,val=False,plots=False,save=True,exist_ok=True,
            auto_augment=None,erasing=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,verbose=False)
        del model
        dump(run/'progress.json',{'completed_head':name,'phase':'training'})
    kind=ClassifierHead(run/'kind/weights/last.pt',FRUITS,embedding=True)
    p=np.full((len(audit),3),np.nan)
    stage=np.full_like(p,np.nan)
    embeddings=None
    for i,r in enumerate(audit):
        if r['excluded_demo_group'] or not r['box']:
            continue
        frame=cv2.imdecode(np.fromfile(str(ROOT/r['crop_path']),np.uint8),cv2.IMREAD_COLOR)
        probabilities,e=kind.raw(frame)
        if embeddings is None:
            embeddings=np.full((len(audit),len(e)),np.nan,dtype=np.float32)
        p[i],embeddings[i]=probabilities,e
        if (i+1)%400==0:
            print('Encoded kinds',i+1,flush=True)
    valid=np.isfinite(p).all(axis=1)
    for fruit in FRUITS:
        model=ClassifierHead(run/(fruit+'_stage')/'weights/last.pt',STAGES)
        ids=[i for i,r in enumerate(audit) if valid[i] and FRUITS[int(p[i].argmax())]==fruit]
        for i in ids:
            frame=cv2.imdecode(np.fromfile(str(ROOT/audit[i]['crop_path']),np.uint8),cv2.IMREAD_COLOR)
            stage[i],_=model.raw(frame)
        print('Encoded routed stages',fruit,len(ids),flush=True)
    np.savez_compressed(run/'features.npz',kind_probabilities=p,embeddings=embeddings,stage_probabilities=stage)
    dump(run/'progress.json',{'phase':'features_complete'})
    print('TWO_STAGE_RUN',str(run),flush=True)


if __name__=='__main__':
    main()
