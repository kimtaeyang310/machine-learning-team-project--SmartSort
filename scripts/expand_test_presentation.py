"""Expand the selected-test demo: ripe tomato replacement, two extra examples."""
from pathlib import Path
import json
import math
import shutil
import cv2
import numpy as np
import joblib
from src.joint import ROOT,JointClassifier
from src.joint_core import CLASSES,dump,sha
from src.rejection import unfamiliarity_scores,acceptance_mask
from src.video import detect_regions,crop_region
from src.paths import project_path,project_relative


def main():
    playback=ROOT/'outputs/video_experiment/replaced_mandarin/run_config.json'
    previous=json.loads(playback.read_text(encoding='utf-8'))
    model_dir=project_path(previous['model_dir']);run=model_dir.parent
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    data=np.load(run/'diversity_features.npz');p,e=data['probabilities'],data['embeddings']
    policy=joblib.load(model_dir/'policy.joblib')
    classifier=JointClassifier(model_dir)
    out=ROOT/'outputs/video_experiment/selected_test_presentation';out.mkdir(parents=True,exist_ok=True)
    selected={};trials={}
    requests={5:('tomato','ripe'),7:('banana','unripe'),8:('mandarin','overripe')}
    for slot,(fruit,stage) in requests.items():
        ids=[i for i,r in enumerate(audit) if r['phase']=='test' and r['fruit']==fruit and r['ripeness']==stage and not r['excluded_demo_group'] and np.isfinite(p[i]).all()]
        scores=unfamiliarity_scores(p[ids],e[ids],policy['reference'])
        accepted=acceptance_mask(p[ids],scores,policy,CLASSES)
        candidates=[(i,float(scores['embedding'][j])) for j,i in enumerate(ids) if accepted[j] and CLASSES[int(p[i].argmax())]==fruit+'_'+stage]
        def rank(pair):
            i,distance=pair;r=audit[i];family=r.get('source_family')
            original=family is None or Path(r['path']).name==family
            return (not original,r.get('proposal_count',1)!=1,distance,-float(p[i].max()))
        trials[slot]=[]
        for i,distance in sorted(candidates,key=rank)[:30]:
            r=audit[i]
            image=cv2.imdecode(np.fromfile(str(ROOT/r['path']),np.uint8),cv2.IMREAD_COLOR)
            h,w=image.shape[:2];scale=320/max(h,w)
            card=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR)
            frame=np.full((540,960,3),255,np.uint8);x,y=(960-card.shape[1])//2,(540-card.shape[0])//2
            frame[y:y+card.shape[0],x:x+card.shape[1]]=card
            probe=out/f'probe_slot_{slot}.mp4'
            writer=cv2.VideoWriter(str(probe),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,540))
            if not writer.isOpened():raise RuntimeError('Cannot open probe video')
            for _ in range(15):writer.write(frame)
            writer.release();cap=cv2.VideoCapture(str(probe));ok,decoded=cap.read();cap.release()
            if not ok:raise ValueError('Cannot decode probe')
            boxes=detect_regions(decoded)
            box=max(boxes,key=lambda b:b[2]*b[3]) if boxes else None
            result=classifier.predict_frame(crop_region(decoded,box)) if box else {'status':'failed'}
            trials[slot].append({'path':r['path'],'test_crop_distance':distance,'encoded_prediction':result})
            if result.get('status')=='ok' and result.get('fruit')==fruit and result.get('ripeness')==stage:
                selected[slot]=r;break
        if slot not in selected:raise ValueError(f'No encoded accepted test example for {fruit}/{stage}')
        print('Selected slot',slot,selected[slot]['path'],flush=True)
    images=out/'images';images.mkdir(exist_ok=True)
    old_images=ROOT/'outputs/video_experiment/replaced_mandarin/images'
    for slot in (1,2,3,4,6):
        options=[p for p in old_images.iterdir() if p.stem==str(slot)]
        if len(options)!=1:raise ValueError('Ambiguous existing demonstration image')
        shutil.copy2(options[0],images/options[0].name)
    for slot,r in selected.items():
        path=ROOT/r['path'];target=images/(str(slot)+path.suffix.lower())
        shutil.copy2(path,target)
    paths=sorted([p for p in images.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}],key=lambda p:int(p.stem))
    if len(paths)!=8:raise ValueError('Expected exactly eight presentation photos')
    cards=[]
    for path in paths:
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2];scale=320/max(h,w)
        cards.append(cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR))
    fps,width,height,side,speed,interval=30,960,540,320,240,2.
    frames=math.ceil(((len(cards)-1)*interval+(width+side)/speed)*fps)+1
    video=out/'fruit_conveyor_selected_tests.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot open presentation writer')
    try:
        for n in range(frames):
            canvas=np.full((height,width,3),255,np.uint8)
            for j,card in enumerate(cards):
                x=round(-side+speed*(n/fps-j*interval));h,w=card.shape[:2];y=(height-h)//2
                left,right=max(0,x),min(width,x+w)
                if right>left:canvas[y:y+h,left:right]=card[:,left-x:right-x]
            writer.write(canvas)
    finally:writer.release()
    dump(out/'selection_manifest.json',{'model_dir':str(model_dir),'model_sha256':classifier.sha,
        'selected':{str(slot):r for slot,r in selected.items()},'trials':trials,
        'order':['mandarin/ripe','grape/unknown','tomato/unripe','banana/ripe','tomato/ripe','lemon/unknown','banana/unripe','mandarin/overripe'],
        'frames':frames,'fps':fps,'duration_seconds':frames/fps,
        'source_hashes':{str(slot):sha(ROOT/r['path']) for slot,r in selected.items()},
        'note':'Successful test examples chosen for presentation; not unbiased accuracy evaluation. Source images and test membership unchanged.'})
    shutil.copy2(playback,out/'previous_run_config.json')
    config={'source':project_relative(video),'model_dir':project_relative(model_dir),'output':project_relative(out/'fruit_conveyor_selected_tests_tracked.mp4')}
    dump(out/'run_config.json',config)
    # Update the existing presentation menu entry to this expanded, matched pair.
    dump(playback,config)
    print('Presentation raw video',video,'frames',frames,flush=True)


if __name__=='__main__':main()
