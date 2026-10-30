"""Replace demo slot 1 with a correctly accepted held-out ripe mandarin."""
from pathlib import Path
from collections import Counter
import copy
import json
import shutil
import math
import cv2
import numpy as np
import joblib
from src.joint import ROOT,JointClassifier
from src.joint_core import dump,CLASSES,sha
from src.rejection import unfamiliarity_scores,acceptance_mask
from src.video import detect_regions,crop_region
from src.paths import project_relative


def main():
    run=Path((ROOT/'outputs/mandarin_diversity/latest_run.txt').read_text(encoding='utf-8'))
    model_dir=run/'diversity'
    audit=json.loads((run/'audit.json').read_text(encoding='utf-8'))
    data=np.load(run/'diversity_features.npz');p,e=data['probabilities'],data['embeddings']
    policy=joblib.load(model_dir/'policy.joblib')
    ids=[i for i,r in enumerate(audit) if r['phase']=='test' and r['fruit']=='mandarin' and r['ripeness']=='ripe' and not r['excluded_demo_group'] and np.isfinite(p[i]).all()]
    scores=unfamiliarity_scores(p[ids],e[ids],policy['reference'])
    mask=acceptance_mask(p[ids],scores,policy,CLASSES)
    accepted=[(i,float(scores['embedding'][j])) for j,i in enumerate(ids) if mask[j] and CLASSES[int(p[i].argmax())]=='mandarin_ripe']
    if not accepted:raise ValueError('No correctly accepted ripe mandarin test photo')
    def rank(pair):
        i,score=pair;r=audit[i]
        original=Path(r['path']).name==r.get('source_family')
        return (not original,r.get('proposal_count',1)!=1,score,-float(p[i].max()))
    ordered=sorted(accepted,key=rank)
    classifier=JointClassifier(model_dir)
    # Probe a centered, actually encoded frame for several selected candidates.
    # This is explicitly presentation selection, not a new accuracy evaluation.
    out=ROOT/'outputs/video_experiment/replaced_mandarin'
    out.mkdir(parents=True,exist_ok=True)
    trials=[];selected=None
    for i,score in ordered[:20]:
        r=audit[i]
        image=cv2.imdecode(np.fromfile(str(ROOT/r['path']),np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2];scale=320/max(h,w)
        resized=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR)
        frame=np.full((540,960,3),255,np.uint8)
        x,y=(960-resized.shape[1])//2,(540-resized.shape[0])//2
        frame[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        probe=out/'codec_probe.mp4'
        writer=cv2.VideoWriter(str(probe),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,540))
        if not writer.isOpened():raise RuntimeError('Cannot open probe writer')
        for _ in range(15):writer.write(frame)
        writer.release();cap=cv2.VideoCapture(str(probe));ok,decoded=cap.read();cap.release()
        if not ok:raise ValueError('Cannot read encoded probe')
        boxes=detect_regions(decoded)
        box=max(boxes,key=lambda b:b[2]*b[3]) if boxes else None
        prediction=classifier.predict_frame(crop_region(decoded,box)) if box else {'status':'failed'}
        trials.append({'path':r['path'],'test_crop_score':score,'encoded_probe_prediction':prediction})
        if prediction.get('status')=='ok' and prediction.get('fruit')=='mandarin' and prediction.get('ripeness')=='ripe':
            selected=r;break
    if selected is None:raise ValueError('No candidate passed the encoded centered probe')
    images=out/'images';images.mkdir(exist_ok=True)
    # Never alter the original M10 or the input folder used by earlier experiments.
    target=images/('1'+Path(selected['path']).suffix.lower())
    shutil.copy2(ROOT/selected['path'],target)
    for path in (ROOT/'data/predict_list_1').iterdir():
        if path.is_file() and path.stem in {'2','3','4','5','6'}:shutil.copy2(path,images/path.name)
    paths=sorted((p for p in images.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}),key=lambda p:int(p.stem))
    if len(paths)!=6:raise ValueError('Replacement folder must contain exactly six photos')
    cards=[]
    for path in paths:
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2];scale=320/max(h,w)
        cards.append(cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR))
    video=out/'fruit_conveyor_replaced.mp4'
    fps,speed,interval,width,height,side=30,240,2.,960,540,320
    travel=(width+side)/speed
    total=math.ceil(((len(cards)-1)*interval+travel)*fps)+1
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('Cannot open replacement video')
    try:
        for n in range(total):
            canvas=np.full((height,width,3),255,np.uint8)
            for j,card in enumerate(cards):
                x=round(-side+speed*(n/fps-j*interval));h,w=card.shape[:2];y=(height-h)//2
                left,right=max(0,x),min(width,x+w)
                if right>left:canvas[y:y+h,left:right]=card[:,left-x:right-x]
            writer.write(canvas)
    finally:writer.release()
    manifest={'replacement_of':'data/predict_list_1/1.jpg (M10), preserved',
        'selected_source':selected,'copied_to':str(target),'source_sha256':sha(ROOT/selected['path']),
        'model_dir':str(model_dir),'model_sha256':classifier.sha,'test_candidates':len(ids),'accepted_correct_candidates':len(accepted),
        'selection':'prefer original non-augmented test photo, one proposal, lower rejection distance; verify MP4-encoded center frame',
        'trials':trials,'raw_video':str(video),'frames':total,
        'note':'Explicitly selected successful test example for presentation; not an unbiased accuracy estimate. Test membership and training labels are unchanged.'}
    dump(out/'replacement_manifest.json',manifest)
    dump(out/'run_config.json',{'source':project_relative(video),'model_dir':project_relative(model_dir),'output':project_relative(out/'fruit_conveyor_replaced_tracked.mp4')})
    print('Selected source',selected['path'],flush=True)
    print('Source group',selected['group'],'phase',selected['phase'],flush=True)
    print('Verified probe',trials[-1]['encoded_probe_prediction']['fruit'],trials[-1]['encoded_probe_prediction']['ripeness'],flush=True)
    print('Replacement folder',out,flush=True)


if __name__=='__main__':main()
