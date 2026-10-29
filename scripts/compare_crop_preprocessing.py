"""Inspect the mandarin ROI and compare fixed preprocessing on selection only."""
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw
from src.joint import ROOT, JointClassifier, stamp
from src.joint_core import dump, CLASSES
from src.rejection import unfamiliarity_scores
from src.video import detect_regions, crop_region, load_font
from scripts.calibrate_crops import prepare_frame
from scripts.select_crop_policy import evaluate

VARIANTS = [
    {'name':'rect_18','padding':.18,'square':False,'pad_color':'edge'},
    {'name':'rect_35','padding':.35,'square':False,'pad_color':'edge'},
    {'name':'rect_65','padding':.65,'square':False,'pad_color':'edge'},
    {'name':'rect_100','padding':1.,'square':False,'pad_color':'edge'},
    {'name':'square_18_edge','padding':.18,'square':True,'pad_color':'edge'},
    {'name':'square_35_edge','padding':.35,'square':True,'pad_color':'edge'},
    {'name':'square_65_edge','padding':.65,'square':True,'pad_color':'edge'},
    {'name':'square_18_white','padding':.18,'square':True,'pad_color':'white'},
]


def apply_variant(frame,box,variant):
    return crop_region(frame,box,**{k:v for k,v in variant.items() if k!='name'})


def main():
    experiment=Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    run=ROOT/'outputs/crop_preprocessing'/('run_'+stamp())
    run.mkdir(parents=True)
    model=JointClassifier(ROOT/'models/joint_nine_crop')
    model_hash=hashlib.sha256((model.directory/'model.pt').read_bytes()).hexdigest()
    policy_hash=hashlib.sha256((model.directory/'policy.joblib').read_bytes()).hexdigest()
    audit=json.loads((experiment/'audit.json').read_text(encoding='utf-8'))
    # Fix candidates and selection rule before predicting the demonstration image.
    config={'variants':VARIANTS,'model_sha256':model_hash,'policy_sha256':policy_hash,
        'source_audit':str(experiment/'audit.json'),
        'rule':'Do not increase unknown false acceptance or decrease overall 9-class Macro F1; require higher mandarin Macro F1; then maximize mandarin F1, overall F1, minimize rejection.',
        'demo_used_for_selection':False,'test_used_for_selection':False}
    dump(run/'config.json',config)
    # Diagnostic image: raw source, detector overlay, actual current video crop.
    source=ROOT/'data/predict_list_1/1.jpg'
    raw=cv2.imdecode(np.fromfile(str(source),np.uint8),cv2.IMREAD_COLOR)
    frame=prepare_frame(raw)
    boxes=detect_regions(frame)
    box=max(boxes,key=lambda b:b[2]*b[3])
    overlay=frame.copy()
    x,y,w,h=box
    cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),(0,160,0),2)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,45,20]),np.array([179,255,255]))
    cap=cv2.VideoCapture(str(ROOT/'outputs/video_experiment/fruit_conveyor.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,90)
    ok,video_frame=cap.read()
    cap.release()
    if not ok:
        raise ValueError('Cannot read diagnostic video frame')
    video_boxes=detect_regions(video_frame)
    # Spatial choice only: this frame's rightmost candidate is the mandarin.
    video_box=max(video_boxes,key=lambda b:b[0])
    video_crop=crop_region(video_frame,video_box)
    gallery=[('원본 사진',raw),('검출 박스',overlay),('색상 마스크',cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)),('사진 크롭 (18%)',crop_region(frame,box)),('실제 영상 크롭 (18%)',video_crop)]
    sheet=Image.new('RGB',(1500,350),'white')
    draw=ImageDraw.Draw(sheet)
    font=load_font(18)
    for index,(label,image) in enumerate(gallery):
        cv2.imencode('.png',image)[1].tofile(str(run/f'diagnostic_{index}.png'))
        pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        pil.thumbnail((280,290))
        sheet.paste(pil,(index*300+(300-pil.width)//2,10))
        draw.text((index*300+10,315),label,font=font,fill='black')
    sheet.save(run/'mandarin_diagnostic.jpg')
    dump(run/'diagnostic.json',{'source':str(source),'normalized_size':list(frame.shape[:2]),'source_box':list(box),'video_frame':90,'video_box':list(video_box),'video_crop_size':list(video_crop.shape[:2]),'interpretation':'ROI review only; not ground-truth localization accuracy'})
    indices=[i for i,r in enumerate(audit) if r['phase']=='selection' and not r['excluded_demo_group']]
    # Decode each source once; all candidates see precisely the same detections.
    frames={}
    for count,i in enumerate(indices):
        raw=cv2.imdecode(np.fromfile(str(ROOT/audit[i]['path']),np.uint8),cv2.IMREAD_COLOR)
        frames[i]=prepare_frame(raw)
        if (count+1)%100==0:
            print('Prepared selection',count+1,flush=True)
    comparisons={}
    for variant in VARIANTS:
        probabilities=np.full((len(audit),len(CLASSES)),np.nan)
        embeddings=[]
        successful=[]
        for i in indices:
            if not audit[i]['box']:
                continue
            crop=apply_variant(frames[i],audit[i]['box'],variant)
            p,e=model.engine.raw(crop)
            probabilities[i]=p
            embeddings.append(e)
            successful.append(i)
        scores={'embedding':np.full(len(audit),np.inf)}
        values=unfamiliarity_scores(probabilities[successful],np.asarray(embeddings),model.engine.policy['reference'])['embedding']
        scores['embedding'][successful]=values
        comparisons[variant['name']]=evaluate(audit,probabilities,scores,model.engine.policy,'selection')
        np.savez_compressed(run/(variant['name']+'_selection.npz'),probabilities=probabilities,scores=scores['embedding'])
        dump(run/'selection_comparison.json',comparisons)
        print(variant['name'],json.dumps(comparisons[variant['name']],ensure_ascii=False),flush=True)
    baseline=comparisons['rect_18']
    eligible=[v for v in VARIANTS if comparisons[v['name']]['unknown_false_acceptance_rate']<=baseline['unknown_false_acceptance_rate'] and comparisons[v['name']]['nine_class_macro_f1']>=baseline['nine_class_macro_f1'] and comparisons[v['name']]['per_fruit']['mandarin']['macro_f1']>baseline['per_fruit']['mandarin']['macro_f1']]
    selected=max(eligible,key=lambda v:(comparisons[v['name']]['per_fruit']['mandarin']['macro_f1'],comparisons[v['name']]['nine_class_macro_f1'],-comparisons[v['name']]['known_rejection_or_missing_rate'])) if eligible else VARIANTS[0]
    dump(run/'decision.json',{'selected':selected,'changed':selected['name']!='rect_18','eligible':[v['name'] for v in eligible],'selection_rule':config['rule'],'demo_used_for_selection':False,'test_used_for_selection':False})
    # Only after freezing the decision, diagnose each variant on the demo.
    diagnostics=[]
    preview=Image.new('RGB',(1200,660),'white')
    draw=ImageDraw.Draw(preview)
    for index,variant in enumerate(VARIANTS):
        crop=apply_variant(video_frame,video_box,variant)
        result=model.predict_frame(crop)
        diagnostics.append({'variant':variant,'prediction':result,'crop_size':list(crop.shape[:2])})
        image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        image.thumbnail((280,235))
        ox,oy=(index%4)*300,(index//4)*330
        preview.paste(image,(ox+(300-image.width)//2,oy+5))
        lines=[variant['name'],result['candidate_class'],f"{result['status']} / 距離 {result['score']:.3f}"]
        for n,line in enumerate(lines):
            draw.text((ox+8,oy+246+n*25),line,font=font,fill='black')
    preview.save(run/'mandarin_variant_comparison.jpg')
    dump(run/'demo_diagnostics.json',diagnostics)
    assert hashlib.sha256((model.directory/'model.pt').read_bytes()).hexdigest()==model_hash
    assert hashlib.sha256((model.directory/'policy.joblib').read_bytes()).hexdigest()==policy_hash
    (run.parent/'latest_run.txt').write_text(str(run),encoding='utf-8')
    print('PREPROCESSING_RUN',str(run),flush=True)
    print('DECISION',selected,flush=True)


if __name__=='__main__':
    main()
