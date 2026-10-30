"""Read-only mandarin dataset/label audit and appearance review artifacts."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from src.joint import ROOT,JointClassifier,stamp
from src.joint_core import dump,CLASSES
from src.video import detect_regions,crop_region,load_font
from scripts.calibrate_crops import prepare_frame


def color_signature(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,45,20]),np.array([179,255,255]))
    histogram=cv2.calcHist([hsv],[0,1],mask,[18,8],[0,180,0,256]).ravel()
    histogram=histogram/max(float(np.linalg.norm(histogram)),1e-12)
    hues=hsv[:,:,0][mask>0]
    values=hsv[:,:,2][mask>0]
    return histogram, {'foreground_pixels':int(len(hues)),
        'green_fraction':float(np.mean((hues>=35)&(hues<85))) if len(hues) else None,
        'yellow_fraction':float(np.mean((hues>=20)&(hues<35))) if len(hues) else None,
        'orange_red_fraction':float(np.mean((hues<20)|(hues>=170))) if len(hues) else None,
        'median_brightness':float(np.median(values)) if len(values) else None}


def main():
    run=ROOT/'outputs/mandarin_label_audit'/('run_'+stamp())
    run.mkdir(parents=True)
    label_path=ROOT/'data/metadata/fruit_labels.csv'
    label_hash=hashlib.sha256(label_path.read_bytes()).hexdigest()
    csv_rows=list(csv.DictReader(label_path.open(encoding='utf-8-sig')))
    lookup=defaultdict(list)
    for row in csv_rows:
        if row['fruit']=='mandarin':
            lookup[row['group_id']].append(row)
    manifest=json.loads((ROOT/'models/joint_nine/split_manifest.json').read_text(encoding='utf-8'))
    mapping={s['path']:s for s in manifest if s['fruit']=='mandarin'}
    crop_run=Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    previous_audit=json.loads((crop_run/'audit.json').read_text(encoding='utf-8'))
    target=cv2.imdecode(np.fromfile(str(ROOT/'data/predict_list_1/1.jpg'),np.uint8),cv2.IMREAD_COLOR)
    tf=prepare_frame(target)
    tb=max(detect_regions(tf),key=lambda b:b[2]*b[3])
    tc=crop_region(tf,tb)
    target_color,target_stats=color_signature(tc)
    target_small=cv2.resize(target,(32,32),interpolation=cv2.INTER_AREA).astype(float)
    images={}
    signatures=[]
    rows=[]
    for index,path in enumerate(sorted((ROOT/'data/raw').rglob('*'))):
        if not path.is_file() or 'mandarin' not in path.name.lower() or path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        relative=path.relative_to(ROOT).as_posix()
        parts=path.stem.split('_')
        date,fruit_id=parts[0],parts[2]
        key=fruit_id+'_'+date
        labels=lookup.get(key,[])
        sample=mapping.get(relative)
        cached_crop = next((r.get('crop_path') for r in previous_audit if r['path']==relative),None)
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR) if not cached_crop else None
        if cached_crop:
            crop=cv2.imdecode(np.fromfile(str(ROOT/cached_crop),np.uint8),cv2.IMREAD_COLOR)
            frame=crop
            boxes=[(0,0,crop.shape[1],crop.shape[0])]
            box=boxes[0]
        else:
            if image is None:
                raise ValueError(f'Cannot read {path}')
            frame=prepare_frame(image)
            boxes=detect_regions(frame)
            box=max(boxes,key=lambda b:b[2]*b[3]) if boxes else None
            crop=crop_region(frame,box) if box else frame
        color,stats=color_signature(crop)
        images[relative]=crop
        signatures.append(color)
        prior = next(r for r in previous_audit if r['path']==relative)
        identity_distance=prior['demo_distance'] if prior['nearest_demo']=='1.jpg' else 255.
        row={'path':relative,'fruit_id':fruit_id,'date':date,'source_group':key,
             'csv_label':labels[0]['ripeness'] if labels else None,'csv_split':labels[0]['split'] if labels else None,
             'csv_row_count':len(labels),'manifest_label':sample['ripeness'] if sample else None,
             'phase':sample['phase'] if sample else None,'group':sample['group'] if sample else 'mandarin:'+fruit_id,
             'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'source_hash_matches_manifest':sample is not None and hashlib.sha256(path.read_bytes()).hexdigest()==sample['sha256'],
             'color_similarity':float(color@target_color),
             'resized_identity_distance':identity_distance,'detected':bool(box),'color_stats':stats}
        rows.append(row)
        if len(rows)%100==0:
            print('Audited mandarin',len(rows),flush=True)
    same_source=min(rows,key=lambda r:r['resized_identity_distance'])
    target_group=same_source['group']
    for row in rows:
        row['same_target_individual']=row['group']==target_group
    independent=[r for r in rows if not r['same_target_individual']]
    def unique_top(key,count=20):
        result=[]; seen=set()
        for r in sorted(independent,key=lambda r:r[key],reverse=True):
            if r['group'] in seen:
                continue
            result.append(r);seen.add(r['group'])
            if len(result)==count:
                break
        return result
    color_top=unique_top('color_similarity')
    font=load_font(17)
    stages={'unripe':'미숙','ripe':'적숙','overripe':'과숙'}
    def gallery(selected,name,metric):
        sheet=Image.new('RGB',(1250,1250),'white');draw=ImageDraw.Draw(sheet)
        cards=[{'path':'시연 귤','fruit_id':same_source['fruit_id'],'date':same_source['date'],'csv_label':same_source['csv_label'],'phase':'진단 대상',metric:1.}]+selected[:19]
        for i,row in enumerate(cards):
            crop=tc if i==0 else images[row['path']]
            image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));image.thumbnail((225,220))
            x,y=(i%5)*250,(i//5)*312
            sheet.paste(image,(x+(250-image.width)//2,y+5))
            text=[f"{row['fruit_id']} / {row['date']}",f"라벨: {stages.get(row['csv_label'],row['csv_label'])} / {row['phase']}",f"유사도: {row[metric]:.3f}"]
            for j,line in enumerate(text):
                draw.text((x+7,y+234+j*23),line,font=font,fill='black')
        sheet.save(run/name)
    gallery(color_top,'similar_color_individuals.jpg','color_similarity')
    # Same-day M01-M10 cards, fixed view suffix when available; no relabeling.
    panel=[]
    for number in range(1,11):
        matching=[r for r in rows if r['fruit_id']==f'M{number:02d}' and r['date']=='20261008']
        if matching:
            row=min(matching,key=lambda r:(not r['path'].endswith('_indoor_01.jpg'),r['path']))
            panel.append(row)
    old_independent=independent
    independent=panel
    # Direct panel rendering, avoiding duplicate target in the comparison.
    sheet=Image.new('RGB',(1250,650),'white');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(panel):
        image=Image.fromarray(cv2.cvtColor(images[row['path']],cv2.COLOR_BGR2RGB));image.thumbnail((225,230))
        x,y=(i%5)*250,(i//5)*325
        sheet.paste(image,(x+(250-image.width)//2,y+5))
        for j,line in enumerate([row['fruit_id']+' / '+row['date'],'라벨: '+stages.get(row['csv_label'],'?'),str(row['phase'])]):
            draw.text((x+10,y+245+j*24),line,font=font,fill='black')
    sheet.save(run/'same_day_M01_M10.jpg')
    independent=old_independent
    # Detect label disagreements in the authoritative CSV vs deployed snapshot.
    disagreements=[r for r in rows if r['csv_label']!=r['manifest_label']]
    conflicts={k:v for k,v in lookup.items() if len({(r['ripeness'],r['split']) for r in v})>1}
    groups=defaultdict(set)
    for r in rows:
        groups[r['group']].add(r['phase'])
    phase_counts={phase:{stage:{'images':sum(r['phase']==phase and r['csv_label']==stage for r in rows),
                     'individuals':len({r['group'] for r in rows if r['phase']==phase and r['csv_label']==stage})}
                  for stage in ('unripe','ripe','overripe')} for phase in ('train','calibration','selection','test')}
    # Opposite-label closest color match per target individual: a review cue,
    # never proof that either maturity annotation is wrong.
    signature_matrix=np.asarray(signatures)
    similarities=signature_matrix@signature_matrix.T
    cues=[]
    for i,row in enumerate(rows):
        ids=[j for j,other in enumerate(rows) if other['group']!=row['group'] and other['csv_label']!=row['csv_label']]
        j=max(ids,key=lambda j:similarities[i,j])
        if similarities[i,j]>=.97:
            cues.append({'path':row['path'],'label':row['csv_label'],'other_path':rows[j]['path'],'other_label':rows[j]['csv_label'],'similarity':float(similarities[i,j]),'reason':'similar foreground color with different maturity label; visual review only'})
    dedup=[];seen=set()
    for cue in sorted(cues,key=lambda c:c['similarity'],reverse=True):
        pair=tuple(sorted([cue['path'],cue['other_path']]))
        if pair not in seen:
            dedup.append(cue);seen.add(pair)
    summary={'raw_images':len(rows),'manifest_images':len(mapping),'csv_rows':sum(len(v) for v in lookup.values()),
        'target_source':same_source,'target_color_stats':target_stats,
        'label_disagreements':len(disagreements),'conflicting_csv_keys':len(conflicts),
        'cross_phase_groups':{k:sorted(v) for k,v in groups.items() if len(v)>1},
        'missing_csv_labels':sum(r['csv_label'] is None for r in rows),
        'hash_mismatch_count':sum(not r['source_hash_matches_manifest'] for r in rows),
        'phase_counts':phase_counts,'similar_color_opposite_label_pairs':len(dedup),
        'interpretation':'Color/model similarity is not ground-truth maturity. Human review is required; no labels or model artifacts were changed.'}
    dump(run/'summary.json',summary)
    dump(run/'inventory.json',rows)
    dump(run/'nearest_color_individuals.json',color_top)
    dump(run/'label_review_candidates.json',dedup)
    dump(run/'label_disagreements.json',disagreements)
    fields=['path','fruit_id','date','csv_label','csv_split','phase','color_similarity','same_target_individual']
    with (run/'review_inventory.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    assert hashlib.sha256(label_path.read_bytes()).hexdigest()==label_hash
    (run.parent/'latest_run.txt').write_text(str(run),encoding='utf-8')
    print('AUDIT_RUN',run,flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
