"""Register all new citrus as user-defined ripe; group source/augmentation families."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json
import re
import shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw
from src.joint import ROOT,stamp
from src.joint_core import dump,sha
from src.joint_labels import refresh_labels
from src.video import detect_regions,crop_region,load_font
from scripts.calibrate_crops import prepare_frame


def canonical_name(name):
    match=re.search(r'Screen Shot .+',name)
    return match.group(0) if match else name


def descriptor(image):
    small=cv2.resize(image,(64,64),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    tiny=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    return small.astype(float),(tiny[:,1:]>tiny[:,:-1]).ravel()


def main():
    source=ROOT/'data/raw/1029mandarin'
    paths=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    if not paths:raise ValueError('No new images')
    run=ROOT/'outputs/mandarin_diversity'/('run_'+stamp());run.mkdir(parents=True)
    families=defaultdict(list)
    for path in paths:families[canonical_name(path.name)].append(path)
    names=sorted(families)
    parent=list(range(len(names)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    reps={};descs={}
    for i,name in enumerate(names):
        options=sorted(families[name],key=lambda p:(0 if p.name==name else 1 if p.name.startswith('vertical_flip_') else 2,len(p.name),p.name))
        path=options[0]
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Unreadable family representative: {path}')
        if path.name.startswith('vertical_flip_'):image=cv2.flip(image,0)
        reps[name]=(path,image);descs[name]=descriptor(image)
    near=[]
    for i,a in enumerate(names):
        sa,ha=descs[a]
        for j in range(i+1,len(names)):
            b=names[j];sb,hb=descs[b]
            hamming=int(np.count_nonzero(ha!=hb))
            if hamming>4:continue
            mae=float(np.abs(sa-sb).mean())
            if mae<=6.:
                union(i,j);near.append({'a':a,'b':b,'hamming':hamming,'rgb_mae':mae})
    clusters=defaultdict(list)
    for i,name in enumerate(names):clusters[find(i)].append(name)
    ordered=sorted(clusters.values(),key=lambda v:v[0])
    rng=np.random.default_rng(42)
    permutation=rng.permutation(len(ordered))
    n=len(ordered);train_end=round(n*.60);cal_end=train_end+round(n*.15);selection_end=cal_end+round(n*.15)
    phases={}
    for rank,i in enumerate(permutation):
        phases[int(i)]='train' if rank<train_end else 'calibration' if rank<cal_end else 'selection' if rank<selection_end else 'test'
    mapping={};csv_new=[];group_records=[]
    for i,cluster in enumerate(ordered):
        token=hashlib.sha256('\n'.join(cluster).encode()).hexdigest()[:12]
        fruit_id='N1029_'+token
        phase=phases[i]
        group_id=fruit_id+'_20261029'
        csv_new.append({'fruit':'mandarin','fruit_id':fruit_id,'date':'2026-10-29','ripeness':'ripe',
                        'group_id':group_id,'split':'test' if phase=='test' else 'train','phase':phase})
        group_records.append({'fruit_id':fruit_id,'group':'mandarin:'+fruit_id,'phase':phase,
            'source_families':cluster,'date_semantics':'2026-10-29 is registration date; screenshot timestamp is not verified capture date'})
        for name in cluster:mapping[name]=(fruit_id,group_id,phase)
    # An exact pixel/SHA duplicate with old data must not silently cross splits.
    baseline_dir=ROOT/'models/joint_nine_crop'
    bundle=json.loads((baseline_dir/'bundle.json').read_text(encoding='utf-8'))
    previous=Path(bundle['source'])
    old_audit=json.loads((previous/'audit.json').read_text(encoding='utf-8'))
    old_hashes={r['sha256']:r for r in old_audit}
    additions=[];crop_dir=run/'crops';crop_dir.mkdir()
    raw_duplicate=[];failed=[]
    for index,path in enumerate(paths):
        fruit_id,source_group,phase=mapping[canonical_name(path.name)]
        digest=sha(path)
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Unreadable new image: {path}')
        frame=prepare_frame(image);boxes=detect_regions(frame)
        box=max(boxes,key=lambda b:b[2]*b[3]) if boxes else None
        row={'path':path.relative_to(ROOT).as_posix(),'fruit':'mandarin','ripeness':'ripe','group':'mandarin:'+fruit_id,
            'source_group':source_group,'phase':phase,'sha256':digest,'box':list(box) if box else None,
            'proposal_count':len(boxes),'excluded_demo_group':False,'new_1029':True,
            'source_family':canonical_name(path.name),'registration_date':'2026-10-29','label_origin':'user instruction: all ripe',
            'capture_date_verified':False,'active':True}
        if digest in old_hashes:
            row['active']=False;row['excluded_demo_group']=True
            raw_duplicate.append({'new_path':row['path'],'old_path':old_hashes[digest]['path']})
        if box:
            crop=crop_region(frame,box)
            target=crop_dir/(hashlib.sha256(row['path'].encode()).hexdigest()[:24]+'.png')
            cv2.imencode('.png',crop)[1].tofile(str(target))
            row['crop_path']=target.relative_to(ROOT).as_posix()
        else:failed.append(row['path'])
        additions.append(row)
        if (index+1)%200==0:print('Prepared new images',index+1,flush=True)
    # If a family overlaps old files, quarantine the entire family to avoid leakage.
    excluded={r['group'] for r in additions if not r['active']}
    for r in additions:
        if r['group'] in excluded:r.update(active=False,excluded_demo_group=True)
    labels=ROOT/'data/metadata/fruit_labels.csv'
    shutil.copy2(labels,run/'fruit_labels.before.csv')
    rows=list(csv.DictReader(labels.open(encoding='utf-8-sig')))
    existing={r['group_id']:r for r in rows if r['fruit']=='mandarin'}
    for r in csv_new:
        if r['group_id'] in existing:
            if any(existing[r['group_id']].get(k,'')!=str(v) for k,v in r.items()):
                raise ValueError('Existing new-source label conflicts; refusing to overwrite')
        else:rows.append(r)
    fields=['fruit','fruit_id','date','ripeness','group_id','split','phase']
    with (run/'fruit_labels.updated.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    manifest_base,changes=refresh_labels(old_audit,run/'fruit_labels.updated.csv')
    audit=manifest_base+additions
    by_group=defaultdict(set)
    for r in audit:by_group[r['group']].add(r['phase'])
    if any(len(v)>1 for v in by_group.values()):raise ValueError('Group crosses phases')
    dump(run/'audit.json',audit)
    dump(run/'new_sources.json',group_records)
    dump(run/'near_duplicate_families.json',near)
    dump(run/'exact_old_overlap.json',raw_duplicate)
    dump(run/'no_crop_proposals.json',failed)
    dump(run/'new_manifest.json',additions)
    # Registered metadata persists the new files for future main-menu training.
    registry=ROOT/'data/metadata/mandarin_1029_manifest.json'
    if registry.exists():shutil.copy2(registry,run/'previous_registered_manifest.json')
    shutil.copy2(run/'new_manifest.json',registry)
    shutil.copy2(run/'fruit_labels.updated.csv',labels)
    summary={'images':len(paths),'source_filename_families':len(families),'conservative_source_clusters':len(ordered),
        'near_duplicate_family_pairs':len(near),'no_crop_proposals':len(failed),
        'quarantined_exact_old_overlap_groups':len(excluded),
        'new_images_by_phase':dict(Counter(r['phase'] for r in additions)),
        'new_source_clusters_by_phase':dict(Counter(r['phase'] for r in group_records)),
        'usable_new_crops_by_phase':dict(Counter(r['phase'] for r in additions if r['box'] and r['active'])),
        'labels':f'All {len(paths)} assigned mandarin/ripe by user instruction; no maturity re-inference',
        'grouping_limit':'Source-name families and conservative visual grouping are not verified physical fruit IDs; true stock-image origins remain unverified',
        'baseline_feature_source':str(previous),'model_source':str(baseline_dir),'seed':42}
    dump(run/'registration.json',summary)
    # Preview proposed merged source families for human review.
    if near:
        sheet=Image.new('RGB',(1000,220*min(8,len(near))),'white');draw=ImageDraw.Draw(sheet)
        for i,pair in enumerate(near[:8]):
            for j,key in enumerate(('a','b')):
                image=Image.fromarray(cv2.cvtColor(reps[pair[key]][1],cv2.COLOR_BGR2RGB));image.thumbnail((220,180))
                sheet.paste(image,(j*500,i*220));draw.text((j*500+230,i*220+30),pair[key][-25:],font=load_font(14),fill='black')
        sheet.save(run/'merged_source_preview.jpg')
    (run.parent/'latest_run.txt').write_text(str(run),encoding='utf-8')
    print('DIVERSITY_RUN',str(run),flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
