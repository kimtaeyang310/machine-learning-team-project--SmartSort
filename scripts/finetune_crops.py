"""Fixed six-epoch crop fine-tuning; original bundle is never overwritten."""
# 잘라낸 과일 사진에 맞춰 6 epoch 동안 추가 학습합니다.
# epoch는 학습 자료를 한 차례 순회하는 단위이며 결과는 원본 모델과 별도로 보관합니다.
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.joint import ROOT, MODEL_DIR
from src.joint_core import NinePredictor, FRUITS, dump, CLASSES
from scripts.calibrate_crops import prepare_frame
from src.video import detect_regions, crop_region


def main():
    run = Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8'))
    audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    # The initial 8/255 screen matched different fruit on the same gray backdrop.
    # Tighten image-identity matching to 2/255 mean RGB error, and exclude entire
    # groups (including one near-duplicate banana). This uses pixels, not labels.
    excluded = {r['group'] for r in audit if r['demo_distance'] < 2.0}
    dump(run/'duplicate_screen_revision.json',{'previous_mean_rgb_error':8.0,'final_mean_rgb_error':2.0,
        'reason':'8/255 confounded similar gray backgrounds with image identity; exact resized copies are below 0.15/255, retained near duplicate banana at 1.78/255',
        'excluded_groups':sorted(excluded)})
    for r in audit:
        r['excluded_demo_group'] = r['group'] in excluded
    dump(run/'audit.json',audit)
    dataset = run/'finetune_dataset'
    crops_dir = run/'crops'
    crops_dir.mkdir(exist_ok=True)
    for split in ('train','val'):
        for label in CLASSES:
            (dataset/split/label).mkdir(parents=True,exist_ok=True)
    seen = set()
    counts = {}
    for index,r in enumerate(audit):
        if r['excluded_demo_group'] or not r['box']:
            continue
        image = cv2.imdecode(np.fromfile(str(ROOT/r['path']),np.uint8),cv2.IMREAD_COLOR)
        crop = crop_region(prepare_frame(image),r['box'])
        name = hashlib.sha256(r['path'].encode()).hexdigest()[:24]+'.png'
        target = crops_dir/name
        cv2.imencode('.png',crop)[1].tofile(str(target))
        r['crop_path'] = str(target.relative_to(ROOT)).replace('\\','/')
        if r['phase']=='train' and r['fruit'] in FRUITS:
            label = r['fruit']+'_'+r['ripeness']
            payload = target.read_bytes()
            (dataset/'train'/label/name).write_bytes(payload)
            counts[label] = counts.get(label,0)+1
            if label not in seen:
                (dataset/'val'/label/name).write_bytes(payload)
                seen.add(label)
        if (index+1)%500 == 0:
            print('Prepared crops',index+1,flush=True)
    if seen != set(CLASSES):
        raise ValueError('Missing training classes')
    dump(run/'audit.json',audit)
    config = {'epochs':6,'imgsz':224,'batch':16,'device':'cpu','seed':42,
              'initial_checkpoint':str(MODEL_DIR/'model.pt'),'training_counts':counts,
              'selection':'fixed last.pt; no test-based early stopping; train-only loader val is ignored'}
    dump(run/'finetune_config.json',config)
    torch.set_num_threads(4)
    checkpoint = run/'finetune/weights/last.pt'
    if not checkpoint.is_file():
        model = YOLO(str(MODEL_DIR/'model.pt'))
        model.train(data=str(dataset),project=str(run),name='finetune',epochs=6,imgsz=224,
            batch=16,device='cpu',workers=0,seed=42,deterministic=True,patience=0,
            val=False,plots=False,save=True,exist_ok=True,auto_augment=None,erasing=0.,
            hsv_h=0.,hsv_s=0.,hsv_v=0.,verbose=False)
    # Preserve the frozen original-model feature cache for baseline comparisons.
    original_cache = run/'original_crop_features.npz'
    if not original_cache.exists():
        (run/'crop_features.npz').rename(original_cache)
    original = np.load(original_cache)
    old_p,old_e = original['probabilities'].copy(),original['embeddings'].copy()
    model = NinePredictor(checkpoint,224)
    old_model = NinePredictor(MODEL_DIR/'model.pt',224)
    probabilities = np.full_like(old_p,np.nan)
    embeddings = np.full_like(old_e,np.nan)
    for i,r in enumerate(audit):
        if 'crop_path' not in r:
            continue
        image = cv2.imdecode(np.fromfile(str(ROOT/r['crop_path']),np.uint8),cv2.IMREAD_COLOR)
        p,e = model.raw(image)
        probabilities[i],embeddings[i] = p,e
        if not np.isfinite(old_p[i]).all():
            p,e = old_model.raw(image)
            old_p[i],old_e[i] = p,e
        if (i+1)%200 == 0:
            print('Fine-tuned features',i+1,flush=True)
    np.savez_compressed(original_cache,probabilities=old_p,embeddings=old_e)
    np.savez_compressed(run/'crop_features.npz',probabilities=probabilities,embeddings=embeddings)
    print('FINETUNE_COMPLETE',str(checkpoint),flush=True)


if __name__ == '__main__':
    main()
