"""Crop-domain calibration experiment with group-separated selection and testing."""
# 사진을 영상과 비슷한 크기로 줄여 과일 영역을 잘라내고 특징을 캐시합니다.
# 시연 사진과 비슷한 그룹은 제외하고 특징 배열과 감사 기록을 후속 정책 선택에 사용합니다.
from pathlib import Path
from collections import Counter
import json
import hashlib
import cv2
import numpy as np
import joblib
from src.joint import ROOT, MODEL_DIR, stamp
from src.joint_core import NinePredictor, CLASSES, FRUITS, dump
from src.video import detect_regions, crop_region


# 긴 변을 320픽셀로 맞추면서 가로세로 비율을 유지합니다.
def prepare_frame(image):
    h,w = image.shape[:2]
    scale = 320 / max(h,w)
    return cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR)


def main():
    run = ROOT/'outputs/crop_calibration'/('run_'+stamp())
    run.mkdir(parents=True)
    samples = json.loads((MODEL_DIR/'split_manifest.json').read_text(encoding='utf-8'))
    demos = sorted((ROOT/'data/predict_list_1').iterdir())
    fingerprints = []
    for path in demos:
        image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        fingerprints.append(cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA).astype(float))
    audit, crops = [], []
    for i,s in enumerate(samples):
        path = ROOT/s['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != s['sha256']:
            raise ValueError(f'Changed source: {path}')
        image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable: {path}')
        fingerprint = cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA).astype(float)
        diffs = [float(np.abs(fingerprint-f).mean()) for f in fingerprints]
        frame = prepare_frame(image)
        boxes = detect_regions(frame)
        box = max(boxes,key=lambda b:b[2]*b[3]) if boxes else None
        crops.append(crop_region(frame,box) if box else None)
        audit.append({**s,'box':list(box) if box else None,'proposal_count':len(boxes),
                      'demo_distance':min(diffs),'nearest_demo':demos[int(np.argmin(diffs))].name})
        if (i+1)%500 == 0:
            print('Audited',i+1,flush=True)
    # Conservative near-duplicate screen; exclude whole matching groups from
    # adaptation and evaluation. The original weights may already know them.
    excluded = {r['group'] for r in audit if r['demo_distance'] < 2.0}
    groups = {}
    for s in samples:
        groups.setdefault(s['group'],set()).add(s['phase'])
    if any(len(phases)>1 for phases in groups.values()):
        raise ValueError('Cross-phase group leakage')
    for r in audit:
        r['excluded_demo_group'] = r['group'] in excluded
    dump(run/'audit.json',audit)
    print('Excluded demo groups',sorted(excluded),flush=True)
    print('Missing proposals',Counter((r['phase'],r['fruit']) for r in audit if not r['box']),flush=True)
    predictor = NinePredictor(MODEL_DIR/'model.pt',224)
    import torch
    torch.set_num_threads(4)
    # 검출하지 못하거나 제외된 사진은 NaN으로 남겨 평가에서 누락 원인을 구분합니다.
    probabilities = np.full((len(samples),len(CLASSES)),np.nan)
    embeddings = None
    for i,(s,crop) in enumerate(zip(samples,crops)):
        if crop is None or s['group'] in excluded:
            continue
        p,e = predictor.raw(crop)
        if embeddings is None:
            embeddings = np.full((len(samples),len(e)),np.nan,dtype=np.float32)
        probabilities[i],embeddings[i] = p,e
        if (i+1)%100 == 0:
            print('Encoded',i+1,'/',len(samples),flush=True)
    np.savez_compressed(run/'crop_features.npz',probabilities=probabilities,embeddings=embeddings)
    (ROOT/'outputs/crop_calibration/latest_run.txt').write_text(str(run),encoding='utf-8')
    print('FEATURE_RUN',run,flush=True)


if __name__ == '__main__':
    main()
