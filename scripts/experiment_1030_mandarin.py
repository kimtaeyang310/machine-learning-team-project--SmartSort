"""Group augmented unripe citrus, register holdouts, retrain and score."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.joint import ROOT, MODEL_DIR, PRETRAINED_WEIGHTS, stamp, publish
from src.joint_core import dump, sha, NinePredictor, materialize, fit, calibrate_nine
from src.joint_labels import merge_registered_samples, refresh_labels
from scripts.experiment_m06_m09_test import score


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    run = ROOT / 'outputs/joint_nine' / ('experiment_1030_unripe_' + stamp())
    run.mkdir(parents=True)
    print('RUN', run, flush=True)
    paths = sorted((ROOT / 'data/raw/1030mandarin').glob('*.jpg'))
    if len(paths) != 850:
        raise ValueError(f'Expected 850 images, found {len(paths)}')
    shutil.copytree(MODEL_DIR, run / 'previous_bundle')
    labels = ROOT / 'data/metadata/fruit_labels.csv'
    shutil.copy2(labels, run / 'fruit_labels.before.csv')
    samples = json.loads((MODEL_DIR / 'split_manifest.json').read_text(encoding='utf-8'))
    if any(s['path'].startswith('data/raw/1030mandarin/') for s in samples):
        raise ValueError('1030 data already published; reuse the registered split with python -m src.joint train')
    for registry in sorted((ROOT / 'data/metadata').glob('mandarin_*_manifest.json')):
        if registry.name == 'mandarin_1030_manifest.json':
            continue
        samples = merge_registered_samples(samples, registry)
    old_hashes = {s['sha256'] for s in samples}
    encoder = NinePredictor(MODEL_DIR / 'model.pt', 224)
    sift = cv2.SIFT_create(nfeatures=300)
    vectors, features, digests = [], [], []
    thumbs = []
    for i, path in enumerate(paths):
        digest = sha(path)
        if digest in old_hashes:
            raise ValueError(f'New data overlaps old data: {path}')
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable image: {path}')
        vectors.append(encoder.raw(path)[1])
        small = cv2.resize(image, (256, 256))
        variants = []
        for gray in (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.flip(small, 1), cv2.COLOR_BGR2GRAY)):
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            variants.append((np.float32([k.pt for k in keypoints]), descriptors))
        features.append(variants)
        digests.append(digest)
        thumbs.append(cv2.cvtColor(cv2.resize(image, (100, 100)), cv2.COLOR_BGR2RGB))
        if (i + 1) % 100 == 0:
            print('Grouping features', i + 1, '/850', flush=True)
    vectors = np.asarray(vectors)
    similarity = vectors @ vectors.T
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # Search visually similar images, then verify matching local geometry.
    pairs = set()
    for i in range(len(paths)):
        for j in np.argsort(-similarity[i])[1:121]:
            pairs.add(tuple(sorted((i, int(j)))))
    hash_groups = defaultdict(list)
    for i, digest in enumerate(digests):
        hash_groups[digest].append(i)
    for ids in hash_groups.values():
        for i in ids[1:]:
            union(ids[0], i)
    matcher = cv2.BFMatcher()
    matches = []
    for count, (i, j) in enumerate(sorted(pairs)):
        if find(i) == find(j):
            continue
        a, da = features[i][0]
        for b, db in features[j]:
            if da is None or db is None or min(len(da), len(db)) < 10:
                continue
            good = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < .72 * n.distance]
            if len(good) < 10:
                continue
            src = np.float32([a[m.queryIdx] for m in good])
            dst = np.float32([b[m.trainIdx] for m in good])
            matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.)
            if matrix is None or mask is None:
                continue
            inliers = mask.ravel().astype(bool)
            n = int(inliers.sum())
            if n >= 10 and n / len(good) >= .65 and n / min(len(da), len(db)) >= .06:
                spread = cv2.contourArea(cv2.convexHull(src[inliers])) / (256 * 256)
                if spread >= .04:
                    union(i, j)
                    matches.append({'a': paths[i].name, 'b': paths[j].name, 'inliers': n, 'spread': spread})
                    break
        if (count + 1) % 20000 == 0:
            print('Geometric pair checks', count + 1, '/', len(pairs), flush=True)
    clusters = defaultdict(list)
    for i in range(len(paths)):
        clusters[find(i)].append(i)
    groups = list(clusters.values())
    if len(groups) < 8:
        raise ValueError('Too few independent inferred source groups for four-way split')
    # Assign whole source groups; balance image counts toward 60/10/10/20.
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    ratios = {'train': .60, 'calibration': .10, 'selection': .10, 'test': .20}
    counts = Counter()
    additions, csv_new, group_info = [], [], []
    sheet = Image.new('RGB', (1000, 120 * min(len(groups), 30)), 'white')
    draw = ImageDraw.Draw(sheet)
    for gi, ids in enumerate(groups):
        phase = max(ratios, key=lambda p: ratios[p] * 850 - counts[p])
        counts[phase] += len(ids)
        token = hashlib.sha256('\n'.join(paths[i].name for i in ids).encode()).hexdigest()[:12]
        fruit_id = 'N1030_' + token
        source_group = fruit_id + '_20261030'
        csv_new.append({'fruit': 'mandarin', 'fruit_id': fruit_id, 'date': '2026-10-30',
                        'ripeness': 'unripe', 'group_id': source_group,
                        'split': 'test' if phase == 'test' else 'train', 'phase': phase})
        group_info.append({'group': fruit_id, 'phase': phase, 'files': [paths[i].name for i in ids]})
        for i in ids:
            additions.append({'path': paths[i].relative_to(ROOT).as_posix(), 'fruit': 'mandarin',
                'ripeness': 'unripe', 'group': 'mandarin:' + fruit_id, 'source_group': source_group,
                'phase': phase, 'sha256': digests[i], 'active': True,
                'label_origin': 'user instruction: all 850 unripe', 'group_origin': 'visual similarity plus SIFT geometric verification'})
        if gi < 30:
            for col, i in enumerate(ids[:9]):
                sheet.paste(Image.fromarray(thumbs[i]), (col * 110, gi * 120 + 20))
            draw.text((0, gi * 120), f'{gi}: {phase}, n={len(ids)}', fill='black')
    sheet.save(run / 'group_preview.jpg')
    dump(run / 'source_groups.json', group_info)
    dump(run / 'verified_matching_pairs.json', matches)
    dump(run / 'new_manifest.json', additions)
    with labels.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row['fruit'] == 'mandarin' and row['fruit_id'] in {'M06', 'M09'}:
            row.update(split='train', phase='train')
    rows = [r for r in rows if not (r['fruit'] == 'mandarin' and r['fruit_id'].startswith('N1030_'))] + csv_new
    with (run / 'fruit_labels.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    samples, changes = refresh_labels(samples + additions, run / 'fruit_labels.csv')
    by_hash = defaultdict(set)
    for s in samples:
        if sha(ROOT / s['path']) != s['sha256']:
            raise ValueError('Changed source: ' + s['path'])
        by_hash[s['sha256']].add(s['phase'])
    new_hashes = set(digests)
    if any(len(phases) > 1 for digest, phases in by_hash.items() if digest in new_hashes):
        raise ValueError('Exact duplicate crosses splits')
    dump(run / 'split_manifest.json', samples)
    dump(run / 'label_changes.json', changes)
    dump(run / 'registration.json', {'new_counts': dict(counts), 'source_group_count': len(groups),
         'source_group_sizes': sorted(map(len, groups), reverse=True), 'seed': 42,
         'limitation': 'Source mapping unavailable. Visual source clusters are inferred, not verified physical fruit identities; residual related images may remain across splits.'})
    shutil.copy2(run / 'fruit_labels.csv', labels)
    shutil.copy2(run / 'new_manifest.json', ROOT / 'data/metadata/mandarin_1030_manifest.json')
    print('REGISTERED', len(groups), 'groups', dict(counts), flush=True)
    config = {'epochs': 6, 'imgsz': 224, 'batch': 16, 'seed': 42, 'device': 'cpu'}
    dump(run / 'config.json', config)
    checkpoint = fit(materialize(samples, run / 'dataset'), run / 'training', PRETRAINED_WEIGHTS, config)
    predictor = NinePredictor(checkpoint, 224)
    calibrate_nine(predictor, samples, run)
    candidate = publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run,
                        destination=run / 'candidate_bundle')
    test = [s for s in samples if s['phase'] == 'test']
    after = score(candidate, test, run / 'evaluation')
    before = score(run / 'previous_bundle', test, run / 'baseline_same_test')
    dump(run / 'comparison.json', {'candidate': after, 'baseline_same_test': before})
    # The user's requested retraining becomes the main photo model; publish keeps a backup.
    publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run)
    from src.joint import evaluate
    evaluate()
    dump(run / 'completed.json', {'published': True, 'run': str(run)})
    print('COMPLETE', run, json.dumps(after['per_fruit'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
