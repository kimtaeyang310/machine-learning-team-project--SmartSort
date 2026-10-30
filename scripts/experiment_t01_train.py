"""Restore all T01 photos to fitting and rerun the fixed photo-model protocol."""
import csv
import json
import shutil
from collections import Counter, defaultdict

import numpy as np
import torch

from src.joint import ROOT, MODEL_DIR, PRETRAINED_WEIGHTS, stamp, publish, evaluate
from src.joint_core import dump, sha, materialize, fit, NinePredictor, calibrate_nine
from src.joint_labels import refresh_labels, merge_registered_samples
from scripts.experiment_m06_m09_test import score


def main(random_unripe40=False):
    torch.set_num_threads(4)
    prefix = 'experiment_tomato_test40_train_' if random_unripe40 else 'experiment_t01_train_'
    run = ROOT / 'outputs/joint_nine' / (prefix + stamp())
    run.mkdir(parents=True)
    print('RUN', run, flush=True)
    labels = ROOT / 'data/metadata/fruit_labels.csv'
    shutil.copy2(labels, run / 'fruit_labels.before.csv')
    shutil.copytree(MODEL_DIR, run / 'previous_bundle')
    samples = json.loads((MODEL_DIR / 'split_manifest.json').read_text(encoding='utf-8'))
    for registry in sorted((ROOT / 'data/metadata').glob('mandarin_*_manifest.json')):
        samples = merge_registered_samples(samples, registry)
    samples, pending = refresh_labels(samples, labels)
    if pending:
        raise ValueError('CSV/model labels differ before experiment')
    if random_unripe40:
        eligible = [s for s in samples if s['fruit'] == 'tomato' and s['ripeness'] == 'unripe' and s['phase'] == 'test']
        if len(eligible) != 46:
            raise ValueError('Expected exactly 46 unripe tomato Test photos')
        # Connect both individual IDs and identical file hashes before sampling.
        parent = {s['group']: s['group'] for s in samples}

        def find(g):
            while parent[g] != g:
                parent[g] = parent[parent[g]]
                g = parent[g]
            return g

        by_hash = defaultdict(list)
        for s in samples:
            by_hash[s['sha256']].append(s['group'])
        for ids in by_hash.values():
            for g in ids[1:]:
                parent[find(g)] = find(ids[0])
        components = defaultdict(list)
        for s in samples:
            components[find(s['group'])].append(s)
        units = [v for _, v in sorted(components.items()) if all(
            s['fruit'] == 'tomato' and s['ripeness'] == 'unripe' and s['phase'] == 'test' for s in v)]
        np.random.default_rng(42).shuffle(units)
        subsets = {0: []}
        for unit in units:
            for count, selected in list(subsets.items()):
                if count + len(unit) <= 40 and count + len(unit) not in subsets:
                    subsets[count + len(unit)] = selected + unit
        if 40 not in subsets:
            raise ValueError('Cannot select exactly 40 while retaining duplicate/individual groups')
        target = subsets[40]
    else:
        target = [s for s in samples if s['group'] == 'tomato:T01']
        if not target or any(s['phase'] != 'test' for s in target):
            raise ValueError('Expected T01 entirely in Test before this experiment')
    target_groups = {s['group'] for s in target}
    source_groups = {s['source_group'] for s in target}
    with labels.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    csv_changes = []
    for row in rows:
        if row['fruit'] == 'tomato' and row['group_id'] in source_groups:
            before = dict(row)
            row.update(split='train', phase='train')
            csv_changes.append({'before': before, 'after': dict(row)})
    updated_csv = run / 'fruit_labels.csv'
    with updated_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    updated, changes = refresh_labels(samples, updated_csv)
    for old, new in zip(samples, updated):
        if old['group'] not in target_groups and old != new:
            raise ValueError('Unexpected changes outside selected individuals')
        if new['group'] in target_groups and new['phase'] != 'train':
            raise ValueError('Selected individual not entirely in training')
        if sha(ROOT / new['path']) != new['sha256']:
            raise ValueError('Source hash changed: ' + new['path'])
    moved_hashes = {s['sha256'] for s in target}
    if any(s['sha256'] in moved_hashes and s['phase'] != 'train' for s in updated):
        raise ValueError('Selected photo duplicates another holdout image')
    dump(run / 'csv_changes.json', csv_changes)
    dump(run / 'label_changes.json', changes)
    dump(run / 'split_manifest.json', updated)
    counts = Counter(s['phase'] + '/' + s['ripeness'] for s in updated if s['fruit'] == 'tomato')
    dump(run / 'selected_samples.json', target)
    dump(run / 'remaining_unripe_test.json', [s for s in updated if s['fruit'] == 'tomato' and s['ripeness'] == 'unripe' and s['phase'] == 'test'])
    dump(run / 'split_summary.json', {'moved_count': len(target), 'tomato_counts': dict(counts),
         'seed': 42 if random_unripe40 else None, 'selection': 'seeded whole-group subset sum; no predictions used' if random_unripe40 else 'all T01'})
    shutil.copy2(updated_csv, labels)
    config = {'epochs': 6, 'imgsz': 224, 'batch': 16, 'seed': 42, 'device': 'cpu'}
    dump(run / 'config.json', config)
    print('Moved to train', len(target), 'Tomato', dict(counts), flush=True)
    checkpoint = fit(materialize(updated, run / 'dataset'), run / 'training', PRETRAINED_WEIGHTS, config)
    predictor = NinePredictor(checkpoint, 224)
    calibrate_nine(predictor, updated, run)
    candidate = publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run,
                        destination=run / 'candidate_bundle')
    test = [s for s in updated if s['phase'] == 'test']
    after = score(candidate, test, run / 'evaluation')
    before = score(run / 'previous_bundle', test, run / 'baseline_same_test')
    dump(run / 'comparison.json', {'candidate': after, 'baseline_same_test': before,
         'note': 'Both models scored on the same remaining Test. Test was repeatedly inspected; regression evaluation, not new blind validation.'})
    publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run)
    evaluate()
    dump(run / 'completed.json', {'published': True, 'run': str(run)})
    print('COMPLETE', run, json.dumps(after['per_fruit'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-unripe40', action='store_true')
    main(parser.parse_args().random_unripe40)
