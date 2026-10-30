"""Move exactly 100 fitting unripe tomato photos to Test by whole groups."""
import csv
import json
import shutil
from collections import Counter, defaultdict

import numpy as np
import torch

from src.joint import ROOT, MODEL_DIR, PRETRAINED_WEIGHTS, stamp, publish, evaluate
from src.joint_core import dump, sha, materialize, fit, NinePredictor, calibrate_nine, performance_metrics
from src.joint_labels import refresh_labels, merge_registered_samples
from scripts.experiment_m06_m09_test import score


def main():
    torch.set_num_threads(4)
    run = ROOT / 'outputs/joint_nine' / ('experiment_tomato_unripe_test100_' + stamp())
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
        raise ValueError('CSV/model split drift: resolve before selecting the baseline')
    groups = defaultdict(list)
    for s in samples:
        groups[s['group']].append(s)
    candidates = [g for g, members in sorted(groups.items())
                  if all(s['fruit'] == 'tomato' and s['ripeness'] == 'unripe' and s['phase'] == 'train'
                         for s in members)]
    # Identical files with different individual IDs must move together as well.
    parent = {g: g for g in groups}

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
    components = defaultdict(set)
    for g in groups:
        components[find(g)].add(g)
    eligible = set(candidates)
    units = [sorted(ids) for ids in components.values() if ids <= eligible]
    rng = np.random.default_rng(42)
    rng.shuffle(units)
    subsets = {0: []}
    for unit in units:
        size = sum(len(groups[g]) for g in unit)
        for count, selected in list(subsets.items()):
            if count + size <= 100 and count + size not in subsets:
                subsets[count + size] = selected + unit
    if 100 not in subsets:
        raise ValueError('Cannot move exactly 100 photos while keeping whole groups')
    selected_groups = set(subsets[100])
    selected_samples = [s for s in samples if s['group'] in selected_groups]
    source_groups = {s['source_group'] for s in selected_samples}
    with labels.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    csv_changes = []
    for row in rows:
        if row['fruit'] == 'tomato' and row['group_id'] in source_groups:
            before = dict(row)
            row.update(split='test', phase='test')
            csv_changes.append({'before': before, 'after': dict(row)})
    updated_csv = run / 'fruit_labels.csv'
    with updated_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    updated, changes = refresh_labels(samples, updated_csv)
    moved = [s for s, old in zip(updated, samples) if s['phase'] != old['phase']]
    if len(moved) != 100 or any(s['phase'] != 'test' for s in moved):
        raise ValueError('Unexpected sample changes')
    moved_hashes = {s['sha256'] for s in moved}
    for s in updated:
        if sha(ROOT / s['path']) != s['sha256']:
            raise ValueError('Source changed: ' + s['path'])
        if s['sha256'] in moved_hashes and s['phase'] != 'test':
            raise ValueError('Moved photo duplicates a development image')
    dump(run / 'selected_samples.json', selected_samples)
    dump(run / 'csv_changes.json', csv_changes)
    dump(run / 'label_changes.json', changes)
    dump(run / 'split_manifest.json', updated)
    dump(run / 'selection.json', {'seed': 42, 'selected_groups': sorted(selected_groups),
         'moved_count': len(moved), 'selection_rule': 'Seeded group order, subset sum exactly 100; no predictions used',
         'tomato_counts': dict(Counter(s['phase'] + '/' + s['ripeness'] for s in updated if s['fruit'] == 'tomato')),
         'limitation': 'Moved individuals were previously used to train the baseline. Baseline on the new Test is contaminated; retrained model starts from pretrained weights.'})
    shutil.copy2(updated_csv, labels)
    config = {'epochs': 6, 'imgsz': 224, 'batch': 16, 'seed': 42, 'device': 'cpu'}
    dump(run / 'config.json', config)
    print('MOVED', len(moved), 'photos in', len(selected_groups), 'groups', flush=True)
    print('Tomato counts', Counter(s['phase'] + '/' + s['ripeness'] for s in updated if s['fruit'] == 'tomato'), flush=True)
    checkpoint = fit(materialize(updated, run / 'dataset'), run / 'training', PRETRAINED_WEIGHTS, config)
    predictor = NinePredictor(checkpoint, 224)
    calibrate_nine(predictor, updated, run)
    candidate = publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run,
                        destination=run / 'candidate_bundle')
    test = [s for s in updated if s['phase'] == 'test']
    after = score(candidate, test, run / 'evaluation')
    before = score(run / 'previous_bundle', test, run / 'baseline_same_test')
    cohorts = {}
    for name, directory in [('candidate', run / 'evaluation'), ('baseline', run / 'baseline_same_test')]:
        predictions = json.loads((directory / 'predictions.json').read_text(encoding='utf-8'))
        old_paths = {str((ROOT / s['path']).resolve()) for s in samples if s['phase'] == 'test'}
        old_rows = [r for r in predictions if r['image_path'] in old_paths]
        cohorts[name] = {'original_test': performance_metrics(old_rows),
                        'newly_moved_100': dict(Counter(r['status'] + '/' + r['candidate_class']
                            for r in predictions if r['image_path'] not in old_paths))}
    dump(run / 'comparison.json', {'candidate': after, 'baseline_same_test': before, 'cohorts': cohorts,
         'baseline_new_test_contaminated': True})
    publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run)
    evaluate()
    dump(run / 'completed.json', {'published': True, 'run': str(run)})
    print('COMPLETE', run, json.dumps(after['per_fruit'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
