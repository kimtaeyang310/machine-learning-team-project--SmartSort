"""Retrain the photo classifier with mandarin M06/M09 held out as Test."""
import csv
import json
import shutil
from collections import Counter
from time import perf_counter

import torch

from src.joint import ROOT, MODEL_DIR, PRETRAINED_WEIGHTS, JointClassifier, publish, stamp
from src.joint_core import dump, sha, materialize, fit, NinePredictor, calibrate_nine, performance_metrics
from src.joint_labels import refresh_labels, merge_registered_samples


def score(directory, samples, destination):
    classifier = JointClassifier(directory)
    rows = []
    for sample in samples:
        start = perf_counter()
        result = classifier.predict(ROOT / sample['path'])
        result.update(truth_fruit=sample['fruit'], truth_ripeness=sample['ripeness'],
                      truth_class=sample['fruit'] + '_' + sample['ripeness'],
                      total_ms=(perf_counter() - start) * 1000)
        rows.append(result)
    metrics = performance_metrics(rows)
    dump(destination / 'metrics.json', metrics)
    dump(destination / 'predictions.json', rows)
    return metrics


def main():
    torch.set_num_threads(4)
    run = ROOT / 'outputs/joint_nine' / ('experiment_m06_m09_test_' + stamp())
    run.mkdir(parents=True)
    labels = ROOT / 'data/metadata/fruit_labels.csv'
    shutil.copy2(labels, run / 'fruit_labels.before.csv')
    shutil.copytree(MODEL_DIR, run / 'previous_bundle')
    with labels.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    changes = []
    for row in rows:
        if row['fruit'] == 'mandarin' and row['fruit_id'] in {'M06', 'M09'}:
            before = dict(row)
            row.update(split='test', phase='test')
            changes.append({'before': before, 'after': dict(row)})
    if {r['after']['fruit_id'] for r in changes} != {'M06', 'M09'}:
        raise ValueError('Both requested individuals must be present')
    updated_csv = run / 'fruit_labels.csv'
    with updated_csv.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    samples = json.loads((MODEL_DIR / 'split_manifest.json').read_text(encoding='utf-8'))
    samples = merge_registered_samples(samples, ROOT / 'data/metadata/mandarin_1029_manifest.json')
    samples, label_changes = refresh_labels(samples, updated_csv)
    for sample in samples:
        if sha(ROOT / sample['path']) != sample['sha256']:
            raise ValueError('Source changed: ' + sample['path'])
        if sample['group'] in {'mandarin:M06', 'mandarin:M09'} and sample['phase'] != 'test':
            raise ValueError('Requested individual leaked into development')
    dump(run / 'csv_changes.json', changes)
    dump(run / 'label_changes.json', label_changes)
    dump(run / 'split_manifest.json', samples)
    shutil.copy2(updated_csv, labels)
    config = {'epochs': 6, 'imgsz': 224, 'batch': 16, 'seed': 42, 'device': 'cpu'}
    dump(run / 'config.json', config)
    print('EXPERIMENT_RUN', run, flush=True)
    print('Mandarin split', Counter((s['phase'], s['ripeness']) for s in samples if s['fruit'] == 'mandarin'), flush=True)
    checkpoint = fit(materialize(samples, run / 'dataset'), run / 'training', PRETRAINED_WEIGHTS, config)
    predictor = NinePredictor(checkpoint, 224)
    calibrate_nine(predictor, samples, run)
    candidate = publish(checkpoint, run / 'nine_rejection_policy.joblib', run / 'split_manifest.json', run,
                        destination=run / 'candidate_bundle')
    test = [s for s in samples if s['phase'] == 'test']
    after = score(candidate, test, run / 'evaluation')
    before = score(run / 'previous_bundle', test, run / 'baseline_same_test')
    dump(run / 'comparison.json', {'baseline_same_test': before, 'candidate': after,
         'production_model_changed': False,
         'note': 'M06 was used in baseline training and M09 in baseline selection; baseline is contaminated. No unripe mandarin Test exists. Candidate trained from pretrained weights; Test excluded from fitting/calibration/selection.'})
    print(json.dumps({'run': str(run), 'candidate': after, 'baseline_same_test': before}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
