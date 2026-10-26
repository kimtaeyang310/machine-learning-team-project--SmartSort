"""Reproducible YOLOv8n vs YOLO26n comparison, confined to the lab."""
import json
import argparse
import csv
import platform
from collections import Counter
from time import perf_counter

import run_experiment as lab
import ultralytics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--relabel-t08-t09', action='store_true')
    args = parser.parse_args()
    baseline = lab.HERE / "artifacts/20261023T071957744434Z"
    run = lab.HERE / ("artifacts/yolo26n_relabel_t08_t09_20261026" if args.relabel_t08_t09
                      else "artifacts/yolo26n_comparison_20261026")
    run.mkdir(parents=True, exist_ok=True)
    before = lab.source_files()
    lab.dump(run / "source_before.json", before)
    samples = json.loads((baseline / "split_manifest.json").read_text(encoding="utf-8"))
    changes = []
    if args.relabel_t08_t09:
        with (lab.ROOT / 'data/metadata/fruit_labels.csv').open(encoding='utf-8-sig', newline='') as f:
            metadata = {(r['fruit'], r['group_id']): r for r in csv.DictReader(f)}
        for sample in samples:
            if sample['group'] not in ('tomato:T08', 'tomato:T09'):
                continue
            row = metadata[(sample['fruit'], sample['source_group'])]
            phase = 'train' if sample['group'] == 'tomato:T08' else 'test'
            if row['ripeness'] != 'ripe' or row['split'] != phase:
                raise ValueError(f"Requested metadata not present: {row}")
            changes.append({'path': sample['path'], 'old_ripeness': sample['ripeness'],
                            'old_phase': sample['phase'], 'ripeness': 'ripe', 'phase': phase})
            sample.update(ripeness='ripe', phase=phase)
        if Counter(s['group'] for s in samples if s['group'] in ('tomato:T08', 'tomato:T09')) != {'tomato:T08': 10, 'tomato:T09': 10}:
            raise ValueError('Expected ten photos per corrected specimen')
    groups, hashes = {}, {}
    for sample in samples:
        if lab.sha(lab.ROOT / sample["path"]) != sample["sha256"]:
            raise ValueError(f"Dataset changed: {sample['path']}")
        groups.setdefault(sample['group'], set()).add(sample['phase'])
        hashes.setdefault(sample['sha256'], set()).add(sample['phase'])
    if any(len(v) > 1 for v in [*groups.values(), *hashes.values()]):
        raise ValueError('Specimen or image leakage')
    lab.dump(run / 'dataset_changes.json', changes)
    print('Split counts:', Counter(s['phase'] for s in samples), flush=True)
    lab.dump(run / "split_manifest.json", samples)
    config = json.loads((baseline / "config.json").read_text())
    lab.dump(run / "config.json", {**config, "ultralytics": ultralytics.__version__,
        "torch": lab.torch.__version__, "platform": platform.platform(),
        "baseline": str(baseline), "comparison": "pretrained v8n-cls versus 26n-cls; same nine-class task"})
    lab.torch.set_num_threads(4)
    snapshot = run / "snapshot"
    snapshot.mkdir(exist_ok=True)
    weights = snapshot / "yolo26n-cls.pt"
    previous_weights = lab.HERE / 'artifacts/yolo26n_comparison_20261026/snapshot/yolo26n-cls.pt'
    if not weights.exists() and previous_weights.is_file():
        import shutil
        shutil.copy2(previous_weights, weights)
    if not weights.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(weights))
    if not weights.is_file():
        raise FileNotFoundError(weights)
    data = lab.materialize(samples, run / "datasets/nine")
    checkpoint = lab.fit(data, run / "training/nine", weights, config)
    v8_checkpoint = baseline / 'training/nine/weights/last.pt'
    if args.relabel_t08_t09:
        v8_checkpoint = lab.fit(data, run / 'training/nine_v8', baseline / 'snapshot/pretrained.pt', config)
    lab.torch.set_num_threads(4)
    predictors, loading = {}, {}
    for name, ckpt in (("yolov8n_cls", v8_checkpoint),
                       ("yolo26n_cls", checkpoint)):
        directory = run / name
        directory.mkdir(exist_ok=True)
        start = perf_counter()
        predictor = lab.NinePredictor(ckpt, config["imgsz"])
        loading[name] = perf_counter() - start
        lab.calibrate_nine(predictor, samples, directory)
        predictors[name] = predictor
    metrics = lab.benchmark(predictors, [s for s in samples if s["phase"] == "test"],
                            config["timing_rounds"], run)
    after = lab.source_files()
    changed = [p for p in before if before[p] != after.get(p)]
    lab.dump(run / "source_integrity.json", {"unchanged": not changed, "changed_files": changed})
    result = {"config": config, "metrics": metrics, "loading_seconds": loading,
              "original_project_unchanged": not changed,
              "dataset_changes": changes,
              "split_counts": dict(Counter(s['phase'] for s in samples)),
              "checkpoints": {"yolov8n_cls": lab.sha(v8_checkpoint),
                              "yolo26n_cls": lab.sha(checkpoint), "pretrained26": lab.sha(weights)}}
    lab.dump(run / "report.json", result)
    a, b = metrics["yolov8n_cls"], metrics["yolo26n_cls"]
    lines = ["# YOLO26n-cls 교체 비교 실험", "",
        "기존 YOLOv8n-cls 9클래스 통합 모델과 YOLO26n-cls를 비교했습니다.",
        f"동일 Train {sum(s['phase']=='train' for s in samples):,}장, 224 입력, batch 16, CPU, seed 42, 6 epochs, last.pt 사용.",
        "두 모델 모두 변경된 데이터로 사전학습 가중치부터 재학습했습니다." if args.relabel_t08_t09 else "v8은 기존 완료 체크포인트를 재사용했습니다.",
        "T08·T09는 적숙으로 수정하고 T08의 10장을 Test에서 Train으로 옮겼습니다. 나머지 분할은 이전 manifest를 유지했습니다." if changes else "기존 분할 유지.",
        "이전 Test 오류를 확인한 뒤 라벨과 분할을 바꾼 회귀 실험이므로 이전 점수와 직접 비교하거나 독립 일반화 성능으로 해석할 수 없습니다." if changes else "기존 Test 회귀 비교입니다.",
        "각 모델의 ImageNet 사전학습 가중치를 사용하며 optimizer=auto 등 기존 실험 설정을 유지했습니다.",
        "모름 판정은 동일 Calibration/Selection 절차로 모델별로 보정했습니다. Test로 선택하지 않았습니다.", "",
        "| 지표 | YOLOv8n-cls | YOLO26n-cls |", "|---|---:|---:|"]
    for title, key in (("9클래스 Macro F1 (거부 포함)", "nine_class_macro_f1"),
                       ("9클래스 Macro F1 (거부 전)", "without_rejection_nine_class_macro_f1"),
                       ("종류·숙도 정답률", "joint_accuracy_including_rejections"),
                       ("알려진 과일 오거부율", "known_false_rejection_rate"),
                       ("미지 과일 오수락률", "unknown_false_acceptance_rate")):
        lines.append(f"| {title} | {a[key]:.4f} | {b[key]:.4f} |")
    for fruit in lab.FRUITS:
        lines.append(f"| {fruit} 숙도 Macro F1 | {a['per_fruit'][fruit]['ripeness_macro_f1']:.4f} | {b['per_fruit'][fruit]['ripeness_macro_f1']:.4f} |")
    for title, key in (("평균 처리 시간 (ms)", "mean_ms"), ("P95 처리 시간 (ms)", "p95_ms"), ("FPS", "fps")):
        lines.append(f"| {title} | {a['timing'][key]:.2f} | {b['timing'][key]:.2f} |")
    lines += ["", f"운영 Macro F1 변화: {(b['nine_class_macro_f1']-a['nine_class_macro_f1'])*100:+.2f}%p.",
        f"평균 처리 시간 변화: {(b['timing']['mean_ms']/a['timing']['mean_ms']-1)*100:+.2f}%.", "",
        f"평가는 알려진 과일 {a['known_count']}장, 미지 과일(포도·레몬) {a['unknown_count']}장입니다. 시간은 워밍업 후 2회이며 이미지 읽기와 모름 판정을 포함합니다.",
        "기존에 분석한 Test를 재사용한 회귀 비교이며, 단일 seed·6 epochs 결과입니다. 충분히 수렴한 최종 성능이나 새로운 미지 과일의 성능을 뜻하지 않습니다.",
        f"처리 실패: v8={a['failure_count']}, 26={b['failure_count']}. 기존 프로젝트 내용 해시 보존: {not changed}."]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if changed:
        raise RuntimeError(f"Source changed externally: {changed}")


if __name__ == "__main__":
    main()
