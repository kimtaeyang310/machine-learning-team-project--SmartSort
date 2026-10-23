"""Train a frozen-backbone fruit classifier, calibrate rejection, then evaluate holdouts."""

from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from src.dataset import load_dataset
from src.fruit_classifier import (ROOT, MODEL_PATH, FRUITS, IMAGE_EXTENSIONS,
                                 ImageEncoder, unfamiliarity_scores,
                                 rejection_threshold, rejection_metrics, threshold_values)


def image_fingerprint(path):
    """Decoded-pixel hash detects renamed/re-encoded exact copies; dHash groups unknown lookalikes."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        gray = np.asarray(im.convert("L").resize((9, 8)))
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    perceptual = sum(int(bit) << i for i, bit in enumerate(bits))
    return digest, perceptual


def split_unknown(paths, seed=42):
    """Conservatively keep near duplicates together; distinct fruit specimens are not identifiable."""
    fingerprints = [image_fingerprint(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (digest, phash) in enumerate(fingerprints):
        for j in range(i):
            other, other_phash = fingerprints[j]
            if digest == other or (phash ^ other_phash).bit_count() <= 4:
                parent[find(i)] = find(j)
    groups = [find(i) for i in range(len(paths))]
    if len(set(groups)) < 2:
        raise ValueError("모르는 과일은 독립된 사진 그룹이 최소 2개 필요합니다.")
    selection, test = next(GroupShuffleSplit(n_splits=1, test_size=0.5,
                                             random_state=seed).split(paths, groups=groups))
    return selection, test, groups


def assign_identity_splits(samples):
    """Balance specimen counts and stage coverage using metadata, never image scores."""
    for fruit in FRUITS:
        indices = [i for i, s in enumerate(samples) if s["fruit"] == fruit and s["phase"] == "train"]
        by_group = {}
        for i in indices:
            by_group.setdefault(samples[i]["group"], []).append(i)
        groups = sorted(by_group)
        if len(groups) < 5:
            raise ValueError(f"{fruit}: independent specimen groups must be at least 5")
        holdout_count = max(1, round(len(groups) * .2))
        total = Counter(samples[i]["ripeness"] for i in indices)
        stage_groups = {stage: len({samples[i]["group"] for i in indices
                                    if samples[i]["ripeness"] == stage}) for stage in total}
        best = None
        for seed in range(42, 542):
            shuffled = np.random.default_rng(seed).permutation(groups).tolist()
            partitions = (shuffled[2 * holdout_count:], shuffled[:holdout_count],
                          shuffled[holdout_count:2 * holdout_count])
            missing, error = [0, 0, 0], 0.0
            for phase_number, (selected, fraction) in enumerate(zip(partitions, (.6, .2, .2))):
                counts = Counter(samples[i]["ripeness"] for group in selected for i in by_group[group])
                missing[phase_number] += sum(counts[stage] == 0 for stage in total
                                             if stage_groups[stage] >= phase_number + 1)
                error += sum((counts[stage] / total[stage] - fraction) ** 2 for stage in total)
            key = (*missing, error, seed)
            if best is None or key < best[0]:
                best = (key, partitions)
        if best[0][0] or best[0][1] or best[0][2]:
            raise ValueError(f"{fruit}: cannot cover available ripeness groups with disjoint specimens")
        for phase, partition in zip(("train", "calibration", "selection"), best[1]):
            for group in partition:
                for i in by_group[group]:
                    samples[i]["phase"] = phase
    phases = {}
    for s in samples:
        phases.setdefault(s["group"], set()).add(s["phase"])
    if any(len(v) > 1 for v in phases.values()):
        raise ValueError("Specimen ID crosses training, validation, or legacy Test")
    return samples


def prepare_samples():
    samples = []
    for sample in load_dataset():
        path = Path(sample["path"])
        samples.append({"path": path.relative_to(ROOT).as_posix(), "fruit": sample["fruit"],
                        "ripeness": sample["ripeness"],
                        "group": f"{sample['fruit']}:{sample.get('fruit_id') or sample['group_id']}",
                        "source_group": sample["group_id"],
                        "phase": "test" if sample["split"] == "test" else "train",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assign_identity_splits(samples)
    for phase in ("train", "calibration", "selection", "test"):
        if {s["fruit"] for s in samples if s["phase"] == phase} != set(FRUITS):
            raise ValueError(f"{phase}: 세 과일이 모두 필요합니다.")
    for fruit in ("grape", "lemon"):
        paths = sorted(p for p in (ROOT / "data" / fruit).rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(paths) < 2:
            raise ValueError(f"data/{fruit}: 사진이 최소 2장 필요합니다.")
        selection, test, groups = split_unknown(paths)
        for phase, indices in (("selection", selection), ("test", test)):
            for i in indices:
                samples.append({"path": paths[i].relative_to(ROOT).as_posix(), "fruit": fruit,
                                "ripeness": "", "group": f"{fruit}:{groups[i]}",
                                "source_group": f"{fruit}:{groups[i]}", "phase": phase,
                                "sha256": hashlib.sha256(paths[i].read_bytes()).hexdigest()})
    # Refuse duplicate leakage instead of silently moving existing Test photos into training.
    hashes, group_phases = {}, {}
    for s in samples:
        hashes.setdefault(s["sha256"], set()).add(s["phase"])
        group_phases.setdefault(s["group"], set()).add(s["phase"])
    if any(len(v) > 1 for v in hashes.values()) or any(len(v) > 1 for v in group_phases.values()):
        raise ValueError("분할 사이 중복 사진 또는 그룹이 있습니다. 데이터 분할을 확인하세요.")
    return samples


def encode_cached(encoder, samples):
    """Cache is keyed by encoder state, preprocessing version, and image bytes."""
    state_hash = hashlib.sha256()
    for key, value in sorted(encoder.state().items()):
        state_hash.update(key.encode())
        state_hash.update(value.tobytes())
    import inspect
    import torch
    import torchvision
    from src.fruit_classifier import prepare_identity_image
    state_hash.update((encoder.preprocess_mode + inspect.getsource(prepare_identity_image)
                       + torch.__version__ + torchvision.__version__).encode())
    cache = ROOT / "data" / "features" / "cache" / "fruit_identity" / state_hash.hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    embeddings = [None] * len(samples)
    missing = []
    for i, s in enumerate(samples):
        target = cache / (s["sha256"] + "_imagenet-v1-exif.npy")
        if target.is_file():
            embeddings[i] = np.load(target, allow_pickle=False)
        else:
            missing.append(i)
    for start in range(0, len(missing), 32):
        indices = missing[start:start + 32]
        vectors = encoder.encode([ROOT / samples[i]["path"] for i in indices])
        for i, vector in zip(indices, vectors):
            embeddings[i] = vector
            np.save(cache / (samples[i]["sha256"] + "_imagenet-v1-exif.npy"), vector)
        print(f"이미지 특징 계산: {min(start + 32, len(missing))}/{len(missing)}", flush=True)
    return np.asarray(embeddings)


def compare_methods(samples, probabilities, scores, thresholds, phase):
    known = np.array([s["fruit"] in FRUITS and s["phase"] == phase for s in samples])
    unknown_phase = "selection" if phase == "calibration" else phase
    unknown = np.array([s["fruit"] not in FRUITS and s["phase"] == unknown_phase for s in samples])
    report = {}
    for method, values in scores.items():
        predicted = np.array(FRUITS)[probabilities.argmax(axis=1)]
        configured_threshold = thresholds[method]
        threshold = threshold_values(configured_threshold, predicted)
        metrics = rejection_metrics(values[known] - threshold[known], values[unknown] - threshold[unknown], 0)
        metrics["threshold"] = configured_threshold
        metrics["auroc_unknown_positive"] = float(roc_auc_score(
            [0] * int(known.sum()) + [1] * int(unknown.sum()), np.r_[values[known], values[unknown]]))
        truth = np.array([s["fruit"] for s in samples])
        metrics["known_type_accuracy_before_rejection"] = float(np.mean(predicted[known] == truth[known]))
        metrics["known_correct_and_accepted_rate"] = float(np.mean(
            (predicted[known] == truth[known]) & (values[known] <= threshold[known])))
        metrics["by_fruit"] = {}
        for fruit in (*FRUITS, "grape", "lemon"):
            mask = (known if fruit in FRUITS else unknown) & (truth == fruit)
            rejected = int(np.sum(values[mask] > threshold[mask]))
            metrics["by_fruit"][fruit] = {"count": int(mask.sum()), "rejected_count": rejected,
                "false_rejection_rate" if fruit in FRUITS else "false_acceptance_rate":
                    float(np.mean(values[mask] > threshold[mask] if fruit in FRUITS else values[mask] <= threshold[mask]))}
            if fruit in FRUITS:
                metrics["by_fruit"][fruit]["type_accuracy"] = float(np.mean(predicted[mask] == fruit))
                metrics["by_fruit"][fruit]["correct_and_accepted_rate"] = float(np.mean(
                    (predicted[mask] == fruit) & (values[mask] <= threshold[mask])))
        metrics["known_balanced_error_rate"] = float(np.mean([
            np.mean((predicted[known & (truth == fruit)] != fruit) |
                    (values[known & (truth == fruit)] > threshold[known & (truth == fruit)]))
            for fruit in FRUITS]))
        report[method] = metrics
    return report


def select_method(selection_report):
    """Only calibration/selection results enter policy selection; Test is never an input."""
    return min(selection_report, key=lambda m: (
        selection_report[m]["unknown_false_acceptance_rate"],
        selection_report[m]["known_false_rejection_rate"], m != "embedding"))


def write_reports(directory, samples, probabilities, scores, thresholds, report):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["path", "fruit", "ripeness", "phase", "group", "source_group", "sha256", "candidate_fruit",
              "method", "score", "threshold", "accepted", "probabilities_json"]
    with (directory / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(samples):
            for method, values in scores.items():
                candidate = FRUITS[int(probabilities[i].argmax())]
                threshold = float(threshold_values(thresholds[method], [candidate])[0])
                writer.writerow({**s, "candidate_fruit": FRUITS[int(probabilities[i].argmax())],
                    "method": method, "score": float(values[i]), "threshold": threshold,
                    "accepted": bool(values[i] <= threshold),
                    "probabilities_json": json.dumps(dict(zip(FRUITS, probabilities[i].tolist())))})
    is_v2 = report.get("version") == 2
    lines = ["# 과일 종류·모르는 과일 평가", "", f"선택 방법: {report['selected_method']}",
             f"전처리: {report.get('preprocess_mode', 'center_crop')}",
             "", "임계값은 Calibration에서 계산했습니다. v2는 정답 과일별 95% 이상을 종류 정답과 함께 통과시키는 기준입니다.",
             "v2 선택은 별도 Selection의 미지 오수락률 → 과일별 평균 오류율(거부 또는 종류 오분류) → 토마토 오거부율 순입니다.",
             "Test는 이번 실행의 선택·임계값 계산에 사용하지 않았습니다. 기존 Test는 앞선 원인 분석에 사용된 회귀 평가 자료입니다."
             if is_v2 else "Test는 이번 실행의 선택·임계값 계산에 사용하지 않았습니다.", "",
             "| 구분 | 방법 | 아는 과일 오거부 | 모르는 과일 오수락 | 종류 정확도(거부 전) |", "|---|---|---|---|---|"]
    for phase in (("calibration", "selection", "test") if "calibration" in report else ("selection", "test")):
        for method, m in report[phase].items():
            lines.append(f"| {phase} | {method} | {m['known_false_rejection_count']}/{m['known_count']} ({m['known_false_rejection_rate']:.2%}) | "
                         f"{m['unknown_false_acceptance_count']}/{m['unknown_count']} ({m['unknown_false_acceptance_rate']:.2%}) | {m['known_type_accuracy_before_rejection']:.2%} |")
    if "calibration" in report:
        lines += ["", "## 선택 정책의 과일별 Calibration 기준 확인", "",
                  "| 과일 | 사진 수 | 종류 정답·수락 비율 | 95% 목표 |", "|---|---|---|---|"]
        for fruit in FRUITS:
            item = report["calibration"][report["selected_method"]]["by_fruit"][fruit]
            if "correct_and_accepted_rate" in item:
                rate = item["correct_and_accepted_rate"]
                lines.append(f"| {fruit} | {item['count']} | {rate:.2%} | {'충족' if rate >= .95 else '미충족'} |")
    lines += ["", "## 기존 Test 회귀 평가: 과일별 상세", "",
              "| 방법 | 과일 | 사진 수 | 오류 종류 | 오류율 |", "|---|---|---|---|---|"]
    for method, metrics in report["test"].items():
        for fruit, item in metrics["by_fruit"].items():
            field = "false_rejection_rate" if fruit in FRUITS else "false_acceptance_rate"
            label = "아는 과일 오거부" if fruit in FRUITS else "모르는 과일 오수락"
            lines.append(f"| {method} | {fruit} | {item['count']} | {label} | {item[field]:.2%} |")
    lines += ["", "v2는 과일+개체 ID/바나나 원본 그룹으로 분리합니다. v1은 개체+날짜를 사용했습니다.",
              "포도·레몬은 dHash 유사 사진을 묶어 분할했습니다. 같은 개체/연속 촬영 여부는 메타데이터가 없어 완전히 확인할 수 없습니다."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def train(model_path=MODEL_PATH):
    from src.fruit_identity_experiment import run_experiment
    return run_experiment(model_path)


def main():
    parser = argparse.ArgumentParser(description="개체 분리·전처리 비교·과일별 95% 통과 기준 학습")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    train(args.model_path)


if __name__ == "__main__":
    main()
