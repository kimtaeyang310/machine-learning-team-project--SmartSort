"""Refresh CSV labels and train/test membership, retaining development phases."""
# CSV의 최신 숙도와 분할을 기존 사진 목록에 반영합니다.
# 같은 개체가 학습·검증·평가에 나뉘면 평가가 부풀려지므로 그룹 단위로 검사합니다.
import csv
import json
from pathlib import Path


# 기존 목록을 복사하고 활성 등록 자료만 추가하며 동일 경로의 식별 정보가 다르면 중단합니다.
def merge_registered_samples(samples, registry_path):
    """Include explicitly registered new sources; keep raw files and snapshots intact."""
    merged = [dict(sample) for sample in samples]
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        return merged
    by_path = {sample['path']: sample for sample in merged}
    for sample in json.loads(registry_path.read_text(encoding='utf-8')):
        if not sample.get('active', True):
            continue
        if sample['path'] in by_path:
            previous = by_path[sample['path']]
            if any(previous[key] != sample[key] for key in ('sha256', 'fruit', 'group', 'source_group')):
                raise ValueError(f"등록 자료와 모델 목록이 충돌합니다: {sample['path']}")
            continue
        merged.append(dict(sample))
        by_path[sample['path']] = sample
    return merged


# 귤·토마토 CSV 라벨을 반영하고 갱신 목록과 변경 내역을 함께 반환합니다.
def refresh_labels(samples, csv_path):
    labels = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (row["fruit"].strip(), row["group_id"].strip())
            stage = row["ripeness"].strip()
            split = row.get("split", "").strip()
            if split not in {"train", "test"}:
                raise ValueError(f"잘못된 split (train/test 필요): {key}: {split}")
            if stage not in {"unripe", "ripe", "overripe"}:
                raise ValueError(f"잘못된 숙도 라벨: {key}: {stage}")
            phase = (row.get("phase") or "").strip()
            if phase and (phase not in {"train", "calibration", "selection", "test"} or
                          (split == "test") != (phase == "test")):
                raise ValueError(f"split과 맞지 않는 phase: {key}: {split}/{phase}")
            if key in labels and labels[key] != (stage, split, phase):
                raise ValueError(f"중복된 그룹의 라벨이 다릅니다: {key}")
            labels[key] = (stage, split, phase)
    refreshed, changes = [], []
    for sample in samples:
        updated = dict(sample)
        if sample["fruit"] in {"mandarin", "tomato"}:
            key = (sample["fruit"], sample["source_group"])
            if key not in labels:
                raise ValueError(f"CSV에 숙도 라벨이 없습니다: {key}")
            updated["ripeness"], split, explicit_phase = labels[key]
            # CSV train includes the existing calibration/selection holdouts.
            # Keep those holdouts; only move test samples back to fitting.
            updated["phase"] = explicit_phase or ("test" if split == "test" else
                                "train" if sample["phase"] == "test" else sample["phase"])
            if (updated["ripeness"], updated["phase"]) != (sample["ripeness"], sample["phase"]):
                changes.append({"path": sample["path"], "phase": sample["phase"],
                                "before": sample["ripeness"], "after": updated["ripeness"],
                                "phase_before": sample["phase"], "phase_after": updated["phase"]})
        refreshed.append(updated)
    groups = {}
    for sample in refreshed:
        groups.setdefault(sample["group"], set()).add(sample["phase"])
    conflicts = [group for group, phases in groups.items() if len(phases) > 1]
    if conflicts:
        raise ValueError(f"같은 개체의 CSV split을 일치시켜 주세요: {conflicts}")
    return refreshed, changes
