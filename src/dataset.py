"""SmartSort 데이터셋 관리.

귤·토마토: fruit_labels.csv의 ripeness / split을 직접 입력합니다.
바나나: 폴더에서 숙도를 읽고, 숙도별 원본 그룹을 무작위 분할합니다.

실행:
    python src/dataset.py
또는:
    python -m src.dataset
"""

import csv
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_PATH = PROJECT_ROOT / "data" / "metadata" / "fruit_labels.csv"

# 바나나 원본 그룹의 약 20%를 Test로 배정합니다.
BANANA_TEST_SIZE = 0.2
BANANA_SPLIT_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RIPENESS_LABELS = {
    "unripe": 0,
    "ripe": 1,
    "overripe": 2,
}

VALID_SPLITS = {"train", "test"}

LABEL_FIELDS = [
    "fruit",
    "fruit_id",
    "date",
    "ripeness",
    "group_id",
    "split",
]

BASE_LABEL_FIELDS = [
    "fruit",
    "fruit_id",
    "date",
    "ripeness",
]

BANANA_FOLDERS = {
    "unripe_banana": "unripe",
    "ripe_banana": "ripe",
    "overripe_banana": "overripe",
}

FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<fruit>mandarin|tomato)_"
    r"(?P<fruit_id>[A-Za-z]+\d+)_"
    r"(?P<light>.+)_"
    r"(?P<shot>\d+)$",
    re.IGNORECASE,
)


def make_group_id(fruit_id, date):
    """직접 촬영한 과일의 개체+날짜 그룹을 만듭니다."""
    return f"{fruit_id.upper()}_{date.replace('-', '')}"


def find_banana_label(relative_path):
    """바나나 상위 폴더에서 숙도를 찾습니다."""
    labels = [
        BANANA_FOLDERS[part.lower()]
        for part in relative_path.parts[:-1]
        if part.lower() in BANANA_FOLDERS
    ]

    if len(labels) > 1:
        raise ValueError(
            f"숙도 폴더가 중복돼 있습니다: {relative_path}"
        )

    return labels[0] if labels else None


def get_banana_group_id(image_path):
    """같은 원본에서 생성된 .rf. 증강 이미지들을 하나로 묶습니다."""
    original_stem = re.split(
        r"\.rf\.",
        image_path.stem,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    original_stem = re.sub(
        r"(?:_|\.)(?:jpg|jpeg|png|bmp)$",
        "",
        original_stem,
        flags=re.IGNORECASE,
    )

    if not original_stem:
        raise ValueError(
            f"바나나 원본 이름을 확인하세요: {image_path.name}"
        )

    return f"mendeley_banana_{original_stem.lower()}"


def scan_images(raw_dir=RAW_DIR):
    """이미지를 검색합니다. 바나나 split은 이후에 배정합니다."""
    raw_dir = Path(raw_dir).resolve()

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"원본 이미지 폴더가 없습니다: {raw_dir}"
        )

    collected, bananas = [], []

    for path in sorted(raw_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in IMAGE_EXTENSIONS
        ):
            continue

        relative_path = path.relative_to(raw_dir)
        banana_label = find_banana_label(relative_path)

        # 바나나는 폴더 이름으로 숙도를 결정합니다.
        if banana_label is not None:
            bananas.append({
                "image_path": relative_path.as_posix(),
                "path": path,
                "fruit": "banana",
                "fruit_id": "",
                "date": "",
                "light": "",
                "shot": "",
                "ripeness": banana_label,
                "ripeness_label": RIPENESS_LABELS[banana_label],
                "original_label": (
                    "semi-ripe"
                    if banana_label == "unripe"
                    else banana_label
                ),
                "source": "mendeley",
                "group_id": get_banana_group_id(path),
                "split": "",
            })
            continue

        # 귤·토마토는 파일명에서 정보를 추출합니다.
        match = FILENAME_PATTERN.fullmatch(path.stem)

        if match is None:
            raise ValueError(
                f"대응하지 않는 이미지입니다: {relative_path}\n"
                "직접 촬영 파일명 예: "
                "20261008_tomato_T10_indoor_01.jpg"
            )

        info = match.groupdict()

        info["date"] = datetime.strptime(
            info["date"], "%Y%m%d"
        ).strftime("%Y-%m-%d")

        info["fruit"] = info["fruit"].lower()
        info["fruit_id"] = info["fruit_id"].upper()

        collected.append({
            **info,
            "image_path": relative_path.as_posix(),
            "path": path,
            "source": "self_collected",
            "group_id": make_group_id(
                info["fruit_id"], info["date"]
            ),
        })

    if not collected and not bananas:
        raise ValueError("처리할 수 있는 이미지가 없습니다.")

    return collected, bananas


def update_label_template(samples, csv_path=LABELS_PATH):
    """귤·토마토의 수동 라벨과 split을 보존하며 새 항목을 추가합니다."""
    csv_path = Path(csv_path)
    rows = {}
    rewrite_csv = False

    if csv_path.exists():
        with csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            fields = set(reader.fieldnames or [])

            missing = set(BASE_LABEL_FIELDS) - fields

            if missing:
                raise ValueError(
                    f"CSV에 필요한 열이 없습니다: {sorted(missing)}"
                )

            rewrite_csv = not set(LABEL_FIELDS).issubset(fields)

            for line_number, row in enumerate(reader, start=2):
                cleaned = {
                    key: (row.get(key) or "").strip()
                    for key in LABEL_FIELDS
                }

                if not all(
                    cleaned[key]
                    for key in ("fruit", "fruit_id", "date")
                ):
                    raise ValueError(
                        f"CSV {line_number}행의 개체 정보를 확인하세요."
                    )

                cleaned["fruit"] = cleaned["fruit"].lower()
                cleaned["fruit_id"] = cleaned["fruit_id"].upper()
                cleaned["ripeness"] = cleaned["ripeness"].lower()
                cleaned["split"] = cleaned["split"].lower()

                datetime.strptime(cleaned["date"], "%Y-%m-%d")

                if cleaned["fruit"] not in {"mandarin", "tomato"}:
                    raise ValueError(
                        f"CSV {line_number}행의 fruit를 확인하세요."
                    )

                if (
                    cleaned["ripeness"]
                    and cleaned["ripeness"] not in RIPENESS_LABELS
                ):
                    raise ValueError(
                        f"CSV {line_number}행의 ripeness를 확인하세요."
                    )

                if (
                    cleaned["split"]
                    and cleaned["split"] not in VALID_SPLITS
                ):
                    raise ValueError(
                        f"CSV {line_number}행의 split은 "
                        "train/test만 가능합니다."
                    )

                expected = make_group_id(
                    cleaned["fruit_id"], cleaned["date"]
                )

                if cleaned["group_id"] != expected:
                    cleaned["group_id"] = expected
                    rewrite_csv = True

                key = (
                    cleaned["fruit"],
                    cleaned["fruit_id"],
                    cleaned["date"],
                )

                if key in rows:
                    raise ValueError(
                        f"CSV에 중복 항목이 있습니다: {key}"
                    )

                rows[key] = cleaned

    added = 0

    for sample in samples:
        key = (
            sample["fruit"],
            sample["fruit_id"],
            sample["date"],
        )

        if key not in rows:
            rows[key] = {
                "fruit": sample["fruit"],
                "fruit_id": sample["fruit_id"],
                "date": sample["date"],
                "ripeness": "",
                "group_id": sample["group_id"],
                "split": "",
            }
            added += 1

        elif rows[key]["group_id"] != sample["group_id"]:
            rows[key]["group_id"] = sample["group_id"]
            rewrite_csv = True

    if added or rewrite_csv or (samples and not csv_path.exists()):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = csv_path.with_suffix(".csv.tmp")

        with temporary_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=LABEL_FIELDS
            )
            writer.writeheader()
            writer.writerows(rows[key] for key in sorted(rows))

        temporary_path.replace(csv_path)

    print(f"\n신규 라벨 항목: {added}개")

    if added:
        print(
            "fruit_labels.csv의 새 항목에 "
            "ripeness와 split을 입력하세요."
        )

    return rows


def assign_banana_splits(
    bananas,
    test_size=BANANA_TEST_SIZE,
    seed=BANANA_SPLIT_SEED,
):
    """숙도별로 원본 그룹을 섞어 Train/Test를 배정합니다."""
    if not 0 < test_size < 1:
        raise ValueError(
            "banana_test_size는 0과 1 사이여야 합니다."
        )

    # 하나의 원본 그룹에 여러 숙도가 연결되었는지 검사합니다.
    group_labels = defaultdict(set)

    for sample in bananas:
        group_labels[sample["group_id"]].add(sample["ripeness"])

    by_label = defaultdict(list)

    for group_id, labels in group_labels.items():
        if len(labels) != 1:
            raise ValueError(
                f"바나나 원본 그룹의 숙도가 충돌합니다: {group_id}"
            )

        label = next(iter(labels))

        if label not in RIPENESS_LABELS:
            raise ValueError(f"알 수 없는 바나나 숙도: {label}")

        by_label[label].append(group_id)

    assignments = {}

    # unripe / ripe / overripe 각각에서 그룹을 분할합니다.
    for label in sorted(by_label):
        groups = sorted(by_label[label])

        if len(groups) < 2:
            raise ValueError(
                f"바나나 {label}: Train/Test 분할에는 "
                "서로 다른 원본이 2개 이상 필요합니다. "
                "증강 이미지 수가 아닌 원본 수입니다."
            )

        # 입력 순서가 달라도 같은 데이터와 seed면 같은 분할입니다.
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(groups)

        # 각 숙도의 Train과 Test에 최소 한 그룹씩 남깁니다.
        test_count = min(
            len(groups) - 1,
            max(1, round(len(groups) * test_size)),
        )

        test_groups = set(groups[:test_count])

        for group_id in groups:
            assignments[group_id] = (
                "test" if group_id in test_groups else "train"
            )

    # 같은 원본의 증강 이미지들은 모두 같은 split을 받습니다.
    for sample in bananas:
        sample["split"] = assignments[sample["group_id"]]

    return bananas


def load_dataset(
    raw_dir=RAW_DIR,
    csv_path=LABELS_PATH,
    banana_test_size=BANANA_TEST_SIZE,
    banana_seed=BANANA_SPLIT_SEED,
):
    """수동 분할한 귤·토마토와 자동 분할한 바나나를 합칩니다."""
    collected, bananas = scan_images(raw_dir)
    labels = update_label_template(collected, csv_path)

    pending = {}

    for sample in collected:
        key = (
            sample["fruit"],
            sample["fruit_id"],
            sample["date"],
        )

        label = labels[key]

        missing = [
            field
            for field in ("ripeness", "split")
            if not label[field]
        ]

        if missing:
            pending[key] = missing
            continue

        sample.update({
            "ripeness": label["ripeness"],
            "ripeness_label": RIPENESS_LABELS[label["ripeness"]],
            "original_label": label["ripeness"],
            "split": label["split"],
        })

    if pending:
        details = "\n".join(
            f"{key}: {', '.join(fields)}"
            for key, fields in list(pending.items())[:20]
        )

        raise ValueError(
            f"라벨 미입력 항목:\n{details}\n"
            f"{csv_path}의 ripeness와 split을 작성하세요."
        )

    # 바나나만 자동 분할합니다.
    assign_banana_splits(
        bananas,
        test_size=banana_test_size,
        seed=banana_seed,
    )

    dataset = collected + bananas
    validate_groups(dataset)

    return sorted(
        dataset,
        key=lambda sample: sample["image_path"],
    )


def validate_groups(samples):
    """동일 그룹의 split 및 숙도 충돌을 검사합니다."""
    group_splits = defaultdict(set)
    group_labels = defaultdict(set)

    for sample in samples:
        if sample["split"] not in VALID_SPLITS:
            raise ValueError(
                f"잘못된 split: {sample['image_path']}"
            )

        group_splits[sample["group_id"]].add(sample["split"])
        group_labels[sample["group_id"]].add(sample["ripeness"])

    for group_id in group_splits:
        if len(group_splits[group_id]) > 1:
            raise ValueError(
                f"Train/Test에 같은 그룹이 있습니다: {group_id}"
            )

        if len(group_labels[group_id]) > 1:
            raise ValueError(
                f"같은 그룹에 서로 다른 숙도가 있습니다: {group_id}"
            )


def split_dataset(samples):
    """이미 배정된 split으로 나눕니다. 이 단계에서 재추첨하지 않습니다."""
    validate_groups(samples)

    train = [
        sample for sample in samples
        if sample["split"] == "train"
    ]

    test = [
        sample for sample in samples
        if sample["split"] == "test"
    ]

    if not train:
        raise ValueError("Train 데이터가 없습니다.")

    if not test:
        print(
            "\n현재 Test 데이터가 없습니다. "
            "학습만 진행할 수 있습니다."
        )

    return train, test


def print_summary(samples, name="전체"):
    """사진 수, 그룹 수, 출처 및 과일별 숙도 분포를 출력합니다."""
    print(f"\n{'=' * 60}\n[{name}]\n{'=' * 60}")
    print(f"사진: {len(samples)}장")
    print(f"그룹: {len({s['group_id'] for s in samples})}개")
    print(
        f"출처별: {dict(Counter(s['source'] for s in samples))}"
    )

    for fruit in sorted({s["fruit"] for s in samples}):
        subset = [
            sample for sample in samples
            if sample["fruit"] == fruit
        ]

        counts = Counter(s["ripeness"] for s in subset)
        groups = defaultdict(set)

        for sample in subset:
            groups[sample["ripeness"]].add(sample["group_id"])

        print(f"\n[{fruit}]")

        for label in RIPENESS_LABELS:
            print(
                f"  {label:<10}: "
                f"{counts[label]}장 / {len(groups[label])}그룹"
            )


if __name__ == "__main__":
    try:
        dataset = load_dataset()
        train_data, test_data = split_dataset(dataset)

        print_summary(dataset, "전체")
        print_summary(train_data, "Train")
        print_summary(test_data, "Test")

        print(
            f"\n분리 완료: "
            f"Train {len(train_data)}장 / "
            f"Test {len(test_data)}장"
        )

    except (ValueError, OSError, RuntimeError) as error:
        print(f"\n확인할 내용:\n{error}")
        raise SystemExit(1)