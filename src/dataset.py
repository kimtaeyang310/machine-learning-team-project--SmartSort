#1. datasey.py : 이미지 경로·숙도 라벨·분할 관리

"""SmartSort: 직접 촬영한 귤·토마토와 공개 바나나 데이터 통합."""

import csv
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 기존에 작성한 귤·토마토 라벨 파일을 계속 사용합니다.
LABELS_PATH = PROJECT_ROOT / "data" / "metadata" / "fruit_labels.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RIPENESS_LABELS = {
    "unripe": 0,    # 덜익음
    "ripe": 1,      # 익음
    "overripe": 2,  # 많이익음
}

LABEL_FIELDS = ["fruit", "fruit_id", "date", "ripeness"]

# 바나나 폴더 이름은 대소문자를 구분하지 않습니다.
BANANA_FOLDERS = {
    "unripe_banana": "unripe",
    "ripe_banana": "ripe",
    "overripe_banana": "overripe",
}

# 직접 촬영한 이미지의 파일명 규칙입니다.
# 20261004_tomato_T01_indoor_01.jpg
# 20261004_tomato_T01_indoor_dim_06.jpg
FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<fruit>mandarin|tomato)_"
    r"(?P<fruit_id>[A-Za-z]+\d+)_"
    r"(?P<light>.+)_"
    r"(?P<shot>\d+)$",
    re.IGNORECASE,
)


def find_banana_label(relative_path):
    """상위 폴더에서 바나나 숙도 폴더를 찾습니다."""
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
    """같은 원본에서 생성된 증강 이미지들을 하나의 그룹으로 묶습니다."""
    # 예:
    # Ripe_1_jpg.rf.abc123.jpg
    # Ripe_1_jpg.rf.def456.jpg
    # 위 두 이미지는 같은 ripe_1 그룹으로 묶습니다.
    #
    # 촬영 날짜와 실제 개체 정보는 추측하지 않습니다.
    original_stem = re.split(
        r"\.rf\.",
        image_path.stem,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # 원본 확장자가 파일명에 포함돼 있으면 제거합니다.
    original_stem = re.sub(
        r"(?:_|\.)"
        r"(?:jpg|jpeg|png|bmp)$",
        "",
        original_stem,
        flags=re.IGNORECASE,
    )

    if not original_stem:
        raise ValueError(
            f"원본 이미지 이름을 확인하세요: {image_path.name}"
        )

    return f"mendeley_banana_{original_stem.lower()}"


def scan_images(raw_dir=RAW_DIR):
    """이미지를 찾아 직접 촬영 데이터와 바나나 데이터를 구분합니다."""
    raw_dir = Path(raw_dir).resolve()

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"원본 이미지 폴더가 없습니다: {raw_dir}"
        )

    collected = []
    bananas = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = path.relative_to(raw_dir)
        banana_label = find_banana_label(relative_path)

        # 바나나는 폴더 이름으로 숙도 라벨을 연결합니다.
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

                # unripe는 기존 semi-ripe를 통일한 라벨입니다.
                "original_label": (
                    "semi-ripe"
                    if banana_label == "unripe"
                    else banana_label
                ),

                "source": "mendeley",
                "group_id": get_banana_group_id(path),
            })
            continue

        # 귤·토마토는 파일명에서 촬영 정보를 읽습니다.
        match = FILENAME_PATTERN.fullmatch(path.stem)

        if match is None:
            raise ValueError(
                f"대응하지 않는 이미지입니다: {relative_path}\n"
                "직접 촬영한 이미지는 지정한 파일명 규칙을 사용하고, "
                "바나나는 Unripe_banana / Ripe_banana / "
                "Overripe_banana 폴더 안에 넣어주세요."
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

            # 날짜가 달라도 같은 개체는 같은 그룹으로 묶습니다.
            "group_id": (
                f"self_collected_{info['fruit']}_{info['fruit_id']}"
            ),
        })

    if not collected and not bananas:
        raise ValueError("처리할 수 있는 이미지가 없습니다.")

    return collected, bananas


def update_label_template(samples, csv_path=LABELS_PATH):
    """귤·토마토의 기존 라벨을 보존하고 새 개체·날짜만 추가합니다."""
    csv_path = Path(csv_path)
    rows = {}

    # 기존 CSV가 있으면 작성한 라벨을 읽어 보존합니다.
    if csv_path.exists():
        with csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            missing = (
                set(LABEL_FIELDS)
                - set(reader.fieldnames or [])
            )

            if missing:
                raise ValueError(
                    f"CSV에 필요한 열이 없습니다: {sorted(missing)}"
                )

            for line_number, row in enumerate(reader, start=2):
                cleaned = {
                    field: (row.get(field) or "").strip()
                    for field in LABEL_FIELDS
                }

                if not all(
                    cleaned[field]
                    for field in LABEL_FIELDS[:-1]
                ):
                    raise ValueError(
                        f"CSV {line_number}행의 개체 정보를 채워주세요."
                    )

                cleaned["fruit"] = cleaned["fruit"].lower()
                cleaned["fruit_id"] = cleaned["fruit_id"].upper()
                cleaned["ripeness"] = cleaned["ripeness"].lower()

                datetime.strptime(
                    cleaned["date"], "%Y-%m-%d"
                )

                if cleaned["fruit"] not in {"mandarin", "tomato"}:
                    raise ValueError(
                        f"CSV {line_number}행의 fruit를 확인하세요."
                    )

                if (
                    cleaned["ripeness"]
                    and cleaned["ripeness"] not in RIPENESS_LABELS
                ):
                    raise ValueError(
                        f"CSV {line_number}행의 숙도 라벨을 확인하세요."
                    )

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

    # 새로 촬영한 개체·날짜 항목만 추가합니다.
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
            }
            added += 1

    # 변경 사항이 있을 때만 CSV를 저장합니다.
    if added or (samples and not csv_path.exists()):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = csv_path.with_suffix(".csv.tmp")

        with temporary_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=LABEL_FIELDS
            )
            writer.writeheader()
            writer.writerows(
                rows[key] for key in sorted(rows)
            )

        # 임시 파일을 완성한 후 기존 CSV를 교체합니다.
        temporary_path.replace(csv_path)

    print(f"귤·토마토 신규 라벨 항목: {added}개")

    return rows


def load_dataset(raw_dir=RAW_DIR, csv_path=LABELS_PATH):
    """귤·토마토와 바나나를 동일한 형식의 목록으로 합칩니다."""
    collected, bananas = scan_images(raw_dir)
    labels = update_label_template(collected, csv_path)

    # 직접 촬영 데이터 중 숙도가 입력되지 않은 항목을 찾습니다.
    pending = sorted({
        (
            sample["fruit"],
            sample["fruit_id"],
            sample["date"],
        )
        for sample in collected
        if not labels[
            (
                sample["fruit"],
                sample["fruit_id"],
                sample["date"],
            )
        ]["ripeness"]
    })

    if pending:
        examples = "\n".join(
            f"  {fruit}, {fruit_id}, {date}"
            for fruit, fruit_id, date in pending[:10]
        )

        raise ValueError(
            f"숙도 미입력 항목: {len(pending)}개\n"
            f"{examples}\n"
            f"{csv_path}의 ripeness 열을 작성한 뒤 다시 실행하세요."
        )

    # CSV의 숙도를 귤·토마토의 각 이미지에 연결합니다.
    for sample in collected:
        key = (
            sample["fruit"],
            sample["fruit_id"],
            sample["date"],
        )

        ripeness = labels[key]["ripeness"]

        sample["ripeness"] = ripeness
        sample["ripeness_label"] = RIPENESS_LABELS[ripeness]
        sample["original_label"] = ripeness

    dataset = collected + bananas

    # 같은 바나나 원본 그룹에 서로 다른 숙도가 연결됐는지 확인합니다.
    group_labels = defaultdict(set)

    for sample in dataset:
        group_labels[sample["group_id"]].add(
            sample["ripeness"]
        )

    conflicts = [
        group_id
        for group_id, values in group_labels.items()
        if (
            group_id.startswith("mendeley_")
            and len(values) > 1
        )
    ]

    if conflicts:
        raise ValueError(
            "같은 바나나 원본 그룹에 서로 다른 숙도가 있습니다: "
            + ", ".join(sorted(conflicts)[:10])
        )

    return sorted(
        dataset,
        key=lambda sample: sample["image_path"],
    )


def split_dataset(samples, test_size=0.2, seed=42):
    """과일별 그룹 분할로 동일 개체·원본의 데이터 누출을 막습니다."""
    if not 0 < test_size < 1:
        raise ValueError(
            "test_size는 0과 1 사이여야 합니다."
        )

    by_fruit = defaultdict(list)

    for sample in samples:
        by_fruit[sample["fruit"]].append(sample)

    train = []
    test = []
    rng = random.Random(seed)

    for fruit, fruit_samples in sorted(by_fruit.items()):
        groups = sorted({
            sample["group_id"]
            for sample in fruit_samples
        })

        if len(groups) < 2:
            raise ValueError(
                f"{fruit}: 분할하려면 서로 다른 그룹이 "
                "2개 이상 필요합니다."
            )

        rng.shuffle(groups)

        test_count = min(
            len(groups) - 1,
            max(1, round(len(groups) * test_size)),
        )

        test_groups = set(groups[:test_count])

        train.extend(
            sample
            for sample in fruit_samples
            if sample["group_id"] not in test_groups
        )

        test.extend(
            sample
            for sample in fruit_samples
            if sample["group_id"] in test_groups
        )

    return train, test


def print_summary(samples, name="전체"):
    """전체 사진 수와 과일별 숙도 분포를 출력합니다."""
    print(f"\n[{name}]")
    print(f"사진: {len(samples)}장")

    group_count = len({
        sample["group_id"]
        for sample in samples
    })
    print(f"개체/원본 그룹: {group_count}개")

    source_counts = Counter(
        sample["source"]
        for sample in samples
    )
    print(f"출처별: {dict(source_counts)}")

    for fruit in sorted({
        sample["fruit"]
        for sample in samples
    }):
        fruit_samples = [
            sample
            for sample in samples
            if sample["fruit"] == fruit
        ]

        counts = Counter(
            sample["ripeness"]
            for sample in fruit_samples
        )
        print(f"{fruit}: {dict(sorted(counts.items()))}")

        missing = set(RIPENESS_LABELS) - set(counts)

        if missing:
            print(
                f"  없는 숙도: {', '.join(sorted(missing))}"
            )


if __name__ == "__main__":
    try:
        dataset = load_dataset()
        print_summary(dataset)

        train_data, test_data = split_dataset(dataset)

        print_summary(train_data, "학습")
        print_summary(test_data, "테스트")

    except (ValueError, OSError) as error:
        print(f"\n확인할 내용:\n{error}")
        raise SystemExit(1)