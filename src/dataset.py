"""
SmartSort - Dataset 관리

역할
------------------------------------------------------------
1. data/raw 이미지 검색
2. 파일명에서 과일 / 고유번호 / 날짜 자동 추출
3. group_id 자동 생성
4. 숙도(ripeness) 수동 라벨 관리
5. Train / Test(split) 수동 관리
6. 바나나 공개 데이터 통합
7. 동일 group_id의 Train/Test 누수 검사
8. 데이터 분포 출력


fruit_labels.csv 형식
------------------------------------------------------------

fruit,fruit_id,date,ripeness,group_id,split


예:

tomato,T01,2026-10-02,ripe,T01_20261002,train
tomato,T02,2026-10-02,unripe,T02_20261002,train
mandarin,M01,2026-10-04,ripe,M01_20261004,train


앞으로 새로 촬영한 데이터:

tomato,T10,2026-10-08,unripe,T10_20261008,test


자동 입력:
    fruit
    fruit_id
    date
    group_id

사람이 입력:
    ripeness
    split
"""

import csv
import re

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ============================================================
# 1. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fruit_labels.csv"
)


# ============================================================
# 2. 기본 설정
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


RIPENESS_LABELS = {
    "unripe": 0,
    "ripe": 1,
    "overripe": 2,
}


VALID_SPLITS = {
    "train",
    "test",
}


# ============================================================
# 3. CSV 컬럼
# ============================================================

LABEL_FIELDS = [
    "fruit",
    "fruit_id",
    "date",
    "ripeness",
    "group_id",
    "split",
]


# 기존 CSV와 호환하기 위한 최소 컬럼
BASE_LABEL_FIELDS = [
    "fruit",
    "fruit_id",
    "date",
    "ripeness",
]


# ============================================================
# 4. 바나나 폴더
# ============================================================

BANANA_FOLDERS = {
    "unripe_banana": "unripe",
    "ripe_banana": "ripe",
    "overripe_banana": "overripe",
}


# ============================================================
# 5. 직접 촬영 이미지 파일명 규칙
# ============================================================
#
# 예:
#
# 20261004_tomato_T01_indoor_01.jpg
#
# date
#     20261004
#
# fruit
#     tomato
#
# fruit_id
#     T01
#
# light
#     indoor
#
# shot
#     01
#
# ============================================================

FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<fruit>mandarin|tomato)_"
    r"(?P<fruit_id>[A-Za-z]+\d+)_"
    r"(?P<light>.+)_"
    r"(?P<shot>\d+)$",
    re.IGNORECASE,
)


# ============================================================
# 6. 직접 촬영 데이터 group_id 자동 생성
# ============================================================

def make_group_id(
    fruit_id,
    date,
):
    """
    파일명에서 얻은 fruit_id와 date를 이용해
    group_id를 자동으로 만듭니다.

    예:

    fruit_id = T10
    date = 2026-10-08

        ↓

    T10_20261008


    강사님 과제 조건:

    "촬영 날짜 단위 그룹 분할"

    을 반영하기 위해 날짜를 group_id에 포함합니다.
    """

    date_text = date.replace(
        "-",
        "",
    )

    return (
        f"{fruit_id.upper()}_"
        f"{date_text}"
    )


# ============================================================
# 7. 바나나 숙도 찾기
# ============================================================

def find_banana_label(
    relative_path,
):
    """
    바나나 이미지의 상위 폴더에서
    숙도를 찾습니다.

    예:

    Ripe_banana
        → ripe

    Unripe_banana
        → unripe
    """

    labels = [

        BANANA_FOLDERS[
            part.lower()
        ]

        for part
        in relative_path.parts[:-1]

        if part.lower()
        in BANANA_FOLDERS
    ]


    if len(labels) > 1:

        raise ValueError(
            "숙도 폴더가 중복돼 있습니다: "
            f"{relative_path}"
        )


    return (
        labels[0]
        if labels
        else None
    )


# ============================================================
# 8. 바나나 group_id 자동 생성
# ============================================================

def get_banana_group_id(
    image_path,
):
    """
    같은 원본 이미지에서 생성된 증강 이미지를
    같은 group_id로 묶습니다.

    예:

    Ripe_1_jpg.rf.abc123.jpg
    Ripe_1_jpg.rf.def456.jpg

        ↓

    mendeley_banana_ripe_1
    """

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
            "바나나 원본 이미지 이름을 "
            f"확인하세요: {image_path.name}"
        )


    return (
        "mendeley_banana_"
        f"{original_stem.lower()}"
    )


# ============================================================
# 9. 이미지 검색
# ============================================================

def scan_images(
    raw_dir=RAW_DIR,
):
    """
    data/raw 안의 모든 이미지를 검색합니다.

    직접 촬영:
        tomato
        mandarin

    공개 데이터:
        banana
    """

    raw_dir = Path(
        raw_dir
    ).resolve()


    if not raw_dir.is_dir():

        raise FileNotFoundError(
            "원본 이미지 폴더가 없습니다: "
            f"{raw_dir}"
        )


    collected = []

    bananas = []


    for path in sorted(
        raw_dir.rglob("*")
    ):

        # 파일이 아니면 제외
        if not path.is_file():
            continue


        # 이미지 확장자가 아니면 제외
        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue


        relative_path = (
            path.relative_to(
                raw_dir
            )
        )


        banana_label = (
            find_banana_label(
                relative_path
            )
        )


        # ====================================================
        # 바나나
        # ====================================================

        if banana_label is not None:

            group_id = (
                get_banana_group_id(
                    path
                )
            )


            bananas.append({

                "image_path":
                    relative_path.as_posix(),

                "path":
                    path,

                "fruit":
                    "banana",

                "fruit_id":
                    "",

                "date":
                    "",

                "light":
                    "",

                "shot":
                    "",

                "ripeness":
                    banana_label,

                "ripeness_label":
                    RIPENESS_LABELS[
                        banana_label
                    ],

                "original_label":
                    (
                        "semi-ripe"
                        if banana_label
                        == "unripe"
                        else banana_label
                    ),

                "source":
                    "mendeley",

                "group_id":
                    group_id,

                # 현재 바나나는 학습용으로 사용
                "split":
                    "train",
            })


            continue


        # ====================================================
        # 귤 / 토마토
        # ====================================================

        match = (
            FILENAME_PATTERN.fullmatch(
                path.stem
            )
        )


        if match is None:

            raise ValueError(

                "대응하지 않는 이미지입니다:\n"

                f"{relative_path}\n\n"

                "직접 촬영 이미지는 "
                "다음 형식을 사용해주세요.\n\n"

                "YYYYMMDD_fruit_ID_light_number.jpg\n\n"

                "예:\n"

                "20261008_tomato_T10_indoor_01.jpg"
            )


        info = (
            match.groupdict()
        )


        # ----------------------------------------------------
        # 날짜 변환
        #
        # 20261008
        #   ↓
        # 2026-10-08
        # ----------------------------------------------------

        info["date"] = (
            datetime.strptime(
                info["date"],
                "%Y%m%d",
            ).strftime(
                "%Y-%m-%d"
            )
        )


        # ----------------------------------------------------
        # fruit 정리
        # ----------------------------------------------------

        info["fruit"] = (
            info["fruit"].lower()
        )


        # ----------------------------------------------------
        # fruit_id 정리
        # ----------------------------------------------------

        info["fruit_id"] = (
            info["fruit_id"].upper()
        )


        # ----------------------------------------------------
        # group_id 자동 생성
        # ----------------------------------------------------

        group_id = (
            make_group_id(
                info["fruit_id"],
                info["date"],
            )
        )


        # ----------------------------------------------------
        # 데이터 추가
        # ----------------------------------------------------

        collected.append({

            **info,

            "image_path":
                relative_path.as_posix(),

            "path":
                path,

            "source":
                "self_collected",

            # 자동 생성
            "group_id":
                group_id,
        })


    if (
        not collected
        and not bananas
    ):

        raise ValueError(
            "처리할 수 있는 이미지가 없습니다."
        )


    return (
        collected,
        bananas,
    )


# ============================================================
# 10. fruit_labels.csv 생성 / 업데이트
# ============================================================

def update_label_template(
    samples,
    csv_path=LABELS_PATH,
):
    """
    fruit_labels.csv를 생성하거나 업데이트합니다.


    자동 입력되는 값:

        fruit
        fruit_id
        date
        group_id


    사람이 작성하는 값:

        ripeness
        split


    예:

    프로그램 자동 생성:

    tomato,T10,2026-10-08,,T10_20261008,


    사람이 작성:

    tomato,T10,2026-10-08,unripe,T10_20261008,test
    """

    csv_path = Path(
        csv_path
    )


    rows = {}


    # 기존 CSV를 새 형식으로 다시 저장해야 하는지
    rewrite_csv = False


    # ========================================================
    # 기존 CSV 읽기
    # ========================================================

    if csv_path.exists():

        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )


            fieldnames = (
                reader.fieldnames
                or []
            )


            # ------------------------------------------------
            # 최소 필수 열 확인
            # ------------------------------------------------

            missing = (
                set(BASE_LABEL_FIELDS)
                - set(fieldnames)
            )


            if missing:

                raise ValueError(
                    "CSV에 필요한 열이 없습니다: "
                    f"{sorted(missing)}"
                )


            # ------------------------------------------------
            # 새로운 컬럼이 기존 CSV에 없다면
            # 새 형식으로 다시 저장
            # ------------------------------------------------

            if (
                "group_id"
                not in fieldnames
                or
                "split"
                not in fieldnames
            ):

                rewrite_csv = True


            # ------------------------------------------------
            # 기존 행 읽기
            # ------------------------------------------------

            for (
                line_number,
                row,
            ) in enumerate(
                reader,
                start=2,
            ):

                cleaned = {

                    "fruit":
                        (
                            row.get("fruit")
                            or ""
                        ).strip(),

                    "fruit_id":
                        (
                            row.get("fruit_id")
                            or ""
                        ).strip(),

                    "date":
                        (
                            row.get("date")
                            or ""
                        ).strip(),

                    "ripeness":
                        (
                            row.get("ripeness")
                            or ""
                        ).strip(),

                    "group_id":
                        (
                            row.get("group_id")
                            or ""
                        ).strip(),

                    "split":
                        (
                            row.get("split")
                            or ""
                        ).strip(),

                }


                # ------------------------------------------------
                # 기본 정보 확인
                # ------------------------------------------------

                if not all([

                    cleaned["fruit"],

                    cleaned["fruit_id"],

                    cleaned["date"],

                ]):

                    raise ValueError(
                        f"CSV {line_number}행의 "
                        "fruit / fruit_id / date를 "
                        "확인하세요."
                    )


                cleaned["fruit"] = (
                    cleaned[
                        "fruit"
                    ].lower()
                )


                cleaned["fruit_id"] = (
                    cleaned[
                        "fruit_id"
                    ].upper()
                )


                cleaned["ripeness"] = (
                    cleaned[
                        "ripeness"
                    ].lower()
                )


                cleaned["split"] = (
                    cleaned[
                        "split"
                    ].lower()
                )


                # ------------------------------------------------
                # 날짜 확인
                # ------------------------------------------------

                datetime.strptime(
                    cleaned["date"],
                    "%Y-%m-%d",
                )


                # ------------------------------------------------
                # 과일 확인
                # ------------------------------------------------

                if (
                    cleaned["fruit"]
                    not in {
                        "mandarin",
                        "tomato",
                    }
                ):

                    raise ValueError(
                        f"CSV {line_number}행의 "
                        "fruit를 확인하세요."
                    )


                # ------------------------------------------------
                # 숙도 확인
                # ------------------------------------------------

                if (
                    cleaned["ripeness"]
                    and
                    cleaned["ripeness"]
                    not in RIPENESS_LABELS
                ):

                    raise ValueError(
                        f"CSV {line_number}행의 "
                        "ripeness를 확인하세요."
                    )


                # ------------------------------------------------
                # split 확인
                # ------------------------------------------------

                if (
                    cleaned["split"]
                    and
                    cleaned["split"]
                    not in VALID_SPLITS
                ):

                    raise ValueError(
                        f"CSV {line_number}행의 "
                        "split은 train 또는 test만 "
                        "사용할 수 있습니다."
                    )


                # ------------------------------------------------
                # group_id 자동 생성
                # ------------------------------------------------

                expected_group_id = (
                    make_group_id(
                        cleaned["fruit_id"],
                        cleaned["date"],
                    )
                )


                # 기존 CSV에 group_id가 없거나 비어 있으면
                # 자동으로 채움
                if (
                    not cleaned[
                        "group_id"
                    ]
                ):

                    cleaned[
                        "group_id"
                    ] = (
                        expected_group_id
                    )

                    rewrite_csv = True


                # 기존 group_id가 자동 생성 값과 다르면
                # 자동값으로 수정
                elif (
                    cleaned["group_id"]
                    != expected_group_id
                ):

                    print(
                        "\n"
                        f"CSV {line_number}행 "
                        "group_id 자동 수정:"
                    )

                    print(
                        f"  기존: "
                        f"{cleaned['group_id']}"
                    )

                    print(
                        f"  변경: "
                        f"{expected_group_id}"
                    )


                    cleaned["group_id"] = (
                        expected_group_id
                    )

                    rewrite_csv = True


                # ------------------------------------------------
                # CSV 행 식별 key
                # ------------------------------------------------

                key = (

                    cleaned["fruit"],

                    cleaned["fruit_id"],

                    cleaned["date"],

                )


                if key in rows:

                    raise ValueError(
                        "CSV에 중복 항목이 있습니다: "
                        f"{key}"
                    )


                rows[key] = (
                    cleaned
                )


    # ========================================================
    # 새로운 이미지 데이터 발견
    # ========================================================

    added = 0


    for sample in samples:

        key = (

            sample["fruit"],

            sample["fruit_id"],

            sample["date"],

        )


        # ----------------------------------------------------
        # 새 행 생성
        # ----------------------------------------------------

        if key not in rows:

            rows[key] = {

                # 자동
                "fruit":
                    sample["fruit"],

                # 자동
                "fruit_id":
                    sample["fruit_id"],

                # 자동
                "date":
                    sample["date"],

                # 사람이 입력
                "ripeness":
                    "",

                # 자동
                "group_id":
                    sample["group_id"],

                # 사람이 입력
                "split":
                    "",
            }


            added += 1


        # ----------------------------------------------------
        # 기존 CSV의 group_id도 항상
        # 파일명 기준 자동 group_id와 맞춤
        # ----------------------------------------------------

        else:

            expected_group_id = (
                sample["group_id"]
            )


            if (
                rows[key]["group_id"]
                != expected_group_id
            ):

                rows[key]["group_id"] = (
                    expected_group_id
                )

                rewrite_csv = True


    # ========================================================
    # CSV 저장
    # ========================================================

    if (
        added
        or rewrite_csv
        or (
            samples
            and not csv_path.exists()
        )
    ):

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        temporary_path = (
            csv_path.with_suffix(
                ".csv.tmp"
            )
        )


        with temporary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=LABEL_FIELDS,
            )


            writer.writeheader()


            writer.writerows(

                rows[key]

                for key
                in sorted(rows)

            )


        temporary_path.replace(
            csv_path
        )


    print(
        "\n"
        f"신규 라벨 항목: "
        f"{added}개"
    )


    if added:

        print(
            "\n새 데이터가 발견되었습니다."
        )

        print(
            "fruit / fruit_id / date / group_id는 "
            "자동으로 입력했습니다."
        )

        print(
            "ripeness와 split만 "
            "확인해서 입력해주세요."
        )


    return rows


# ============================================================
# 11. 데이터셋 불러오기
# ============================================================

def load_dataset(
    raw_dir=RAW_DIR,
    csv_path=LABELS_PATH,
):
    """
    귤·토마토와 바나나를
    하나의 데이터 목록으로 합칩니다.
    """

    collected, bananas = (
        scan_images(
            raw_dir
        )
    )


    labels = (
        update_label_template(
            collected,
            csv_path,
        )
    )


    # ========================================================
    # 사람이 입력해야 하는 값 확인
    #
    # group_id는 자동이므로 검사 대상에서 제외
    # ========================================================

    pending = {}


    for sample in collected:

        key = (

            sample["fruit"],

            sample["fruit_id"],

            sample["date"],

        )


        label = (
            labels[key]
        )


        missing_fields = []


        # 숙도
        if not label[
            "ripeness"
        ]:

            missing_fields.append(
                "ripeness"
            )


        # Train / Test
        if not label[
            "split"
        ]:

            missing_fields.append(
                "split"
            )


        if missing_fields:

            pending[key] = (
                missing_fields
            )


    # ========================================================
    # 미입력값이 있다면 사용자에게 안내
    # ========================================================

    if pending:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "라벨 입력이 필요합니다."
        )

        print(
            "=" * 60
        )


        for (
            key,
            missing_fields,
        ) in list(
            pending.items()
        )[:20]:

            fruit, fruit_id, date = key


            group_id = (
                make_group_id(
                    fruit_id,
                    date,
                )
            )


            print(
                f"\n"
                f"{fruit} / "
                f"{fruit_id} / "
                f"{date}"
            )


            print(
                "  group_id:"
                f" {group_id} "
                "(자동)"
            )


            print(
                "  입력 필요:"
                f" {', '.join(missing_fields)}"
            )


        raise ValueError(

            "\nfruit_labels.csv의 "
            "ripeness와 split을 "
            "작성한 뒤 다시 실행하세요."

        )


    # ========================================================
    # CSV 라벨을 이미지 데이터에 연결
    # ========================================================

    for sample in collected:

        key = (

            sample["fruit"],

            sample["fruit_id"],

            sample["date"],

        )


        label = (
            labels[key]
        )


        sample["ripeness"] = (
            label[
                "ripeness"
            ]
        )


        sample["ripeness_label"] = (
            RIPENESS_LABELS[
                label["ripeness"]
            ]
        )


        sample["original_label"] = (
            label[
                "ripeness"
            ]
        )


        # group_id는 파일명 기준 자동 생성
        sample["group_id"] = (
            make_group_id(
                sample["fruit_id"],
                sample["date"],
            )
        )


        sample["split"] = (
            label[
                "split"
            ]
        )


    dataset = (
        collected
        + bananas
    )


    # ========================================================
    # group 검증
    # ========================================================

    validate_groups(
        dataset
    )


    return sorted(

        dataset,

        key=lambda sample:
            sample[
                "image_path"
            ],

    )


# ============================================================
# 12. group 검증
# ============================================================

def validate_groups(
    samples,
):
    """
    검사 내용

    1. 동일 group_id가 Train/Test에 동시에 존재하는지
    2. 동일 group_id에 서로 다른 숙도가 있는지

    데이터 누수를 방지합니다.
    """

    group_splits = (
        defaultdict(set)
    )


    group_ripeness = (
        defaultdict(set)
    )


    for sample in samples:

        group_id = (
            sample[
                "group_id"
            ]
        )


        group_splits[
            group_id
        ].add(
            sample[
                "split"
            ]
        )


        group_ripeness[
            group_id
        ].add(
            sample[
                "ripeness"
            ]
        )


    # ========================================================
    # Train/Test 누수 검사
    # ========================================================

    split_conflicts = [

        group_id

        for group_id, values
        in group_splits.items()

        if (
            "train" in values
            and
            "test" in values
        )

    ]


    if split_conflicts:

        raise ValueError(

            "같은 group_id가 "
            "Train과 Test에 동시에 있습니다:\n"

            + "\n".join(
                sorted(
                    split_conflicts
                )[:20]
            )

        )


    # ========================================================
    # 숙도 충돌 검사
    # ========================================================

    ripeness_conflicts = [

        group_id

        for group_id, values
        in group_ripeness.items()

        if len(values) > 1

    ]


    if ripeness_conflicts:

        raise ValueError(

            "같은 group_id에 "
            "서로 다른 숙도가 연결되어 있습니다:\n"

            + "\n".join(
                sorted(
                    ripeness_conflicts
                )[:20]
            )

        )


# ============================================================
# 13. Train / Test 분리
# ============================================================

def split_dataset(
    samples,
):
    """
    CSV의 split 값을 기준으로
    Train / Test를 나눕니다.

    랜덤 분할은 하지 않습니다.

    train
        → 현재까지 촬영한 기존 데이터

    test
        → 앞으로 새 날짜에 촬영할
          새로운 개체 데이터
    """

    train = [

        sample

        for sample in samples

        if (
            sample["split"]
            == "train"
        )

    ]


    test = [

        sample

        for sample in samples

        if (
            sample["split"]
            == "test"
        )

    ]


    # ========================================================
    # Train 확인
    # ========================================================

    if not train:

        raise ValueError(
            "Train 데이터가 없습니다."
        )


    # ========================================================
    # Test가 아직 없다면 경고만 출력
    # ========================================================

    if not test:

        print(
            "\n주의:"
        )

        print(
            "현재 Test 데이터가 없습니다."
        )

        print(
            "앞으로 새로운 날짜에 촬영한 "
            "새로운 과일을 test로 지정하세요."
        )


    # ========================================================
    # Train/Test group_id 최종 확인
    # ========================================================

    train_groups = {

        sample["group_id"]

        for sample in train

    }


    test_groups = {

        sample["group_id"]

        for sample in test

    }


    overlap = (

        train_groups
        & test_groups

    )


    if overlap:

        raise RuntimeError(

            "Train/Test 사이에 "
            "동일한 group_id가 존재합니다:\n"

            + "\n".join(
                sorted(
                    overlap
                )
            )

        )


    return (
        train,
        test,
    )


# ============================================================
# 14. 데이터 요약
# ============================================================

def print_summary(
    samples,
    name="전체",
):
    """
    데이터 현황 출력

    - 사진 수
    - group 수
    - 출처
    - 과일별 숙도 사진 수
    - 과일별 숙도 group 수
    """

    print(
        "\n"
        + "=" * 60
    )


    print(
        f"[{name}]"
    )


    print(
        "=" * 60
    )


    # ========================================================
    # 사진 수
    # ========================================================

    print(
        f"사진: "
        f"{len(samples)}장"
    )


    # ========================================================
    # group 수
    # ========================================================

    groups = {

        sample["group_id"]

        for sample in samples

    }


    print(
        f"그룹: "
        f"{len(groups)}개"
    )


    # ========================================================
    # 출처
    # ========================================================

    source_counts = Counter(

        sample["source"]

        for sample
        in samples

    )


    print(
        "출처별: "
        f"{dict(source_counts)}"
    )


    # ========================================================
    # 과일 종류
    # ========================================================

    fruits = sorted({

        sample["fruit"]

        for sample
        in samples

    })


    for fruit in fruits:

        fruit_samples = [

            sample

            for sample
            in samples

            if (
                sample["fruit"]
                == fruit
            )

        ]


        print(
            f"\n[{fruit}]"
        )


        # ----------------------------------------------------
        # 사진 기준 숙도
        # ----------------------------------------------------

        image_counts = Counter(

            sample["ripeness"]

            for sample
            in fruit_samples

        )


        print(
            "사진 기준 숙도:"
        )


        for ripeness in (
            RIPENESS_LABELS
        ):

            print(

                f"  "
                f"{ripeness:<10}: "
                f"{image_counts.get(ripeness, 0)}장"

            )


        # ----------------------------------------------------
        # group 기준 숙도
        # ----------------------------------------------------

        group_sets = (
            defaultdict(set)
        )


        for sample in fruit_samples:

            group_sets[
                sample["ripeness"]
            ].add(
                sample[
                    "group_id"
                ]
            )


        print(
            "group 기준 숙도:"
        )


        for ripeness in (
            RIPENESS_LABELS
        ):

            print(

                f"  "
                f"{ripeness:<10}: "
                f"{len(group_sets[ripeness])}개"

            )


# ============================================================
# 15. 직접 실행
# ============================================================

if __name__ == "__main__":

    try:

        # ----------------------------------------------------
        # 데이터 불러오기
        # ----------------------------------------------------

        dataset = (
            load_dataset()
        )


        # ----------------------------------------------------
        # 전체 데이터 현황
        # ----------------------------------------------------

        print_summary(
            dataset,
            "전체",
        )


        # ----------------------------------------------------
        # Train / Test 분리
        # ----------------------------------------------------

        train_data, test_data = (
            split_dataset(
                dataset
            )
        )


        # ----------------------------------------------------
        # Train 현황
        # ----------------------------------------------------

        print_summary(
            train_data,
            "Train",
        )


        # ----------------------------------------------------
        # Test 현황
        # ----------------------------------------------------

        if test_data:

            print_summary(
                test_data,
                "Test",
            )


        # ----------------------------------------------------
        # 최종 결과
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )


        print(
            "Train/Test 분리 완료"
        )


        print(
            "=" * 60
        )


        print(
            f"Train: "
            f"{len(train_data)}장"
        )


        print(
            f"Test : "
            f"{len(test_data)}장"
        )


    except (
        ValueError,
        OSError,
        RuntimeError,
    ) as error:

        print(
            "\n확인할 내용:\n"
            f"{error}"
        )

        raise SystemExit(1)