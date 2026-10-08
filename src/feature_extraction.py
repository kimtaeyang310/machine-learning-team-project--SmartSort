# ============================================================
# 3. feature_extraction : 과일 영역의 HSV 평균·색상 비율 등 계산
# ============================================================

"""
SmartSort - 과일 숙도 판별용 특징 추출

하는 일
1. dataset.py에서 이미지 목록 불러오기
2. preprocessing.py로 과일 영역 분리
3. HSV 기반 특징 추출
4. 성공한 결과를 fruit_features.csv에 저장
5. 실패한 이미지와 실패 이유를 feature_extraction_failed.csv에 저장
6. 전처리에 실패한 이미지를 outputs/preprocessing_failed에 저장

실행 방법
프로젝트 루트에서:

python -m src.feature_extraction
"""
import sys
from pathlib import Path

# 파일을 직접 실행할 때 프로젝트 루트를 모듈 검색 경로에 추가
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import (
    preprocess,
    save_failed_image,
)

from src.dataset import load_dataset


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

FEATURE_CSV = FEATURE_DIR / "fruit_features.csv"

FAILED_CSV = FEATURE_DIR / "feature_extraction_failed.csv"


# ------------------------------------------------------------
# 전처리 실패 이미지 저장 폴더
# ------------------------------------------------------------

PREPROCESSING_FAILED_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_failed"
)


# ============================================================
# 2. HSV 색상 범위
# ============================================================

# OpenCV HSV 기준
# H : 0~179
# S : 0~255
# V : 0~255

COLOR_RANGES = {
    "red": [
        (0, 10),
        (170, 179),
    ],
    "orange": [
        (10, 25),
    ],
    "yellow": [
        (25, 40),
    ],
    "green": [
        (40, 85),
    ],
    "blue": [
        (85, 130),
    ],
    "purple": [
        (130, 160),
    ],
}


# ============================================================
# 3. 과일 영역의 픽셀 가져오기
# ============================================================

def get_fruit_pixels(hsv, mask):
    """
    mask가 과일 영역이라고 판단한 픽셀만 가져온다.

    hsv
        전체 이미지의 HSV

    mask
        과일 영역이면 255, 아니면 0

    반환
        과일 영역에 해당하는 HSV 픽셀들
    """

    fruit_pixels = hsv[mask > 0]

    if len(fruit_pixels) == 0:
        raise ValueError(
            "과일 영역의 픽셀이 없습니다."
        )

    return fruit_pixels


# ============================================================
# 4. HSV 평균
# ============================================================

def calculate_hsv_mean(hsv, mask):
    """
    과일 영역의 H, S, V 평균을 계산한다.
    """

    fruit_pixels = get_fruit_pixels(
        hsv,
        mask,
    )

    h_mean = float(
        np.mean(fruit_pixels[:, 0])
    )

    s_mean = float(
        np.mean(fruit_pixels[:, 1])
    )

    v_mean = float(
        np.mean(fruit_pixels[:, 2])
    )

    return {
        "h_mean": h_mean,
        "s_mean": s_mean,
        "v_mean": v_mean,
    }


# ============================================================
# 5. HSV 표준편차
# ============================================================

def calculate_hsv_std(hsv, mask):
    """
    과일 영역의 H, S, V 표준편차를 계산한다.

    표준편차가 크다
        → 과일 내부의 색상 변화가 크다

    표준편차가 작다
        → 과일 내부의 색상이 비교적 균일하다
    """

    fruit_pixels = get_fruit_pixels(
        hsv,
        mask,
    )

    h_std = float(
        np.std(fruit_pixels[:, 0])
    )

    s_std = float(
        np.std(fruit_pixels[:, 1])
    )

    v_std = float(
        np.std(fruit_pixels[:, 2])
    )

    return {
        "h_std": h_std,
        "s_std": s_std,
        "v_std": v_std,
    }


# ============================================================
# 6. HSV 히스토그램
# ============================================================

def calculate_hsv_histogram(hsv, mask):
    """
    과일 영역의 HSV 히스토그램을 계산한다.

    H : 12개 구간
    S : 8개 구간
    V : 8개 구간

    총 28개의 특징이 만들어진다.
    """

    fruit_pixels = get_fruit_pixels(
        hsv,
        mask,
    )

    features = {}

    # --------------------------------------------------------
    # H 히스토그램
    # --------------------------------------------------------

    h_hist, _ = np.histogram(
        fruit_pixels[:, 0],
        bins=12,
        range=(0, 180),
    )

    # --------------------------------------------------------
    # S 히스토그램
    # --------------------------------------------------------

    s_hist, _ = np.histogram(
        fruit_pixels[:, 1],
        bins=8,
        range=(0, 256),
    )

    # --------------------------------------------------------
    # V 히스토그램
    # --------------------------------------------------------

    v_hist, _ = np.histogram(
        fruit_pixels[:, 2],
        bins=8,
        range=(0, 256),
    )

    # --------------------------------------------------------
    # 정규화
    # --------------------------------------------------------

    total_pixels = len(fruit_pixels)

    h_hist = h_hist / total_pixels
    s_hist = s_hist / total_pixels
    v_hist = v_hist / total_pixels

    # --------------------------------------------------------
    # Dictionary에 저장
    # --------------------------------------------------------

    for i, value in enumerate(h_hist):

        features[
            f"h_hist_{i}"
        ] = float(value)

    for i, value in enumerate(s_hist):

        features[
            f"s_hist_{i}"
        ] = float(value)

    for i, value in enumerate(v_hist):

        features[
            f"v_hist_{i}"
        ] = float(value)

    return features


# ============================================================
# 7. 색상 비율
# ============================================================

def calculate_color_ratios(hsv, mask):
    """
    과일 영역에서

    빨강
    주황
    노랑
    초록
    파랑
    보라

    각각의 비율을 계산한다.

    너무 어둡거나 채도가 낮은 픽셀은
    색상 분류에서 제외한다.
    """

    fruit_pixels = get_fruit_pixels(
        hsv,
        mask,
    )

    H = fruit_pixels[:, 0]
    S = fruit_pixels[:, 1]
    V = fruit_pixels[:, 2]

    # --------------------------------------------------------
    # 너무 흐리거나 어두운 픽셀 제외
    # --------------------------------------------------------

    valid = (
        (S >= 40)
        & (V >= 30)
    )

    valid_h = H[valid]

    features = {}

    # --------------------------------------------------------
    # 유효한 색상 픽셀이 하나도 없는 경우
    # --------------------------------------------------------

    if len(valid_h) == 0:

        for color_name in COLOR_RANGES:

            features[
                f"{color_name}_ratio"
            ] = 0.0

        return features

    # --------------------------------------------------------
    # 색상별 비율 계산
    # --------------------------------------------------------

    total = len(valid_h)

    for color_name, ranges in (
        COLOR_RANGES.items()
    ):

        color_count = 0

        for lower, upper in ranges:

            color_count += np.sum(
                (valid_h >= lower)
                & (valid_h <= upper)
            )

        ratio = color_count / total

        features[
            f"{color_name}_ratio"
        ] = float(ratio)

    return features


# ============================================================
# 8. 어두운 영역 비율
# ============================================================

def calculate_dark_ratio(hsv, mask):
    """
    V 값이 60보다 작은 픽셀의 비율.

    숙도가 높아지면서 갈변이나 어두운 부분이 증가한다면
    숙도 판별에 도움이 될 수 있다.
    """

    fruit_pixels = get_fruit_pixels(
        hsv,
        mask,
    )

    V = fruit_pixels[:, 2]

    dark_ratio = np.mean(V < 60)

    return {
        "dark_ratio": float(
            dark_ratio
        ),
    }


# ============================================================
# 9. 낮은 채도 비율
# ============================================================

def calculate_low_saturation_ratio(
    hsv,
    mask,
):
    """
    S 값이 40보다 작은 픽셀의 비율.

    색이 선명하지 않은 영역이 얼마나 있는지 나타낸다.
    """

    fruit_pixels = get_fruit_pixels(
        hsv,
        mask,
    )

    S = fruit_pixels[:, 1]

    low_saturation_ratio = np.mean(
        S < 40
    )

    return {
        "low_saturation_ratio": float(
            low_saturation_ratio
        ),
    }


# ============================================================
# 10. 전체 특징 추출
# ============================================================

def extract_features(processed):
    """
    preprocessing.py의 결과에서
    머신러닝에 사용할 모든 특징을 추출한다.
    """

    hsv = processed["hsv"]

    mask = processed["mask"]

    features = {}

    # --------------------------------------------------------
    # HSV 평균
    # --------------------------------------------------------

    features.update(
        calculate_hsv_mean(
            hsv,
            mask,
        )
    )

    # --------------------------------------------------------
    # HSV 표준편차
    # --------------------------------------------------------

    features.update(
        calculate_hsv_std(
            hsv,
            mask,
        )
    )

    # --------------------------------------------------------
    # HSV 히스토그램
    # --------------------------------------------------------

    features.update(
        calculate_hsv_histogram(
            hsv,
            mask,
        )
    )

    # --------------------------------------------------------
    # 색상 비율
    # --------------------------------------------------------

    features.update(
        calculate_color_ratios(
            hsv,
            mask,
        )
    )

    # --------------------------------------------------------
    # 어두운 영역 비율
    # --------------------------------------------------------

    features.update(
        calculate_dark_ratio(
            hsv,
            mask,
        )
    )

    # --------------------------------------------------------
    # 낮은 채도 영역 비율
    # --------------------------------------------------------

    features.update(
        calculate_low_saturation_ratio(
            hsv,
            mask,
        )
    )

    # --------------------------------------------------------
    # 과일 영역 비율
    # --------------------------------------------------------

    features["area_ratio"] = float(
        processed["area_ratio"]
    )

    # --------------------------------------------------------
    # 과일 픽셀 개수
    # --------------------------------------------------------

    features[
        "fruit_pixel_count"
    ] = int(
        np.sum(mask > 0)
    )

    return features


# ============================================================
# 11. 이미지 1장의 특징 추출
# ============================================================

def extract_sample_features(sample):
    """
    dataset.py에서 가져온 이미지 1장에 대해

    이미지 읽기
        ↓
    전처리
        ↓
    과일 영역 분리
        ↓
    HSV 특징 추출

    과정을 수행한다.
    """

    processed = preprocess(
        sample["path"],
        size=320,
        k=3,
        min_saturation=40,
        denoise=True,
    )

    features = extract_features(
        processed
    )

    # --------------------------------------------------------
    # 이미지 정보 + 특징을 하나의 dictionary로 합치기
    # --------------------------------------------------------

    result = {
        "image_path": sample["image_path"],
        "fruit": sample["fruit"],
        "fruit_id": sample["fruit_id"],
        "date": sample["date"],
        "ripeness": sample["ripeness"],
        "ripeness_label": sample["ripeness_label"],
        "source": sample["source"],
        "group_id": sample["group_id"],
    }

    result.update(features)

    return result


# ============================================================
# 12. 전체 데이터 특징 추출
# ============================================================

def build_feature_dataset(samples):
    """
    전체 이미지에서 특징을 추출한다.

    성공한 이미지
        → feature_rows

    실패한 이미지
        → failed_samples

    두 개를 동시에 반환한다.
    """

    feature_rows = []

    failed_samples = []

    total = len(samples)

    print()
    print("=" * 70)
    print("특징 추출 시작")
    print("=" * 70)

    print(
        f"전체 이미지 수: {total}"
    )

    print()

    # --------------------------------------------------------
    # 이미지 하나씩 처리
    # --------------------------------------------------------

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        try:

            features = extract_sample_features(
                sample
            )

            feature_rows.append(
                features
            )

            print(
                f"[{index}/{total}] 완료: "
                f"{sample['image_path']}"
            )

        except Exception as error:

            print(
                f"[{index}/{total}] 실패: "
                f"{sample['image_path']}"
            )

            print(
                f"    이유: {error}"
            )

            # ------------------------------------------------
            # 실패 이미지 저장 경로 생성
            # ------------------------------------------------

            relative_path = Path(
                sample["image_path"]
            )

            failed_output_path = (
                PREPROCESSING_FAILED_DIR
                / relative_path.parent
                / f"{relative_path.stem}_failed.png"
            )

            # ------------------------------------------------
            # 실패 이미지 + 과일 + 숙도 + 실패 원인 저장
            # ------------------------------------------------

            try:

                save_failed_image(
                    image_path=sample["path"],
                    output_path=failed_output_path,
                    fruit=sample["fruit"],
                    ripeness=sample["ripeness"],
                    error_message=str(error),
                )

                print(
                    f"    실패 이미지 저장: "
                    f"{failed_output_path}"
                )

            except Exception as save_error:

                print(
                    f"    실패 이미지 저장 실패: "
                    f"{save_error}"
                )

            # ------------------------------------------------
            # 실패 CSV에 기록
            # ------------------------------------------------

            failed_samples.append(
                {
                    "image_path": sample["image_path"],
                    "fruit": sample["fruit"],
                    "fruit_id": sample["fruit_id"],
                    "date": sample["date"],
                    "ripeness": sample["ripeness"],
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # 모든 이미지 처리가 끝난 후 결과 반환
    # --------------------------------------------------------

    return (
        feature_rows,
        failed_samples,
    )


# ============================================================
# 13. 특징 CSV 저장
# ============================================================

def save_features(
    feature_rows,
    output_path=FEATURE_CSV,
):
    """
    성공적으로 특징 추출된 데이터를
    fruit_features.csv로 저장한다.
    """

    if not feature_rows:

        print()
        print(
            "저장할 특징 데이터가 없습니다."
        )

        return

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 모든 컬럼 이름 가져오기
    # --------------------------------------------------------

    fieldnames = list(
        feature_rows[0].keys()
    )

    # --------------------------------------------------------
    # CSV 저장
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            feature_rows
        )

    print()
    print("=" * 70)
    print("특징 데이터 저장 완료")
    print("=" * 70)

    print(
        f"파일: {output_path}"
    )

    print(
        f"저장된 이미지 수: "
        f"{len(feature_rows)}"
    )


# ============================================================
# 14. 실패 이미지 CSV 저장
# ============================================================

def save_failed_samples(
    failed_samples,
    output_path=FAILED_CSV,
):
    """
    특징 추출에 실패한 이미지 목록을
    feature_extraction_failed.csv로 저장한다.
    """

    # --------------------------------------------------------
    # 실패한 이미지가 없는 경우
    # --------------------------------------------------------

    if not failed_samples:

        print()
        print(
            "실패한 이미지가 없습니다."
        )

        return

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "image_path",
        "fruit",
        "fruit_id",
        "date",
        "ripeness",
        "error",
    ]

    # --------------------------------------------------------
    # CSV 저장
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            failed_samples
        )

    print()
    print("=" * 70)
    print("실패 이미지 목록 저장 완료")
    print("=" * 70)

    print(
        f"파일: {output_path}"
    )

    print(
        f"실패한 이미지 수: "
        f"{len(failed_samples)}"
    )


# ============================================================
# 15. 실패 원인 요약
# ============================================================

def print_failure_summary(
    failed_samples,
):
    """
    실패한 이미지가 어떤 과일에서 많이 발생했는지
    간단하게 요약한다.
    """

    if not failed_samples:
        return

    print()
    print("=" * 70)
    print("실패 이미지 요약")
    print("=" * 70)

    # --------------------------------------------------------
    # 과일별 실패 개수
    # --------------------------------------------------------

    fruit_counts = defaultdict(int)

    for sample in failed_samples:

        fruit_counts[
            sample["fruit"]
        ] += 1

    print()
    print("[과일별 실패 개수]")

    for fruit, count in sorted(
        fruit_counts.items()
    ):

        print(
            f"  {fruit}: {count}장"
        )

    # --------------------------------------------------------
    # 숙도별 실패 개수
    # --------------------------------------------------------

    ripeness_counts = defaultdict(int)

    for sample in failed_samples:

        ripeness_counts[
            sample["ripeness"]
        ] += 1

    print()
    print("[숙도별 실패 개수]")

    for ripeness, count in sorted(
        ripeness_counts.items()
    ):

        print(
            f"  {ripeness}: {count}장"
        )


# ============================================================
# 16. 실행
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SmartSort 특징 추출 프로그램")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 데이터셋 불러오기
    # --------------------------------------------------------

    print()
    print(
        "[1/4] 데이터셋 불러오는 중..."
    )

    dataset = load_dataset()

    print(
        f"전체 데이터: "
        f"{len(dataset)}장"
    )

    # --------------------------------------------------------
    # 2. 특징 추출
    # --------------------------------------------------------

    print()
    print(
        "[2/4] 특징 추출 중..."
    )

    feature_rows, failed_samples = (
        build_feature_dataset(dataset)
    )

    # --------------------------------------------------------
    # 3. 성공 데이터 저장
    # --------------------------------------------------------

    print()
    print(
        "[3/4] 성공한 데이터 저장 중..."
    )

    save_features(
        feature_rows
    )

    # --------------------------------------------------------
    # 4. 실패 데이터 저장
    # --------------------------------------------------------

    print()
    print(
        "[4/4] 실패 데이터 저장 중..."
    )

    save_failed_samples(
        failed_samples
    )

    # --------------------------------------------------------
    # 실패 원인 요약
    # --------------------------------------------------------

    print_failure_summary(
        failed_samples
    )

    # --------------------------------------------------------
    # 최종 결과
    # --------------------------------------------------------

    total = len(dataset)

    success = len(
        feature_rows
    )

    failure = len(
        failed_samples
    )

    print()
    print("=" * 70)
    print("모든 작업이 완료되었습니다.")
    print("=" * 70)

    print(
        f"전체 이미지 : {total}"
    )

    print(
        f"성공        : {success}"
    )

    print(
        f"실패        : {failure}"
    )

    if total > 0:

        success_rate = (
            success
            / total
            * 100
        )

        print(
            f"성공률      : "
            f"{success_rate:.2f}%"
        )

    print()

    print("성공 데이터:")

    print(
        FEATURE_CSV
    )

    print()

    print("실패 이미지 목록:")

    print(
        FAILED_CSV
    )

    print()

    print("실패 이미지 저장 폴더:")

    print(
        PREPROCESSING_FAILED_DIR
    )

    print()


# ============================================================
# 17. 프로그램 시작점
# ============================================================

if __name__ == "__main__":
    main()