"""잡음 완화, 크기 조정, 그림자 억제, 과일 마스크 생성."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = PROJECT_ROOT / "outputs" / "preprocessing_preview"

# 초기 설정값입니다. 미리보기를 확인하며 조정하세요.
DEFAULT_SIZE = 320
DEFAULT_K = 3
DEFAULT_MIN_SATURATION = 40
DEFAULT_DENOISE = True


def read_image(image_path):
    """한글 경로를 지원하며 이미지를 BGR 배열로 읽습니다."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"이미지가 없습니다: {image_path}")

    image_bytes = np.frombuffer(
        image_path.read_bytes(), dtype=np.uint8
    )

    if image_bytes.size == 0:
        raise ValueError(f"빈 이미지 파일입니다: {image_path}")

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    return image


def reduce_noise(image, enabled=True):
    """약한 중앙값 필터로 작은 점 잡음을 줄입니다."""
    if not enabled:
        return image.copy()

    # 실제 작은 반점도 일부 줄어들 수 있으므로 결과를 확인하세요.
    return cv2.medianBlur(image, 3)


def resize_image(image, size=DEFAULT_SIZE):
    """비율을 유지하며 긴 변을 지정한 크기로 맞춥니다."""
    if not isinstance(size, int) or size < 32:
        raise ValueError("size는 32 이상의 정수로 지정하세요.")

    height, width = image.shape[:2]
    scale = size / max(height, width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = (
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )


def create_fruit_mask(
    image,
    k=DEFAULT_K,
    min_saturation=DEFAULT_MIN_SATURATION,
    seed=42,
):
    """K-means 후보 영역에서 채도가 낮은 배경·그림자를 제외합니다."""
    if not isinstance(k, int) or not 2 <= k <= 8:
        raise ValueError("k는 2부터 8 사이의 정수로 지정하세요.")

    if not 0 <= min_saturation <= 255:
        raise ValueError("min_saturation은 0부터 255 사이여야 합니다.")

    height, width = image.shape[:2]

    # 이 흐림 처리는 마스크 생성에만 사용합니다.
    segmentation_image = cv2.GaussianBlur(image, (3, 3), 0)

    lab = cv2.cvtColor(
        segmentation_image, cv2.COLOR_BGR2LAB
    )
    pixels = lab.reshape(-1, 3).astype(np.float32)

    if np.max(np.std(pixels, axis=0)) < 1.0:
        raise ValueError(
            "색상 변화가 너무 적어 과일을 분리하기 어렵습니다."
        )

    cv2.setRNGSeed(seed)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.5,
    )

    _, labels, _ = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    label_map = labels.reshape(height, width)

    # 사진 가장자리에서 가장 많이 나타나는 그룹을 배경으로 봅니다.
    border_width = max(1, round(min(height, width) * 0.03))
    border = np.zeros((height, width), dtype=bool)

    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    border_counts = np.bincount(
        label_map[border], minlength=k
    )
    background_label = int(np.argmax(border_counts))

    kmeans_candidate = label_map != background_label

    # 회색 그림자와 흰 배경은 채도가 낮다는 점을 이용합니다.
    hsv_for_mask = cv2.cvtColor(
        segmentation_image, cv2.COLOR_BGR2HSV
    )
    saturation = hsv_for_mask[:, :, 1]
    color_candidate = saturation >= min_saturation

    # 두 조건을 모두 만족하는 영역만 남깁니다.
    candidate_mask = (
        (kmeans_candidate & color_candidate).astype(np.uint8) * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    # 작은 틈을 연결하고 떨어진 작은 점을 제거합니다.
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_CLOSE, kernel
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "과일 영역을 찾지 못했습니다. "
            "채도 기준이나 배경을 확인하세요."
        )

    # 과일 한 개가 촬영됐다는 가정으로 가장 큰 영역을 선택합니다.
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros((height, width), dtype=np.uint8)

    # 내부를 채워 반점·꼭지·반사광의 낮은 채도 영역도 포함합니다.
    # 실제 이미지 색상은 이 단계에서 변경하지 않습니다.
    cv2.drawContours(
        mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    area_ratio = np.count_nonzero(mask) / mask.size

    if not 0.005 <= area_ratio <= 0.85:
        raise ValueError(
            f"과일 후보 영역 비율이 비정상적입니다: {area_ratio:.1%}. "
            "촬영 구도, 채도 기준 또는 k 값을 확인하세요."
        )

    return mask


def pad_to_square(image, mask, size):
    """여백을 추가해 이미지와 마스크 크기를 통일합니다."""
    height, width = image.shape[:2]

    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left

    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    padded_mask = cv2.copyMakeBorder(
        mask,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    return padded_image, padded_mask


def preprocess(
    image_path,
    size=DEFAULT_SIZE,
    k=DEFAULT_K,
    seed=42,
    min_saturation=DEFAULT_MIN_SATURATION,
    denoise=DEFAULT_DENOISE,
):
    """이미지 한 장을 처리하고 특징 추출용 배열을 반환합니다."""
    original = read_image(image_path)

    # 점 잡음이 크기 축소 과정에서 퍼지기 전에 먼저 완화합니다.
    cleaned = reduce_noise(original, enabled=denoise)

    resized_original = resize_image(original, size=size)
    resized_cleaned = resize_image(cleaned, size=size)

    mask = create_fruit_mask(
        resized_cleaned,
        k=k,
        min_saturation=min_saturation,
        seed=seed,
    )

    # 여백 추가 전 사진을 기준으로 영역 비율을 계산합니다.
    area_ratio = np.count_nonzero(mask) / mask.size

    image, padded_mask = pad_to_square(
        resized_cleaned, mask, size=size
    )
    original_image, _ = pad_to_square(
        resized_original, mask, size=size
    )

    # 특징 추출에는 잡음을 완화한 이미지의 색상을 사용합니다.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    fruit_image = cv2.bitwise_and(
        image, image, mask=padded_mask
    )

    return {
        "image_path": str(image_path),
        "original_image": original_image,
        "image": image,
        "hsv": hsv,
        "mask": padded_mask,
        "fruit_image": fruit_image,
        "area_ratio": area_ratio,
        "denoise": denoise,
        "min_saturation": min_saturation,
    }


def save_preview(result, output_path):
    """필터 전·필터 후 윤곽·마스크·배경 제거 결과를 저장합니다."""
    original_image = result["original_image"]
    image = result["image"]
    mask = result["mask"]
    fruit_image = result["fruit_image"]

    outlined_image = image.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        outlined_image, contours, -1, (0, 255, 0), 2
    )

    mask_image = cv2.cvtColor(
        mask, cv2.COLOR_GRAY2BGR
    )

    preview = np.hstack([
        original_image,
        outlined_image,
        mask_image,
        fruit_image,
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(".png", preview)

    if not success:
        raise OSError(f"결과 이미지 저장 실패: {output_path}")

    output_path.write_bytes(encoded.tobytes())


def preview_dataset(
    samples,
    per_label=2,
    size=DEFAULT_SIZE,
    k=DEFAULT_K,
    min_saturation=DEFAULT_MIN_SATURATION,
    denoise=DEFAULT_DENOISE,
):
    """과일·숙도별 일부 사진을 처리해 점검용 이미지를 저장합니다."""
    counts = defaultdict(int)
    success_count = 0
    failure_count = 0

    for sample in samples:
        key = (sample["fruit"], sample["ripeness"])

        if counts[key] >= per_label:
            continue

        counts[key] += 1

        try:
            result = preprocess(
                sample["path"],
                size=size,
                k=k,
                min_saturation=min_saturation,
                denoise=denoise,
            )

            relative_path = Path(sample["image_path"])
            output_path = (
                PREVIEW_DIR
                / relative_path.parent
                / f"{relative_path.stem}_preview.png"
            )

            save_preview(result, output_path)
            success_count += 1

            print(
                f"완료: {sample['image_path']} "
                f"/ 영역 비율: {result['area_ratio']:.1%}"
            )

        except (ValueError, OSError, cv2.error) as error:
            failure_count += 1
            print(
                f"확인 필요: {sample['image_path']}\n  {error}"
            )

    print(f"\n미리보기 성공: {success_count}장")
    print(f"미리보기 실패: {failure_count}장")
    print(f"저장 위치: {PREVIEW_DIR}")


if __name__ == "__main__":
    from src.dataset import load_dataset

    dataset = load_dataset()
    preview_dataset(dataset)