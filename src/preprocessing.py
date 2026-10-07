"""잡음 완화, 크기 조정, 그림자 억제, 과일 마스크 생성.

전처리 실패 시 outputs/preprocessing_failed에 진단 자료를 저장합니다.
실패한 결과는 학습에 반환하지 않고 기존 오류를 다시 발생시킵니다.
"""

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREVIEW_DIR = PROJECT_ROOT / "outputs" / "preprocessing_preview"
FAILED_DIR = PROJECT_ROOT / "outputs" / "preprocessing_failed"

DEFAULT_SIZE = 320
DEFAULT_K = 3
DEFAULT_MIN_SATURATION = 40
DEFAULT_DENOISE = True


def write_png(path, image):
    """한글 경로에 PNG를 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise OSError(f"이미지 저장 실패: {path}")

    path.write_bytes(encoded.tobytes())


def save_failure_debug(
    image_path,
    error,
    settings,
    debug,
    output_dir,
):
    """실패 직전 배열과 오류 정보를 진단용으로 저장합니다."""
    source = Path(image_path).resolve()

    try:
        relative = source.relative_to(
            PROJECT_ROOT / "data" / "raw"
        )
    except ValueError:
        # 프로젝트 외부 이미지도 파일명 충돌을 줄여 저장합니다.
        digest = hashlib.sha256(
            str(source).encode("utf-8")
        ).hexdigest()[:12]

        relative = Path("external") / digest / source.name

    # 원본 파일명 전체를 폴더명으로 사용합니다.
    folder = Path(output_dir) / relative.parent / relative.name
    folder.mkdir(parents=True, exist_ok=True)

    # 이미지 읽기 실패처럼 마스크가 없어도 오류 정보는 저장합니다.
    report = {
        "image_path": str(source),
        "error_type": type(error).__name__,
        "error": str(error),
        "stage": debug.get("stage"),
        "settings": settings,
        "area_ratio": debug.get("area_ratio"),
        "mask_available": "mask" in debug,
        "diagnostic_only": True,
    }

    (folder / "error.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 실패 시점까지 만들어진 배열만 저장합니다.
    for key in (
        "original",
        "processed",
        "kmeans_mask",
        "color_mask",
        "candidate_mask",
        "mask",
    ):
        if key in debug:
            write_png(folder / f"{key}.png", debug[key])

    # 원본·전처리 이미지·마스크가 있을 때 비교 이미지를 만듭니다.
    if all(
        key in debug
        for key in ("original", "processed", "mask")
    ):
        mask = debug["mask"]
        outlined = debug["processed"].copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            outlined, contours, -1, (0, 255, 0), 2
        )

        panels = [
            debug["original"],
            debug["processed"],
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            outlined,
        ]

        titles = (
            "Original",
            "Processed",
            "Failed mask",
            "Outline",
        )

        labeled = []

        for title, panel in zip(titles, panels):
            canvas = cv2.copyMakeBorder(
                panel,
                30,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

            cv2.putText(
                canvas,
                title,
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            labeled.append(canvas)

        write_png(
            folder / "debug.png",
            np.hstack(labeled),
        )

    return folder

def save_failed_image(
    image_path,
    output_path,
    fruit,
    ripeness,
    error_message,
):
    """
    전처리에 실패한 이미지를
    과일·숙도·실패 원인과 함께 저장합니다.
    """

    image = read_image(image_path)

    # 너무 큰 원본 이미지는 보기 편하도록 축소
    image = resize_image(
        image,
        size=DEFAULT_SIZE,
    )

    # 이미지 아래에 정보 영역 추가
    info_height = 180

    info = np.full(
        (info_height, image.shape[1], 3),
        255,
        dtype=np.uint8,
    )

    # OpenCV 기본 폰트는 한글을 지원하지 않으므로
    # 화면에는 영어/숫자 위주로 표시합니다.
    lines = [
        "[PREPROCESSING FAILED]",
        f"fruit: {fruit}",
        f"ripeness: {ripeness}",
        f"reason: {error_message}",
    ]

    y = 35

    for line in lines:
        cv2.putText(
            info,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        y += 35

    result = np.vstack([
        image,
        info,
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_png(
        output_path,
        result,
    )

def read_image(image_path):
    """한글 경로를 지원하며 이미지를 BGR 배열로 읽습니다."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"이미지가 없습니다: {image_path}"
        )

    image_bytes = np.frombuffer(
        image_path.read_bytes(),
        dtype=np.uint8,
    )

    if image_bytes.size == 0:
        raise ValueError(
            f"빈 이미지 파일입니다: {image_path}"
        )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"이미지를 읽을 수 없습니다: {image_path}"
        )

    return image


def reduce_noise(image, enabled=True):
    """약한 중앙값 필터로 작은 점 잡음을 줄입니다."""
    if not enabled:
        return image.copy()

    # 실제 작은 반점도 일부 줄어들 수 있습니다.
    return cv2.medianBlur(image, 3)


def resize_image(image, size=DEFAULT_SIZE):
    """비율을 유지하며 긴 변을 지정한 크기로 맞춥니다."""
    if not isinstance(size, int) or size < 32:
        raise ValueError(
            "size는 32 이상의 정수로 지정하세요."
        )

    height, width = image.shape[:2]
    scale = size / max(height, width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
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
    debug=None,
):
    """K-means 후보 영역에서 낮은 채도의 배경·그림자를 제외합니다."""
    debug = {} if debug is None else debug
    debug["stage"] = "mask_parameters"

    if not isinstance(k, int) or not 2 <= k <= 8:
        raise ValueError(
            "k는 2부터 8 사이의 정수로 지정하세요."
        )

    if not 0 <= min_saturation <= 255:
        raise ValueError(
            "min_saturation은 0부터 255 사이여야 합니다."
        )

    height, width = image.shape[:2]

    # 이 흐림 처리는 마스크 생성에만 사용합니다.
    segmentation_image = cv2.GaussianBlur(
        image, (3, 3), 0
    )

    lab = cv2.cvtColor(
        segmentation_image,
        cv2.COLOR_BGR2LAB,
    )

    pixels = lab.reshape(-1, 3).astype(np.float32)

    debug["stage"] = "color_variation"

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

    debug["stage"] = "kmeans"

    _, labels, _ = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    label_map = labels.reshape(height, width)

    # 가장자리에서 가장 많이 나타나는 그룹을 배경으로 봅니다.
    border_width = max(
        1, round(min(height, width) * 0.03)
    )

    border = np.zeros(
        (height, width),
        dtype=bool,
    )

    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    border_counts = np.bincount(
        label_map[border],
        minlength=k,
    )

    background_label = int(np.argmax(border_counts))
    kmeans_candidate = label_map != background_label

    # 회색 그림자와 흰 배경은 채도가 낮다는 점을 이용합니다.
    hsv_for_mask = cv2.cvtColor(
        segmentation_image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv_for_mask[:, :, 1]
    color_candidate = saturation >= min_saturation

    # 두 조건을 각각 확인할 수 있도록 기록합니다.
    debug["kmeans_mask"] = (
        kmeans_candidate.astype(np.uint8) * 255
    )
    debug["color_mask"] = (
        color_candidate.astype(np.uint8) * 255
    )

    candidate_mask = (
        (kmeans_candidate & color_candidate).astype(np.uint8)
        * 255
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    # 작은 틈을 연결하고 떨어진 작은 점을 제거합니다.
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # 윤곽이 없더라도 빈 후보 마스크를 진단용으로 남깁니다.
    debug["candidate_mask"] = candidate_mask.copy()
    debug["mask"] = candidate_mask.copy()
    debug["stage"] = "find_contours"

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
    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    # 내부를 채워 낮은 채도의 반점·꼭지·반사광도 포함합니다.
    cv2.drawContours(
        mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    area_ratio = np.count_nonzero(mask) / mask.size

    # 검사에서 실패해도 최종 후보 마스크가 남도록 먼저 기록합니다.
    debug["mask"] = mask.copy()
    debug["area_ratio"] = float(area_ratio)
    debug["stage"] = "area_ratio_check"

    if not 0.005 <= area_ratio <= 0.85:
        raise ValueError(
            "과일 후보 영역 비율이 비정상적입니다: "
            f"{area_ratio:.4%}. "
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

def load_manual_mask(image_path, original_shape, target_shape):
    """
    수동 마스크가 있으면 읽어 전처리 영상 크기에 맞춥니다.
    없으면 None을 반환해 자동 검출을 사용합니다.
    """
    try:
        relative = Path(image_path).resolve().relative_to(
            (PROJECT_ROOT / "data" / "raw").resolve()
        )
    except ValueError:
        return None

    mask_path = (
        PROJECT_ROOT
        / "outputs"
        / "manual_masks"
        / relative.parent
        / (relative.name + ".png")
    )

    if not mask_path.is_file():
        return None

    data = np.frombuffer(
        mask_path.read_bytes(),
        dtype=np.uint8,
    )
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(
            f"수동 마스크를 읽을 수 없습니다: {mask_path}"
        )

    if mask.shape != tuple(original_shape[:2]):
        raise ValueError(
            f"수동 마스크와 원본의 크기가 다릅니다: {mask_path}"
        )

    # 마스크는 반드시 0 또는 255만 갖도록 합니다.
    mask = (mask >= 128).astype(np.uint8) * 255

    target_height, target_width = target_shape[:2]

    # 일반 이미지 보간 대신 최근접 보간을 사용해 이진값을 유지합니다.
    mask = cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )

    area_ratio = np.count_nonzero(mask) / mask.size

    if not 0.005 <= area_ratio <= 0.85:
        raise ValueError(
            "수동 마스크 영역 비율을 확인하세요: "
            f"{area_ratio:.4%}"
        )

    return mask

def preprocess(
    image_path,
    size=DEFAULT_SIZE,
    k=DEFAULT_K,
    seed=42,
    min_saturation=DEFAULT_MIN_SATURATION,
    denoise=DEFAULT_DENOISE,
    save_failed=True,
    failed_dir=FAILED_DIR,
):
    """이미지 한 장을 처리합니다. 실패하면 진단 저장 후 오류를 냅니다."""
    debug = {"stage": "read_image"}

    settings = {
        "size": size,
        "k": k,
        "seed": seed,
        "min_saturation": min_saturation,
        "denoise": denoise,
    }

    try:
        original = read_image(image_path)

        debug["stage"] = "resize_and_denoise"

        # 축소 전에 점 잡음을 완화합니다.
        cleaned = reduce_noise(
            original,
            enabled=denoise,
        )

        resized_original = resize_image(
            original, size=size
        )
        resized_cleaned = resize_image(
            cleaned, size=size
        )

        debug["original"] = resized_original
        debug["processed"] = resized_cleaned
        debug["stage"] = "load_manual_mask"

        mask = load_manual_mask(
            image_path,
            original_shape=original.shape,
            target_shape=resized_cleaned.shape,
        )

        if mask is None:
            # 수동 마스크가 없는 사진은 기존 자동 검출을 사용합니다.
            mask = create_fruit_mask(
                resized_cleaned,
                k=k,
                min_saturation=min_saturation,
                seed=seed,
                debug=debug,
            )
        else:
            debug["mask"] = mask.copy()
            debug["area_ratio"] = float(
                np.count_nonzero(mask) / mask.size
            )
            print(f"수동 마스크 사용: {Path(image_path).name}")

        
        

        debug["stage"] = "padding_and_hsv"

        # 여백 추가 전 사진을 기준으로 계산합니다.
        area_ratio = np.count_nonzero(mask) / mask.size

        image, padded_mask = pad_to_square(
            resized_cleaned,
            mask,
            size=size,
        )

        original_image, _ = pad_to_square(
            resized_original,
            mask,
            size=size,
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        fruit_image = cv2.bitwise_and(
            image,
            image,
            mask=padded_mask,
        )

        # 기존 반환 형식을 유지합니다.
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

    except (ValueError, OSError, cv2.error) as error:
        if save_failed:
            try:
                folder = save_failure_debug(
                    image_path,
                    error,
                    settings,
                    debug,
                    failed_dir,
                )

                print(f"실패 진단 저장: {folder}")

            except Exception as save_error:
                # 저장 실패가 원래 전처리 오류를 가리지 않도록 합니다.
                print(
                    f"실패 진단 저장 오류: {save_error}",
                    file=sys.stderr,
                )

        # 실패한 마스크를 정상 결과로 반환하지 않습니다.
        raise


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
        outlined_image,
        contours,
        -1,
        (0, 255, 0),
        2,
    )

    mask_image = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    preview = np.hstack([
        original_image,
        outlined_image,
        mask_image,
        fruit_image,
    ])

    write_png(output_path, preview)


def preview_dataset(
    samples,
    per_label=2,
    size=DEFAULT_SIZE,
    k=DEFAULT_K,
    min_saturation=DEFAULT_MIN_SATURATION,
    denoise=DEFAULT_DENOISE,
):
    """과일·숙도별 일부 사진의 점검용 이미지를 저장합니다."""
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
                f"확인 필요: {sample['image_path']}\n"
                f"  {error}"
            )

    print(f"\n미리보기 성공: {success_count}장")
    print(f"미리보기 실패: {failure_count}장")
    print(f"저장 위치: {PREVIEW_DIR}")


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.dataset import load_dataset

    dataset = load_dataset()
    preview_dataset(dataset)