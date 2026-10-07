"""실패 사진의 과일 외곽을 클릭하여 수동 마스크를 만듭니다."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MASK_DIR = PROJECT_ROOT / "outputs" / "manual_masks"
FAILED_DIR = PROJECT_ROOT / "outputs" / "preprocessing_failed"


def read_image(path):
    path = Path(path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")

    return image


def mask_path_for(image_path):
    relative = Path(image_path).resolve().relative_to(
        RAW_DIR.resolve()
    )

    # 원본 확장자도 유지해 파일명 충돌을 피합니다.
    # 예: Overripe_banana/example.jpg.png
    return MASK_DIR / relative.parent / (relative.name + ".png")


def annotate(image_path):
    image_path = Path(image_path).resolve()
    output_path = mask_path_for(image_path)

    image = read_image(image_path)
    height, width = image.shape[:2]

    # 화면에 표시할 크기만 변경합니다.
    scale = min(1.0, 1000 / width, 700 / height)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))

    display = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    points = []
    window = "Manual mask"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < display_width and 0 <= y < display_height:
                points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    print(f"\n작업 중: {image_path}")
    print("왼쪽 클릭: 점 추가 / 오른쪽 클릭 또는 U: 마지막 점 취소")
    print("R: 초기화 / S: 저장 / N: 건너뛰기 / Esc: 전체 종료")

    try:
        while True:
            canvas = display.copy()

            if len(points) >= 3:
                polygon = np.array(points, dtype=np.int32)
                preview_mask = np.zeros(
                    (display_height, display_width),
                    dtype=np.uint8,
                )
                cv2.fillPoly(preview_mask, [polygon], 255)

                overlay = canvas.copy()
                overlay[preview_mask > 0] = (0, 255, 0)
                canvas = cv2.addWeighted(
                    canvas, 0.75, overlay, 0.25, 0
                )

            if len(points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.array(points, dtype=np.int32)],
                    isClosed=len(points) >= 3,
                    color=(0, 255, 0),
                    thickness=2,
                )

            for point in points:
                cv2.circle(canvas, point, 3, (0, 0, 255), -1)

            cv2.putText(
                canvas,
                "Click outline | U: undo R: reset S: save N: skip Esc: quit",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(
                window, cv2.WND_PROP_VISIBLE
            ) < 1:
                return False

            if key in (ord("u"), ord("U")):
                if points:
                    points.pop()

            elif key in (ord("r"), ord("R")):
                points.clear()

            elif key in (ord("n"), ord("N")):
                print("건너뛰었습니다.")
                return True

            elif key == 27:
                return False

            elif key in (ord("s"), ord("S")):
                if len(points) < 3:
                    print("외곽점을 3개 이상 지정하세요.")
                    continue

                # 표시 좌표를 원본 사진 좌표로 변환합니다.
                polygon = np.array(points, dtype=np.float64)
                polygon[:, 0] *= width / display_width
                polygon[:, 1] *= height / display_height
                polygon = np.rint(polygon).astype(np.int32)

                polygon[:, 0] = np.clip(
                    polygon[:, 0], 0, width - 1
                )
                polygon[:, 1] = np.clip(
                    polygon[:, 1], 0, height - 1
                )

                mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )
                cv2.fillPoly(mask, [polygon], 255)

                if not np.any(mask):
                    print("마스크가 비어 있습니다. 다시 지정하세요.")
                    continue

                success, encoded = cv2.imencode(".png", mask)

                if not success:
                    raise OSError("마스크 PNG 생성에 실패했습니다.")

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                output_path.write_bytes(encoded.tobytes())

                print(f"저장 완료: {output_path}")
                return True

    finally:
        cv2.destroyAllWindows()


def find_failed_images(failed_dir):
    paths = set()

    for report_path in Path(failed_dir).rglob("error.json"):
        try:
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            image_path = Path(report["image_path"]).resolve()

            if image_path.is_file():
                paths.add(image_path)
            else:
                print(f"원본이 없습니다: {image_path}")

        except (OSError, ValueError, KeyError) as error:
            print(f"진단 파일 확인 필요: {report_path} / {error}")

    return sorted(paths, key=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images",
        nargs="*",
        help="작업할 원본 이미지 경로",
    )
    parser.add_argument(
        "--failed-dir",
        default=str(FAILED_DIR),
        help="실패 진단 폴더",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="이미 마스크가 있는 사진도 다시 작업",
    )
    args = parser.parse_args()

    images = (
        [Path(path).resolve() for path in args.images]
        if args.images
        else find_failed_images(args.failed_dir)
    )

    print(f"대상 이미지: {len(images)}장")

    for image_path in images:
        try:
            output_path = mask_path_for(image_path)

            if output_path.exists() and not args.redo:
                print(f"기존 마스크 사용: {image_path.name}")
                continue

            if not annotate(image_path):
                break

        except (OSError, ValueError, cv2.error) as error:
            print(f"작업 실패: {image_path} / {error}")


if __name__ == "__main__":
    main()