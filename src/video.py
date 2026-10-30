"""Controlled-background fruit proposals and per-frame SmartSort classification.

This is a color/foreground baseline, not a trained object detector. No source
photo coordinates, labels, slide timestamps, or previous predictions are used.
"""
# 영상의 한 장면에서 색이 있는 전경을 찾고 잘라낸 영역(ROI)을 분류합니다.
# 중성 배경을 가정한 HSV 규칙이며 일반적인 학습 기반 객체 검출기는 아닙니다.
from pathlib import Path
import argparse
import json
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.joint import JointClassifier, ROOT, MODEL_DIR
from src.paths import project_path

FRUITS = {'banana': '바나나', 'mandarin': '귤', 'tomato': '토마토'}
STAGES = {'unripe': '미숙', 'ripe': '적숙', 'overripe': '과숙'}


# 채도 있는 전경에서 잡음을 제거하고 외곽선을 상자 좌표로 변환합니다.
def detect_regions(frame, min_area=500):
    """Return candidate boxes from saturated foreground on a neutral background."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 45, 20]), np.array([179, 255, 255]))
    # 열기 연산은 작은 잡음을 없애고 뒤의 닫기 연산은 전경 내부의 작은 틈을 메웁니다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes)


# 상자에 여백을 붙여 자르고 요청 시 테두리를 채워 정사각형으로 만듭니다.
def crop_region(frame, box, padding=.18, *, square=False, pad_color='edge'):
    if padding < 0 or pad_color not in {'edge', 'white'}:
        raise ValueError('Invalid crop padding or background')
    x, y, w, h = box
    margin = max(8, round(max(w, h) * padding))
    crop = frame[max(0, y-margin):min(frame.shape[0], y+h+margin),
                 max(0, x-margin):min(frame.shape[1], x+w+margin)].copy()
    if square and crop.size:
        height, width = crop.shape[:2]
        side = max(height, width)
        border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
        color = (255, 255, 255) if pad_color == 'white' else tuple(int(v) for v in np.median(border, axis=0))
        top, left = (side-height)//2, (side-width)//2
        crop = cv2.copyMakeBorder(crop, top, side-height-top, left, side-width-left,
                                  cv2.BORDER_CONSTANT, value=color)
    return crop


# Windows·Linux 한글 폰트를 찾고 없으면 기본 폰트를 사용합니다.
def load_font(size=19):
    for path in (Path('C:/Windows/Fonts/malgun.ttf'), Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError('A Korean font is required for video labels')


# 묶음의 구조 정보를 보고 통합 모델 또는 2단계 모델을 불러옵니다.
def load_classifier(model_dir):
    model_dir = project_path(model_dir)
    metadata = json.loads((Path(model_dir)/'bundle.json').read_text(encoding='utf-8'))
    if metadata.get('architecture') == 'two_stage':
        from src.two_stage import TwoStageClassifier
        return TwoStageClassifier(model_dir)
    return JointClassifier(model_dir)


# PIL로 변환해 한글과 상자를 그린 뒤 OpenCV의 BGR 형식으로 돌려줍니다.
def annotate(frame, detections, font):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x, y, w, h = detection['box']
        result = detection['prediction']
        if result['status'] == 'ok':
            label = f"{FRUITS[result['fruit']]} / {STAGES[result['ripeness']]}"
            action = result['agent_decision']['action']
            state = {'auto_classify': '자동 분류', 'human_review': '확인 필요', 'retake': '재촬영 권고'}[action]
            color = (25, 140, 65) if action == 'auto_classify' else (185, 110, 0)
        elif result['status'] == 'unknown':
            label, state, color = '모름 / 숙도 표시 안 함', '확인 필요', (180, 90, 0)
        elif result['status'] == 'pending':
            label, color = '분석 중', (40, 100, 175)
            state = {'edge_wait':'전체 모습 확인 중', 'collecting':'여러 장면 수집 중',
                     'unstable':'판정 불안정 · 확인 필요'}.get(result.get('temporal_state'), '장면 확인 중')
        else:
            label, state, color = '처리 오류', '확인 필요', (190, 40, 40)
        if 'track_id' in result:
            label = f"ID {result['track_id']} · {label}"
            if result.get('temporal_state') == 'held_at_edge':
                state = '이전 안정 판정 유지 · 가장자리'
            elif result.get('temporal_state') == 'confirming_change':
                state = '이전 판정 유지 · 변경 확인 중'
            elif result.get('temporal_state') == 'stable':
                state = f"안정 {result['temporal_support']}/{result['temporal_samples']}장면 · {state}"
        lines = [label, state]
        text_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        tw, th = text_width + 16, 58
        tx = max(0, min(x, frame.shape[1]-tw))
        ty = y+h+5
        if ty+th > frame.shape[0]:
            ty = max(0, y-th-5)
        draw.rectangle((x,y,x+w-1,y+h-1), outline=color, width=3)
        draw.rectangle((tx,ty,tx+tw,ty+th), fill=(255,255,255), outline=color, width=2)
        for i, line in enumerate(lines):
            draw.text((tx+8,ty+4+i*26), line, font=font, fill=color)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# 프레임별 후보를 독립적으로 분류하고 주석 영상과 JSON 기록을 저장합니다.
def process_video(source, destination, *, display=False, min_area=500, model_dir=None):
    source, destination = project_path(source), project_path(destination)
    if source == destination:
        raise ValueError('Input and output video must be different files')
    destination.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    writer = None
    records, counts, times = [], {}, []
    started = perf_counter()
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid input FPS')
        if model_dir is None:
            crop_bundle = ROOT / 'models/joint_nine_crop'
            model_dir = crop_bundle if (crop_bundle/'bundle.json').is_file() else MODEL_DIR
        classifier, font = load_classifier(model_dir), load_font()
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
                if not writer.isOpened():
                    raise RuntimeError('Cannot open output video writer')
            tick = perf_counter()
            detections = []  # Never carry labels over from earlier frames.
            for box in detect_regions(frame, min_area):
                try:
                    prediction = classifier.predict_frame(crop_region(frame, box))
                except (ValueError, RuntimeError, OSError) as error:
                    prediction = {'status': 'failed', 'error': str(error)}
                detections.append({'box': list(box), 'prediction': prediction})
                status = prediction['status']
                counts[status] = counts.get(status, 0) + 1
            rendered = annotate(frame, detections, font)
            writer.write(rendered)
            times.append((perf_counter()-tick)*1000)
            records.append({'frame':index, 'time_seconds':index/fps, 'detections':detections})
            if index in (90, 150, 210, 270, 330, 390):
                cv2.imencode('.jpg',rendered)[1].tofile(str(destination.with_name(f'{destination.stem}_frame_{index:04d}.jpg')))
            if display:
                cv2.imshow('SmartSort - press Q to stop',rendered)
                if cv2.waitKey(max(1,round(1000/fps))) & 0xFF == ord('q'):
                    index += 1
                    break
            index += 1
            if index % 90 == 0:
                print(f'Processed {index} frames', flush=True)
        if index == 0:
            raise ValueError('Video has no readable frames')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    summary = {'source':str(source), 'output':str(destination), 'frames':index, 'fps':fps,
               'duration_seconds':index/fps, 'wall_seconds':perf_counter()-started,
               'mean_processing_ms':float(np.mean(times)), 'detection_status_counts':counts,
               'model_sha256':classifier.sha,
               'model_dir':str(classifier.directory),
               'model_bundle':json.loads((classifier.directory/'bundle.json').read_text(encoding='utf-8')),
               'detector':'HSV saturation foreground baseline; neutral background only; not a trained fruit detector',
               'classification':'every candidate independently classified each frame; no cached labels',
               'evaluation_note':'Demo only; repeated frames are not independent evaluation samples.'}
    destination.with_suffix('.json').write_text(json.dumps({'summary':summary,'frames':records}, ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description='영상의 과일 후보 영역과 종류·숙도 표시 및 저장')
    parser.add_argument('source',nargs='?',type=Path,default=ROOT/'outputs/video_experiment/fruit_conveyor.mp4')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/video_experiment/fruit_conveyor_analyzed.mp4')
    parser.add_argument('--display',action='store_true',help='처리 중 미리보기 (Q: 종료)')
    parser.add_argument('--model-dir',type=Path,help='모델 폴더 (기본: 검증된 crop 모델이 있으면 사용)')
    parser.add_argument('--min-area',type=int,default=500)
    parser.add_argument('--tracking',action='store_true',help='과일 추적·내부 장면 투표로 표시 안정화')
    parser.add_argument('--compare-framewise',action='store_true',help='추적 모드에서 매 프레임 원본 판정도 기록해 비교')
    args = parser.parse_args()
    if args.min_area <= 0:
        parser.error('--min-area must be positive')
    if args.tracking:
        from src.video_tracking import process_tracked_video
        process_tracked_video(args.source,args.output,display=args.display,min_area=args.min_area,
                              model_dir=args.model_dir,compare_framewise=args.compare_framewise)
    else:
        if args.compare_framewise:
            parser.error('--compare-framewise requires --tracking')
        process_video(args.source,args.output,display=args.display,min_area=args.min_area,model_dir=args.model_dir)


if __name__ == '__main__':
    main()
