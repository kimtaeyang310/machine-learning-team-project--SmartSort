"""Portable selected-test video presentation; required assets live outside outputs."""
# 설정의 사진을 움직이는 영상으로 만든 뒤 추적 분류 시연을 실행합니다.
# 경로를 프로젝트 내부 상대경로로 제한하여 다른 컴퓨터에서도 자료를 찾도록 합니다.
from pathlib import Path,PureWindowsPath
import argparse
import json
import math
import cv2
import numpy as np
from src.paths import ROOT

DEFAULT_CONFIG='config/presentation.json'


# 절대경로와 프로젝트 밖으로 빠져나가는 경로를 거부합니다.
def config_path(value,root):
    if not isinstance(value,str) or not value.strip():
        raise ValueError('시연 설정의 파일 경로를 확인해 주세요.')
    path=Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError('시연 설정에는 프로젝트 기준 상대경로를 사용해 주세요.')
    resolved=(root/path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('시연 파일은 프로젝트 폴더 안에 있어야 합니다.')
    return resolved


# 설정과 사진·모델 파일 존재 여부 및 입력·출력 충돌을 검사합니다.
def load_presentation(config=DEFAULT_CONFIG,*,root=ROOT):
    root=Path(root).resolve()
    path=config_path(config,root)
    if not path.is_file():raise ValueError(f'시연 설정 파일이 없습니다: {config}')
    settings=json.loads(path.read_text(encoding='utf-8'))
    images=[config_path(value,root) for value in settings['images']]
    if not images or len(set(images))!=len(images):raise ValueError('시연 사진 목록을 확인해 주세요.')
    model=config_path(settings['model_dir'],root)
    source=config_path(settings['source'],root)
    output=config_path(settings['output'],root)
    if source==output or source in images or output in images:raise ValueError('입력·출력 경로가 겹칩니다.')
    missing=[str(p.relative_to(root)) for p in images+[model/'bundle.json',model/'model.pt',model/'policy.joblib',model/'split_manifest.json'] if not p.is_file()]
    if missing:raise ValueError('시연에 필요한 파일이 없습니다: '+', '.join(missing))
    return settings,images,model,source,output


# 사진 비율을 유지해 흰 배경 위에서 일정 간격으로 이동하는 입력 영상을 만듭니다.
def build_conveyor(images,destination,settings):
    width,height,fps,side=(int(settings[k]) for k in ('width','height','fps','image_size'))
    speed,interval=float(settings['speed']),float(settings['spacing_seconds'])
    if min(width,height,fps,side,speed,interval)<=0 or width%2 or height%2 or side>min(width,height):
        raise ValueError('영상 크기·속도 설정을 확인해 주세요.')
    cards=[]
    for path in images:
        image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'사진을 읽을 수 없습니다: {path.name}')
        h,w=image.shape[:2];scale=side/max(h,w)
        cards.append(cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR))
    total=math.ceil(((len(cards)-1)*interval+(width+side)/speed)*fps)+1
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('시연 입력 영상을 저장할 수 없습니다.')
    try:
        for n in range(total):
            canvas=np.full((height,width,3),255,np.uint8)
            for index,card in enumerate(cards):
                x=round(-side+speed*(n/fps-index*interval));h,w=card.shape[:2];y=(height-h)//2
                left,right=max(0,x),min(width,x+w)
                if right>left:canvas[y:y+h,left:right]=card[:,left-x:right-x]
            writer.write(canvas)
    finally:writer.release()
    return total


# 설정 검사와 영상 생성 후 지정된 모델로 추적 시연을 실행합니다.
def run_presentation(*,display=True,config=DEFAULT_CONFIG):
    from src.video_tracking import process_tracked_video
    settings,images,model,source,output=load_presentation(config)
    print('선별 테스트 과일 시연: 입력 영상 준비 중...',flush=True)
    build_conveyor(images,source,settings['video'])
    return process_tracked_video(source,output,model_dir=model,display=display)


def main():
    parser=argparse.ArgumentParser(description='선별 테스트 과일 시연')
    parser.add_argument('--no-display',action='store_true',help='화면 창 없이 결과 영상 저장')
    args=parser.parse_args()
    run_presentation(display=not args.no_display)


if __name__=='__main__':main()
