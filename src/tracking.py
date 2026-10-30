"""Geometry/appearance tracking and voting on complete interior observations.

No source-photo names, slide timings, or known labels are used. Stable means
consistent model outputs, not verified ripeness or improved recognition accuracy.
"""
# 여러 프레임의 같은 과일에 같은 ID를 붙이고 최근 예측을 투표로 모읍니다.
# box는 (왼쪽 x, 위쪽 y, 너비, 높이)이며 안정된 표시가 정답을 보장하지는 않습니다.
from dataclasses import dataclass,field
from collections import deque,Counter
import copy
import math
import numpy as np


# 저채도 배경을 제외한 HSV 히스토그램으로 과일의 색상 특징을 요약합니다.
def appearance(frame,box):
    import cv2
    x,y,w,h=box
    hsv=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,45,20]),np.array([179,255,255]))
    histogram=cv2.calcHist([hsv],[0,1],mask,[18,8],[0,180,0,256]).ravel()
    return histogram/max(float(np.linalg.norm(histogram)),1e-12)


# 상자 중심 좌표를 구해 이전 프레임의 이동량과 비교합니다.
def center(box):
    x,y,w,h=box
    return np.array([x+w/2,y+h/2],dtype=float)


# 예측을 투표용 튜플로 바꾸며 처리 실패는 투표에서 제외합니다.
def result_key(result):
    if result is None or result.get('status')=='failed':
        return None
    if result['status']=='unknown':
        return ('unknown',)
    if result['status']=='ok':
        return ('ok',result['fruit'],result['ripeness'])
    return None


# 과일 하나의 위치·속도·관측 기록·안정된 예측 상태를 보관합니다.
@dataclass
class Track:
    id:int
    box:tuple
    last_seen:int
    signature:np.ndarray
    velocity:np.ndarray=field(default_factory=lambda:np.zeros(2))
    hits:int=1
    full_run:int=0
    areas:deque=field(default_factory=lambda:deque(maxlen=4))
    history:deque=field(default_factory=deque)
    stable:tuple|None=None
    representative:dict|None=None
    evidence_frame:int|None=None
    last_support_frame:int|None=None
    challenger:tuple|None=None
    challenger_hits:int=0
    ambiguous_hits:int=0
    unstable:bool=False
    switches:int=0


# 이동 예상 위치와 색상·크기를 비교해 검출 영역과 기존 ID를 연결합니다.
class FruitTracker:
    def __init__(self,max_missed=6):
        self.max_missed=max_missed
        self.tracks={}
        self.next_id=1
        self.finished=[]

    # 오래 안 보인 ID는 종료하고 비용이 작은 후보부터 연결한 뒤 새 ID를 발급합니다.
    def update(self,boxes,frame,frame_index):
        expired=[key for key,t in self.tracks.items() if frame_index-t.last_seen>self.max_missed]
        for key in expired:
            self.finished.append(self.tracks.pop(key))
        signatures=[appearance(frame,box) for box in boxes]
        pairs=[]
        for key,t in self.tracks.items():
            elapsed=frame_index-t.last_seen
            predicted=center(t.box)+t.velocity*elapsed
            for index,box in enumerate(boxes):
                distance=float(np.linalg.norm(center(box)-predicted))
                gate=max(45.,max(t.box[2:])*.75+np.linalg.norm(t.velocity)*elapsed)
                similarity=float(t.signature@signatures[index])
                ratio=(box[2]*box[3])/max(t.box[2]*t.box[3],1)
                if distance>gate or similarity<.15 or not .2<=ratio<=5.:
                    continue
                # 예상 위치에서 멀거나 색상·크기가 다를수록 연결 비용을 높입니다.
                cost=distance/gate+.6*(1-similarity)+.15*abs(math.log(ratio))
                pairs.append((cost,key,index))
        assigned={};used=set()
        # 작은 비용부터 연결하되 한 ID와 검출 영역이 중복 연결되지 않게 합니다.
        for cost,key,index in sorted(pairs):
            if key in used or index in assigned:
                continue
            t=self.tracks[key]
            elapsed=frame_index-t.last_seen
            velocity=(center(boxes[index])-center(t.box))/max(1,elapsed)
            t.velocity=.5*t.velocity+.5*velocity
            t.box=tuple(boxes[index]);t.last_seen=frame_index;t.hits+=1
            t.signature=.8*t.signature+.2*signatures[index]
            t.signature/=max(float(np.linalg.norm(t.signature)),1e-12)
            assigned[index]=t;used.add(key)
        for index,box in enumerate(boxes):
            if index not in assigned:
                t=Track(self.next_id,tuple(box),frame_index,signatures[index])
                self.tracks[t.id]=t;assigned[index]=t;self.next_id+=1
        return [assigned[i] for i in range(len(boxes))]


# 여백까지 화면 내부에 있고 크기가 안정된 관측이 3회 이어져야 투표에 사용합니다.
def interior_observation(track,frame_shape,padding=.18):
    height,width=frame_shape[:2]
    x,y,w,h=track.box
    margin=max(12,round(max(w,h)*padding))
    interior=(x>=margin and y>=margin and x+w<=width-margin and y+h<=height-margin)
    area=w*h
    stable_size=not track.areas or min(area,track.areas[-1])/max(area,track.areas[-1])>=.9
    track.areas.append(area)
    track.full_run=track.full_run+1 if interior and stable_size else 0
    return track.full_run>=3


# 최근 예측의 지지율로 표시를 안정화하고 결론 변경에는 추가 확인을 기다립니다.
class TemporalVote:
    def __init__(self,fps,min_samples=5,window=15,support=.8,sample_seconds=.1,switch_confirmations=3,hold_seconds=1.5):
        self.min_samples=min_samples;self.window=window;self.support=support
        self.interval=max(1,round(fps*sample_seconds))
        self.switch_confirmations=switch_confirmations
        self.max_age=max(1,round(fps*hold_seconds))

    # 유효한 관측 중 설정된 프레임 간격을 지난 경우에만 투표에 추가합니다.
    def needs_sample(self,track,frame_index,eligible):
        return eligible and (not track.history or frame_index-track.history[-1]['frame']>=self.interval)

    # 최근 window개 결과의 다수 의견을 구하고 결론을 바꿀 반복 지지가 있는지 확인합니다.
    def add(self,track,result,frame_index):
        key=result_key(result)
        if key is None:
            return
        track.history.append({'frame':frame_index,'key':key,'prediction':copy.deepcopy(result)})
        while len(track.history)>self.window:
            track.history.popleft()
        counts=Counter(s['key'] for s in track.history)
        leader,count=counts.most_common(1)[0]
        ratio=count/len(track.history)
        # 관측 수와 같은 결론을 지지한 수·비율을 모두 충족해야 안정된 결과로 인정합니다.
        consensus=(len(track.history)>=self.min_samples and count>=self.min_samples and ratio>=self.support)
        if track.stable==key:
            track.last_support_frame=frame_index
        if consensus:
            track.ambiguous_hits=0;track.unstable=False
            if track.stable is None or track.stable==leader:
                track.stable=leader;track.challenger=None;track.challenger_hits=0
            else:
                # 다수 의견이 한 번 바뀌어도 바로 표시를 뒤집지 않고 연속 확인 횟수를 쌓습니다.
                track.challenger_hits=track.challenger_hits+1 if track.challenger==leader else 1
                track.challenger=leader
                if track.challenger_hits>=self.switch_confirmations:
                    track.stable=leader;track.switches+=1
                    track.challenger=None;track.challenger_hits=0
            matches=[s for s in track.history if s['key']==track.stable]
            if matches:
                track.representative=copy.deepcopy(matches[-1]['prediction'])
                track.evidence_frame=matches[-1]['frame']
                track.last_support_frame=matches[-1]['frame']
        else:
            track.challenger=None;track.challenger_hits=0
            if len(track.history)>=self.min_samples and ratio<.6:
                track.ambiguous_hits+=1
                track.unstable=track.ambiguous_hits>=3

    # 근거가 오래됐거나 투표가 불안정하면 확정 라벨 대신 대기 상태를 반환합니다.
    def display(self,track,frame_index,eligible):
        age=frame_index-track.last_support_frame if track.last_support_frame is not None else None
        if track.representative is None or age is None or age>self.max_age:
            result={'status':'pending','fruit':'unknown','ripeness':None,'temporal_state':'collecting' if eligible else 'edge_wait'}
        elif track.unstable:
            result={'status':'pending','fruit':'unknown','ripeness':None,'temporal_state':'unstable'}
        else:
            result=copy.deepcopy(track.representative)
            result['temporal_state']='stable' if eligible else 'held_at_edge'
            if eligible and track.challenger is not None:
                result['temporal_state']='confirming_change'
        counts=Counter(s['key'] for s in track.history)
        result.update(track_id=track.id,temporal_samples=len(track.history),
            temporal_support=counts.get(track.stable,0),evidence_frame=track.evidence_frame,
            evidence_age_frames=age)
        return result
