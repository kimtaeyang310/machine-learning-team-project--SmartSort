"""Experimental fruit-kind gate followed by a separate per-fruit ripeness model."""
# 실험용 2단계 구조로 종류 모델이 받아들인 과일의 전용 숙도 모델만 실행합니다.
# 기본 통합 9클래스 모델과 비교하는 용도입니다.
from pathlib import Path
import hashlib
import json
from time import perf_counter
import joblib
import numpy as np
import torch
from ultralytics import YOLO
from src.joint import ROOT
from src.rejection import unfamiliarity_scores
from src.agent import decide_next_action
from src.paths import project_path

FRUITS=tuple(sorted(('banana','mandarin','tomato')))
STAGES=tuple(sorted(('unripe','ripe','overripe')))


# 종류·숙도 모델의 출력 순서를 맞추는 공통 래퍼이며 특징 추출은 선택 사항입니다.
class ClassifierHead:
    def __init__(self, checkpoint, labels, *, embedding=False, imgsz=224):
        self.labels=list(labels)
        self.imgsz=imgsz
        self.network=YOLO(str(checkpoint))
        self.captured=[]
        self.capture_embedding=embedding
        if embedding:
            buffer=self.captured
            def capture(module,args):
                buffer.clear();buffer.append(args[0].detach().cpu().numpy().copy())
            self.hook=self.network.model.model[-1].linear.register_forward_pre_hook(capture)

    def raw(self, image):
        self.captured.clear()
        result=self.network.predict(image if isinstance(image,np.ndarray) else str(image),imgsz=self.imgsz,device='cpu',verbose=False)[0]
        if set(result.names.values())!=set(self.labels):
            raise ValueError(f'Unexpected labels: {result.names}')
        values=result.probs.data.detach().cpu().numpy()
        probabilities=np.array([values[next(i for i,name in result.names.items() if name==label)] for label in self.labels],dtype=float)
        embedding=None
        if self.capture_embedding:
            if not self.captured or self.captured[-1].shape[0]!=1:
                raise ValueError('Missing type embedding')
            embedding=self.captured[-1][0]
            embedding=embedding/max(float(np.linalg.norm(embedding)),1e-12)
        return probabilities,embedding


# 종류 확률과 특징 거리를 기준과 비교해 수락 여부·거리·임계값을 반환합니다.
def kind_gate(probabilities, embeddings, policy, *, distances=None):
    p=np.asarray(probabilities)
    indices=p.argmax(axis=1)
    fruits=np.array(FRUITS)[indices]
    if distances is None:
        distances=unfamiliarity_scores(p,embeddings,policy['reference'])['embedding']
    thresholds=np.array([policy['thresholds'][str(fruit)] for fruit in fruits])
    ordered=np.sort(p,axis=1)
    config=policy['gate']
    strong=(ordered[:,-1]>=config.get('min_confidence',0.)) & ((ordered[:,-1]-ordered[:,-2])>=config.get('min_margin',0.))
    kind=config['kind']
    distance_ok=distances<=thresholds*config.get('distance_multiplier',1.)
    if kind=='distance':
        accepted=distance_ok
    elif kind=='confidence':
        accepted=strong
    elif kind=='both':
        accepted=distance_ok & strong
    elif kind=='rescue':
        accepted=(distances<=thresholds) | (distance_ok & strong)
    else:
        raise ValueError(f'Unknown kind gate: {kind}')
    return accepted & np.isfinite(p).all(axis=1) & np.isfinite(distances),distances,thresholds


# 두 모델 출력을 공통 형식으로 묶고 거부된 사진의 숙도를 숨깁니다.
def format_prediction(kind_p,stage_p,accepted,distance,threshold,policy):
    fruit=FRUITS[int(np.argmax(kind_p))]
    stage=STAGES[int(np.argmax(stage_p))] if stage_p is not None else None
    stage_scores=dict(zip(STAGES,map(float,stage_p))) if stage_p is not None else None
    result={'status':'ok' if accepted else 'unknown','fruit':fruit if accepted else 'unknown',
        'ripeness':stage if accepted else None,'candidate_fruit':fruit,'candidate_ripeness':stage,
        'candidate_class':fruit+'_'+stage if stage else None,'probabilities':stage_scores if accepted else None,
        'kind_probabilities':dict(zip(FRUITS,map(float,kind_p))),'stage_probabilities':stage_scores,
        'kind_confidence':float(np.max(kind_p)),'score':float(distance),'threshold':float(threshold),
        'method':'two_stage_kind_gate','acceptance_rule':policy['gate'],
        'probability_semantics':'stage probabilities are conditional on selected fruit; kind gate is independent',
        'image_path':'','error':''}
    result['agent_decision']=decide_next_action(result) if accepted else {
        'action':'human_review','reason':'지원 과일 여부가 불확실합니다. 숙도 결과를 표시하지 않습니다.',
        'suggested_label':None,'confidence':None,'requires_confirmation':True}
    return result


# 묶음의 해시를 검증하고 종류 모델 하나와 숙도 모델 세 개를 준비합니다.
class TwoStageClassifier:
    def __init__(self,model_dir):
        self.directory=project_path(model_dir)
        metadata=json.loads((self.directory/'bundle.json').read_text(encoding='utf-8'))
        if metadata.get('architecture')!='two_stage':
            raise ValueError('Not a two-stage bundle')
        for file,expected in metadata['sha256'].items():
            if hashlib.sha256((self.directory/file).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Changed two-stage artifact: {file}')
        self.sha=hashlib.sha256(json.dumps(metadata['sha256'],sort_keys=True).encode()).hexdigest()
        self.model_name=metadata['model_name']
        torch.set_num_threads(4)
        self.kind=ClassifierHead(self.directory/'kind.pt',FRUITS,embedding=True)
        self.stages={f:ClassifierHead(self.directory/(f+'_stage.pt'),STAGES) for f in FRUITS}
        self.policy=joblib.load(self.directory/'policy.joblib')

    # BGR 이미지의 종류를 판정하고 통과한 과일의 숙도 모델만 실행합니다.
    def predict_frame(self,frame):
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or not frame.size:
            raise ValueError('frame must be a nonempty uint8 BGR image')
        start=perf_counter()
        p,e=self.kind.raw(frame)
        accepted,distance,threshold=kind_gate(p[None],e[None],self.policy)
        stage_p=None
        if accepted[0]:
            stage_p,_=self.stages[FRUITS[int(p.argmax())]].raw(frame)
        result=format_prediction(p,stage_p,bool(accepted[0]),distance[0],threshold[0],self.policy)
        result.update(model=self.model_name,total_ms=(perf_counter()-start)*1000)
        result['fruit_identity']={'accepted':bool(accepted[0]),'candidate_fruit':result['candidate_fruit'],
            'method':result['method'],'score':result['score'],'threshold':result['threshold'],'model_sha256':self.sha}
        return result
