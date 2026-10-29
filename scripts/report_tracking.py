"""Verify tracking artifacts, build before/after video, and document stability."""
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from src.joint import ROOT
from src.video import annotate,load_font


def label_key(d):
    p=d['prediction']
    return (d['track_id'],p['status'],p.get('fruit'),p.get('ripeness'),p.get('temporal_state'))


def main():
    out=ROOT/'outputs/video_experiment'
    comparison=json.loads((out/'fruit_conveyor_tracked_comparison.json').read_text(encoding='utf-8'))
    production=json.loads((out/'fruit_conveyor_tracked.json').read_text(encoding='utf-8'))
    assert len(comparison['frames'])==len(production['frames'])==461
    for a,b in zip(comparison['frames'],production['frames']):
        assert [label_key(x) for x in a['detections']]==[label_key(x) for x in b['detections']]
    for name in ('fruit_conveyor_tracked.mp4','fruit_conveyor_tracked_comparison.mp4'):
        cap=cv2.VideoCapture(str(out/name));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(540,960)
            n+=1
        cap.release();assert n==461
    font=load_font(19)
    header=Image.new('RGB',(1920,52),'white')
    draw=ImageDraw.Draw(header)
    draw.text((20,12),'기존: 매 프레임 판정',font=load_font(22),fill='black')
    draw.text((980,12),'개선: 추적 ID + 내부 장면 종합',font=load_font(22),fill='black')
    header=cv2.cvtColor(np.asarray(header),cv2.COLOR_RGB2BGR)
    source=cv2.VideoCapture(str(out/'fruit_conveyor.mp4'))
    tracked=cv2.VideoCapture(str(out/'fruit_conveyor_tracked.mp4'))
    result=out/'fruit_conveyor_tracking_before_after.mp4'
    writer=cv2.VideoWriter(str(result),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,592))
    if not writer.isOpened():raise RuntimeError('Cannot open comparison writer')
    try:
        for record in comparison['frames']:
            ok,frame=source.read();ok2,right=tracked.read()
            assert ok and ok2
            raw=[{'box':d['box'],'prediction':d['raw_prediction']} for d in record['detections']]
            left=annotate(frame,raw,font)
            canvas=np.vstack((header,np.hstack((left,right))))
            writer.write(canvas)
            if record['frame']==300:
                cv2.imencode('.jpg',canvas)[1].tofile(str(out/'tracking_before_after_preview.jpg'))
    finally:
        source.release();tracked.release();writer.release()
    cap=cv2.VideoCapture(str(result));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(592,1920);n+=1
    cap.release();assert n==461
    metrics=comparison['summary']['metrics']
    summaries=comparison['summary']['per_track']
    confirmed=[r for r in summaries if r['drawn_frames']]
    c,p=comparison['summary'],production['summary']
    report=['# 과일 추적·영상 연속성 활용 실험','',
        '## 결과','',
        '- 같은 모델과 같은 461프레임 이동 영상에서 매 프레임 판정과 추적 기반 장면 종합을 비교했습니다.',
        f"- 검출 후보 추적 ID {metrics['track_count']}개 중 {metrics['confirmed_track_count']}개가 지속 관측되어 표시됐고, 1~2프레임만 나타난 임시 후보 {metrics['suppressed_short_track_count']}개는 화면 표시에서 제외했습니다.",
        f"- 동일 추적 ID 안에서 원본 종류·숙도·모름 상태의 변경은 {metrics['raw_label_changes']}회, 안정 판정의 변경은 {metrics['stable_label_changes']}회였습니다.",
        '- 분석 중→안정 판정 같은 대기 상태 전환은 위 안정 판정 변경 횟수에 포함하지 않았습니다. 이 전환은 per_track의 display_changes_including_pending에 별도로 기록했습니다.',
        f"- 가장자리 장면의 투표 사용은 {metrics['edge_samples_used']}회입니다.",
        '- 여섯 지속 추적 모두 첫 유효 내부 장면 후 약 0.4초에 최초 안정 판정을 얻었습니다.',
        f"- 안정 결과 표시 {metrics['stable_display_frames']}개 / 실제 박스 표시 {metrics['drawn_object_frames']}개 개체-프레임 = {metrics['stable_display_frames']/metrics['drawn_object_frames']:.1%}. 나머지는 장면 수집/유효 장면 대기입니다.",'',
        '## 실제로 확인한 효과와 범위','',
        '- 원본의 내부 유효 장면만 보면 이미 라벨 변화가 0회였습니다. 따라서 이 슬라이드 이동 영상에서의 주된 개선은 가장자리의 불완전한 장면을 판정에서 제외한 효과입니다.',
        '- 단발 오판 투표 억제, 지속적인 실제 결과 변경 반영, 모름을 알려진 과일로 강제 승격하지 않음, 오래된 증거 만료, 다른 개체의 결과 상속 방지는 합성 입력을 이용한 단위 테스트로 확인했습니다.',
        '- 0회 변경은 정확도 100%를 뜻하지 않습니다. M10 귤은 기존 모델에서 계속 모름이므로 안정화 후에도 모름입니다. 잘못된 판정이 반복되면 잘못된 결과도 안정화될 수 있습니다.',
        '- 추적 정답 ID나 실제 숙도 영상 라벨을 수작업으로 만들지 않았으므로 ID 정확도·숙도 정확도를 평가한 실험은 아닙니다.',
        '- 프레임들은 원본 사진을 움직인 연속 자료이며 독립 테스트 표본이 아닙니다.','',
        '## 동작','',
        '1. 색상 기반 후보 영역을 검출하고 위치·이동 속도·크기·색 분포로 이전 프레임과 일대일 연결합니다.',
        '2. 세 프레임 이상 이어진 추적만 화면에 ID와 함께 표시합니다. 잠깐 생긴 잡음 박스는 표시하지 않습니다.',
        '3. 박스와 18% 크롭 여백이 화면 안쪽에 있고, 크기 변화가 작은 상태가 세 프레임 이어져야 유효 관측으로 간주합니다. 완전한 가시성을 증명하는 규칙은 아닙니다.',
        '4. 유효 관측을 3프레임(30 FPS 기준 약 0.1초)마다 모읍니다. 최근 최대 15개 관측 중 최소 5개가 같은 결과이고 비율이 80% 이상이면 표시합니다.',
        '5. 이미 안정된 결과를 바꾸려면 새 다수 결과가 세 번 연속 확인되어야 합니다. 애매한 혼합이 지속되면 판정 불안정으로 표시합니다.',
        '6. 가장자리로 나갈 때는 새 판정을 섞지 않고 이전 안정 결과를 최대 1.5초 유지합니다. 화면에는 유지 중임을 표시합니다.',
        '7. 검출이 없는 프레임에는 이전 박스를 그리지 않습니다. 6프레임 넘게 사라진 추적은 종료하고, 다시 나타나면 새 ID와 빈 증거 목록으로 시작합니다.',
        '8. 모름은 독립적인 투표 결과입니다. 후보 이름만 보고 알려진 과일로 바꾸지 않습니다.','',
        '## 추적별 기록','',
        '| ID | 검출 프레임 범위 | 유효 관측 수 | 최초 안정 지연 | 원본 라벨 변화 | 안정 라벨 변화 | 안정 결과 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in confirmed:
        report.append(f"| {r['track_id']} | {r['first_frame']}~{r['last_frame']} | {r['voting_samples']} | {r['confirmation_delay_seconds']:.1f}초 | {r['raw_label_changes_all_views']} | {r['stable_label_changes']} | {', '.join(r['stable_label_counts'])} |")
    report += ['', '## 실행 비용','',
        f"- 매 프레임 비교 모드는 모델을 {c['classifier_calls']}회 호출했습니다. 안정화만 실행한 모드는 {p['classifier_calls']}회 호출했습니다.",
        f"- 이 컴퓨터의 이번 실행에서 비교 모드 전체 소요는 {c['wall_seconds']:.2f}초, 안정화 전용은 {p['wall_seconds']:.2f}초였습니다. 모델 로딩을 포함한 일회성 측정이므로 보장 FPS나 공정한 반복 벤치마크가 아닙니다.",
        '- 출력 영상의 30 FPS는 저장/재생 규격이며 추론 속도와 구분합니다.',
        '- 모델 가중치와 모름 기준은 변경하지 않았습니다. 기존 매 프레임 모드도 유지했습니다.','',
        '## 한계','',
        '- 색상 기반 검출은 잎이나 일부 과일을 분리하거나 붙은 과일을 하나로 잡을 수 있습니다.',
        '- 가림·교차·급격한 이동·같은 외관의 연속 물체에서 ID가 바뀌거나 섞이는 경우는 추가 검증이 필요합니다.',
        '- 내부 여백/크기 안정성만으로 실제 과일 전체가 보인다고 보장할 수는 없습니다.',
        '- 새로운 결과를 보여주기까지 약 0.4초 이상의 지연을 의도적으로 추가했습니다.','',
        '## 결과물·실행','',
        '- `fruit_conveyor_tracked.mp4`: 추적·장면 종합 결과.',
        '- `fruit_conveyor_tracking_before_after.mp4`: 동일 입력의 기존/개선 나란히 비교.',
        '- `fruit_conveyor_tracked_comparison.json`: 모든 원본 판정과 추적/투표 증거, 개체별 변화 횟수.',
        '- `fruit_conveyor_tracked.json`: 실제 안정화 전용 실행 기록. 두 실행의 프레임별 표시 결과가 같은지 확인했습니다.',
        '- 두 결과 영상과 비교 영상의 전체 461프레임을 다시 읽어 확인했습니다.','',
        '메인 메뉴 **7번**에서 실행하거나 다음 명령을 사용합니다.','',
        '```powershell',
        'python -m src.video --tracking --output outputs/video_experiment/fruit_conveyor_tracked.mp4 --display',
        'python -m src.video --tracking --compare-framewise --output outputs/video_experiment/fruit_conveyor_tracked_comparison.mp4',
        '```','',
        '같은 추적·투표 규칙을 다른 실험 모델에도 적용하려면 `--model-dir`를 함께 지정합니다. 이번 비교는 기존 발표용 모델 한 개만 사용해 시간적 처리 효과를 분리했습니다.']
    (out/'tracking_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    default=ROOT/'models/joint_nine_crop'
    assert hashlib.sha256((default/'model.pt').read_bytes()).hexdigest()=='986b5ed327e4c5b39f28a552d4da99b759659fe4b53c2eb178b7635a34abb565'
    assert hashlib.sha256((default/'policy.joblib').read_bytes()).hexdigest()=='041d60f4c9fde4e67dc41a3b8a5187dcb7b190308425a5af89689ecc602a32fa'
    print('Verified 461 frames in all outputs; production and comparison labels match; model unchanged')
    print(json.dumps({'metrics':metrics,'production_calls':p['classifier_calls'],'production_seconds':p['wall_seconds']},ensure_ascii=False))


if __name__=='__main__':
    main()
