"""Track fruit and aggregate only sufficiently interior, size-stable views."""
from pathlib import Path
from collections import Counter,defaultdict
import json
from time import perf_counter
import cv2
import numpy as np
from src.joint import ROOT,MODEL_DIR
from src.video import detect_regions,crop_region,annotate,load_font,load_classifier
from src.tracking import FruitTracker,TemporalVote,interior_observation,result_key
from src.paths import project_path


def count_changes(keys):
    return sum(a!=b for a,b in zip(keys,keys[1:]))


def process_tracked_video(source,destination,*,display=False,min_area=500,model_dir=None,compare_framewise=False):
    source,destination=project_path(source),project_path(destination)
    if source==destination:
        raise ValueError('Input and output video must be different files')
    destination.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    writer=None
    records=[];per_track=defaultdict(list);times=[];raw_calls=0
    started=perf_counter()
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:
            raise ValueError('Invalid source FPS')
        if model_dir is None:
            model_dir=ROOT/'models/joint_nine_crop'
            if not (model_dir/'bundle.json').is_file():
                model_dir=MODEL_DIR
        classifier,font=load_classifier(model_dir),load_font()
        tracker=FruitTracker(max_missed=max(2,round(fps*.2)))
        vote=TemporalVote(fps)
        index=0
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            tick=perf_counter()
            if writer is None:
                height,width=frame.shape[:2]
                writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
                if not writer.isOpened():
                    raise RuntimeError('Cannot open output video writer')
            boxes=detect_regions(frame,min_area)
            tracks=tracker.update(boxes,frame,index)
            detections=[]
            for box,track in zip(boxes,tracks):
                eligible=interior_observation(track,frame.shape)
                sample=vote.needs_sample(track,index,eligible)
                raw=None
                if compare_framewise or sample:
                    try:
                        raw=classifier.predict_frame(crop_region(frame,box))
                    except (ValueError,OSError,RuntimeError) as error:
                        raw={'status':'failed','error':str(error)}
                    raw_calls+=1
                if sample:
                    vote.add(track,raw,index)
                prediction=vote.display(track,index,eligible)
                row={'box':list(box),'track_id':track.id,'eligible_full_view':eligible,
                    'track_confirmed':track.hits>=3,
                    'used_for_vote':bool(sample and result_key(raw) is not None),
                    'prediction':prediction,'raw_prediction':raw}
                detections.append(row)
                per_track[track.id].append({'frame':index,**row})
            drawn=[r for r in detections if r['track_confirmed']]
            rendered=annotate(frame,drawn,font)
            writer.write(rendered)
            times.append((perf_counter()-tick)*1000)
            records.append({'frame':index,'time_seconds':index/fps,'detections':detections})
            if index in (6,30,90,150,210,270,330,390,450):
                cv2.imencode('.jpg',rendered)[1].tofile(str(destination.with_name(f'{destination.stem}_frame_{index:04d}.jpg')))
            if display:
                cv2.imshow('SmartSort tracking - Q to stop',rendered)
                if cv2.waitKey(max(1,round(1000/fps)))&0xFF==ord('q'):
                    index+=1
                    break
            index+=1
            if index%90==0:
                print('Tracked frames',index,flush=True)
        if index==0:
            raise ValueError('Video has no readable frames')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    summaries=[]
    for track_id,rows in per_track.items():
        raw=[result_key(r['raw_prediction']) for r in rows if result_key(r['raw_prediction']) is not None]
        raw_full=[result_key(r['raw_prediction']) for r in rows if r['eligible_full_view'] and result_key(r['raw_prediction']) is not None]
        stable_rows=[r for r in rows if result_key(r['prediction']) is not None]
        stable=[result_key(r['prediction']) for r in stable_rows]
        visible=[result_key(r['prediction']) or ('pending',r['prediction']['temporal_state']) for r in rows]
        first_full=next((r['frame'] for r in rows if r['eligible_full_view']),None)
        first_stable=stable_rows[0]['frame'] if stable_rows else None
        summaries.append({'track_id':track_id,'first_frame':rows[0]['frame'],'last_frame':rows[-1]['frame'],
            'visible_frames':len(rows),'eligible_frames':sum(r['eligible_full_view'] for r in rows),
            'drawn_frames':sum(r['track_confirmed'] for r in rows),
            'voting_samples':sum(r['used_for_vote'] for r in rows),
            'edge_samples_used':sum(r['used_for_vote'] and not r['eligible_full_view'] for r in rows),
            'raw_label_changes_all_views':count_changes(raw) if compare_framewise else None,
            'raw_label_changes_interior_only':count_changes(raw_full) if compare_framewise else None,
            'stable_label_changes':count_changes(stable),'display_changes_including_pending':count_changes(visible),
            'first_eligible_frame':first_full,'first_stable_frame':first_stable,
            'confirmation_delay_seconds':(first_stable-first_full)/fps if first_stable is not None and first_full is not None else None,
            'stable_display_frames':len(stable_rows),
            'stable_display_coverage':len(stable_rows)/len(rows),
            'stable_label_counts':dict(Counter('/'.join(result_key(r['prediction'])) for r in stable_rows))})
    metrics={'track_count':len(summaries),'raw_label_changes':sum(r['raw_label_changes_all_views'] or 0 for r in summaries) if compare_framewise else None,
        'confirmed_track_count':sum(r['drawn_frames']>0 for r in summaries),
        'suppressed_short_track_count':sum(r['drawn_frames']==0 for r in summaries),
        'raw_interior_label_changes':sum(r['raw_label_changes_interior_only'] or 0 for r in summaries) if compare_framewise else None,
        'stable_label_changes':sum(r['stable_label_changes'] for r in summaries),
        'edge_samples_used':sum(r['edge_samples_used'] for r in summaries),
        'voting_samples':sum(r['voting_samples'] for r in summaries),
        'stable_display_frames':sum(r['stable_display_frames'] for r in summaries),
        'total_detected_object_frames':sum(r['visible_frames'] for r in summaries),
        'drawn_object_frames':sum(r['drawn_frames'] for r in summaries),
        'pending_object_frames':sum(r['visible_frames']-r['stable_display_frames'] for r in summaries)}
    summary={'source':str(source),'output':str(destination),'frames':index,'fps':fps,'duration_seconds':index/fps,
        'wall_seconds':perf_counter()-started,'mean_processing_ms':float(np.mean(times)),
        'classifier_calls':raw_calls,'model_dir':str(classifier.directory),'model_sha256':classifier.sha,
        'compare_framewise':compare_framewise,
        'tracking':{'association':'one-to-one constant-velocity center + size + foreground HSV appearance',
            'max_missed_frames':tracker.max_missed,'sample_interval_frames':vote.interval,
            'min_supporting_samples':vote.min_samples,'window_samples':vote.window,'required_support_ratio':vote.support,
            'switch_confirmations':vote.switch_confirmations,'max_hold_frames':vote.max_age,
            'track_confirmation_hits':3,
            'full_view':'bbox and 18% crop margin inside frame, 3 consecutive size-stable observations'},
        'metrics':metrics,'per_track':summaries,
        'limitations':'Interior/size checks are heuristics, not proof of complete visibility. Stable does not mean correct. No manual track or ripeness ground truth; HSV proposals can fragment/merge. Comparison mode includes extra framewise inference, so timing is not tracking-only performance.'}
    destination.with_suffix('.json').write_text(json.dumps({'summary':summary,'frames':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary
