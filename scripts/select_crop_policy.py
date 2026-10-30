"""Select a crop policy on development data; report frozen test results."""
# 캐시된 crop 특징으로 거부 정책을 보정·선택한 뒤 고정 정책의 Test 결과를 보고합니다.
# calibration은 임계값 계산에, selection은 후보 비교에 사용합니다.
from pathlib import Path
import argparse
import json
import shutil
from collections import Counter
import numpy as np
import joblib
from sklearn.metrics import f1_score
from src.joint import ROOT, MODEL_DIR, JointClassifier
from src.joint_core import CLASSES, FRUITS, dump, sha
from src.rejection import unfamiliarity_scores, calibrate_per_fruit, threshold_values, acceptance_mask


# 미검출도 평가에 포함하고 지원 과일 정확도와 미지원 과일 오수락을 따로 계산합니다.
def evaluate(audit, probabilities, scores, policy, phase):
    indices = [i for i,r in enumerate(audit) if r['phase']==phase and not r['excluded_demo_group']]
    known = [i for i in indices if audit[i]['fruit'] in FRUITS]
    unknown = [i for i in indices if audit[i]['fruit'] not in FRUITS]
    predicted, accepted = {}, {}
    accepted_mask = acceptance_mask(probabilities, scores, policy, CLASSES)
    for i in indices:
        if not np.isfinite(probabilities[i]).all():
            predicted[i],accepted[i] = 'no_detection',False
            continue
        label = CLASSES[int(probabilities[i].argmax())]
        fruit = label.split('_',1)[0]
        accepted[i] = bool(accepted_mask[i])
        predicted[i] = label if accepted[i] else 'unknown'
    truth = [audit[i]['fruit']+'_'+audit[i]['ripeness'] for i in known]
    present_labels = sorted(set(truth))
    result = {'known_count':len(known),'unknown_count':len(unknown),
        'class_support':{label:truth.count(label) for label in CLASSES},
        'supported_class_macro_f1':float(f1_score(truth,[predicted[i] for i in known],labels=present_labels,average='macro',zero_division=0)) if present_labels else None,
        'nine_class_macro_f1':float(f1_score(truth,[predicted[i] for i in known],labels=CLASSES,average='macro',zero_division=0)),
        'joint_accuracy':float(np.mean([predicted[i]==label for i,label in zip(known,truth)])),
        'known_rejection_or_missing_rate':float(np.mean([not accepted[i] for i in known])),
        'unknown_false_acceptance_rate':float(np.mean([accepted[i] for i in unknown])) if unknown else None,
        'missing_detection_count':sum(predicted[i]=='no_detection' for i in indices),
        'per_fruit':{}}
    for fruit in FRUITS:
        selected = [i for i in known if audit[i]['fruit']==fruit]
        result['per_fruit'][fruit] = {'count':len(selected),
            'macro_f1':float(f1_score([audit[i]['fruit']+'_'+audit[i]['ripeness'] for i in selected],
                                    [predicted[i] for i in selected],labels=[fruit+'_'+s for s in ('unripe','ripe','overripe')],average='macro',zero_division=0)),
            'rejection_or_missing_rate':float(np.mean([not accepted[i] for i in selected])) if selected else None}
    return result


def main(run):
    audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
    cache = np.load(run/'crop_features.npz')
    p,e = cache['probabilities'],cache['embeddings']
    baseline_cache = run/'original_crop_features.npz'
    baseline_data = np.load(baseline_cache) if baseline_cache.is_file() else cache
    baseline_p,baseline_e = baseline_data['probabilities'],baseline_data['embeddings']
    valid = np.isfinite(p).all(axis=1)
    # Test와 시연 유사 그룹은 정책 보정·선택 자료에서 제외합니다.
    development = [i for i,r in enumerate(audit) if valid[i] and r['phase']!='test' and not r['excluded_demo_group']]
    training = [i for i in development if audit[i]['phase']=='train' and audit[i]['fruit'] in FRUITS]
    reference = e[training]
    old = joblib.load(MODEL_DIR/'policy.joblib')
    def all_scores(ref, query_p, query_e):
        query_valid = np.isfinite(query_p).all(axis=1)
        scores = {method:np.full(len(audit),np.inf) for method in ('msp','embedding')}
        values = unfamiliarity_scores(query_p[query_valid],query_e[query_valid],ref)
        for method in scores:
            scores[method][query_valid] = values[method]
        return scores
    scores_old = all_scores(old['reference'],baseline_p,baseline_e)
    scores_crop = all_scores(reference,p,e)
    baseline = evaluate(audit,baseline_p,scores_old,old,'selection')
    dump(run/'baseline_selection.json',baseline)
    fruit_choice = np.array([FRUITS.index(CLASSES[int(p[i].argmax())].split('_',1)[0]) for i in development])
    try:
        thresholds = calibrate_per_fruit({k:v[development] for k,v in scores_crop.items()},np.eye(3)[fruit_choice],[audit[i] for i in development])
    except ValueError as error:
        dump(run/'calibration_failure.json',{'error':str(error),'baseline_selection':baseline,
            'calibration_kind_accuracy':{f:float(np.mean([CLASSES[int(p[i].argmax())].split('_',1)[0]==f for i in development if audit[i]['phase']=='calibration' and audit[i]['fruit']==f])) for f in FRUITS}})
        print('CALIBRATION_FAILED',error,flush=True)
        return
    comparisons = {}
    for method in ('msp','embedding'):
        policy = {'reference':reference,'thresholds':thresholds,'method':method}
        comparisons[method] = evaluate(audit,p,scores_crop,policy,'selection')
    # 오수락 최소화, F1 최대화, 거부 최소화 순으로 후보 정책을 비교합니다.
    key = lambda m:(m['unknown_false_acceptance_rate'],-m['nine_class_macro_f1'],m['known_rejection_or_missing_rate'])
    method = min(comparisons,key=lambda m:(key(comparisons[m]),m!='embedding'))
    policy = {'reference':reference,'thresholds':thresholds,'method':method}
    promote = key(comparisons[method]) < key(baseline)
    selection = {'baseline':baseline,'candidates':comparisons,'chosen_method':method,'promote_video_policy':promote,
                 'selection_rule':'minimize unknown false acceptance, maximize macro F1, minimize rejection; strict improvement over baseline required',
                 'thresholds':thresholds,'reference_count':len(reference),'test_used_for_selection':False,
                 'demo_groups_excluded':sorted({r['group'] for r in audit if r['excluded_demo_group']})}
    dump(run/'selection.json',selection)
    # Freeze policy and decision before reading any test metrics.
    joblib.dump(policy,run/'crop_policy.joblib')
    dump(run/'test_metrics.json',{'baseline':evaluate(audit,baseline_p,scores_old,old,'test'),
                                  'crop_calibrated':evaluate(audit,p,scores_crop,policy,'test')})
    if promote:
        destination = ROOT/'models/joint_nine_crop'
        if destination.exists():
            destination = run/'bundle'
        destination.mkdir(parents=True)
        checkpoint = run/'finetune/weights/last.pt'
        shutil.copy2(checkpoint if checkpoint.is_file() else MODEL_DIR/'model.pt',destination/'model.pt')
        shutil.copy2(run/'crop_policy.joblib',destination/'policy.joblib')
        shutil.copy2(MODEL_DIR/'split_manifest.json',destination/'split_manifest.json')
        dump(destination/'bundle.json',{'format':1,'imgsz':224,'model_name':'yolo26n_cls_joint_nine_crop',
             'source':str(run),'adaptation':'six-epoch crop fine-tuning and crop-domain reference/rejection calibration' if checkpoint.is_file() else 'crop-domain reference/rejection calibration',
             'sha256':{name:sha(destination/name) for name in ('model.pt','policy.joblib','split_manifest.json')}})
        JointClassifier(destination)
        dump(run/'installed.json',{'model_dir':str(destination)})
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)
    print((run/'test_metrics.json').read_text(encoding='utf-8'),flush=True)


def cli(argv=None):
    """명령행 인자를 해석하고 작업을 실행합니다. argv 생략 시 터미널 인자를 사용합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument('run',nargs='?',type=Path)
    args = parser.parse_args(argv)
    main(args.run or Path((ROOT/'outputs/crop_calibration/latest_run.txt').read_text(encoding='utf-8')))


if __name__ == "__main__":
    cli()
