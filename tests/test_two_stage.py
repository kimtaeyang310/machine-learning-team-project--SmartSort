"""Two-stage routing and refusal tests, without model weights."""
# 실제 가중치 없이 종류별 숙도 모델 연결과 거부 결과 처리를 검사합니다.
import unittest
from unittest.mock import Mock
import numpy as np
from src.two_stage import TwoStageClassifier,kind_gate,FRUITS,STAGES
from scripts.select_two_stage import evaluate


class TwoStageTests(unittest.TestCase):
    def classifier(self, accepted=True):
        model=object.__new__(TwoStageClassifier)
        model.kind=Mock()
        model.kind.raw.return_value=(np.array([.01,.98,.01]),np.array([1.,0.]))
        model.stages={fruit:Mock() for fruit in FRUITS}
        for head in model.stages.values():
            head.raw.return_value=(np.array([.1,.8,.1]),None)
        model.policy={'reference':np.array([[1.,0.]]),'thresholds':{f:.2 for f in FRUITS},
            'gate':{'kind':'confidence','min_confidence':.9 if accepted else .99}}
        model.sha='test'
        model.model_name='two_stage_test'
        return model

    def test_routes_only_to_selected_fruit_stage_model(self):
        model=self.classifier()
        result=model.predict_frame(np.zeros((40,40,3),np.uint8))
        self.assertEqual((result['fruit'],result['ripeness']),('mandarin','ripe'))
        model.stages['mandarin'].raw.assert_called_once()
        model.stages['banana'].raw.assert_not_called()
        model.stages['tomato'].raw.assert_not_called()
        self.assertAlmostEqual(sum(result['probabilities'].values()),1.)

    def test_unknown_kind_does_not_run_any_ripeness_head(self):
        model=self.classifier(accepted=False)
        result=model.predict_frame(np.zeros((40,40,3),np.uint8))
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['ripeness'])
        self.assertIsNone(result['probabilities'])
        for head in model.stages.values():
            head.raw.assert_not_called()

    def test_uncertain_stage_is_reviewed_without_erasing_known_fruit(self):
        model=self.classifier()
        model.stages['mandarin'].raw.return_value=(np.array([.4,.35,.25]),None)
        result=model.predict_frame(np.zeros((40,40,3),np.uint8))
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['fruit'],'mandarin')
        self.assertEqual(result['agent_decision']['action'],'retake')

    def test_wrong_kind_cannot_count_as_correct_joint_prediction(self):
        audit=[{'phase':'selection','excluded_demo_group':False,'fruit':'mandarin','ripeness':'ripe'}]
        p=np.array([[.1,.1,.8]])
        stage=np.array([[.1,.8,.1]])
        policy={'reference':np.array([[1.,0.]]),'thresholds':{f:1. for f in FRUITS},'gate':{'kind':'distance'}}
        metric=evaluate(audit,p,stage,np.array([.1]),policy,'selection')
        self.assertEqual(metric['kind_accuracy_before_gate_including_missing'],0)
        self.assertEqual(metric['joint_accuracy'],0)
        self.assertIsNone(metric['stage_accuracy_on_correct_kind'])


if __name__=='__main__':
    unittest.main()
