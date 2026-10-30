"""Crop evaluation must count missed fruit and keep unknown acceptance separate."""
import unittest
from unittest.mock import Mock
import numpy as np
from scripts.select_crop_policy import evaluate
from src.joint_core import CLASSES, NinePredictor


class CropEvaluationTests(unittest.TestCase):
    def test_runtime_and_evaluation_use_the_same_class_threshold(self):
        encoder = object.__new__(NinePredictor)
        p = np.zeros(len(CLASSES))
        p[CLASSES.index('banana_overripe')] = 1
        encoder.raw = Mock(return_value=(p, np.array([1., 0.])))
        encoder.policy = {'method':'embedding','reference':np.array([[.9, np.sqrt(.19)]]),
            'thresholds':{'embedding':{'banana':.2}},
            'class_thresholds':{'banana_overripe':.05}}
        prediction = encoder.predict('unused')
        self.assertEqual(prediction['status'],'unknown')
        self.assertEqual(prediction['threshold'],.05)
        audit = [{'phase':'selection','excluded_demo_group':False,'fruit':'banana','ripeness':'overripe'}]
        metrics = evaluate(audit,p[None],{'embedding':np.array([.1])},encoder.policy,'selection')
        self.assertEqual(metrics['known_rejection_or_missing_rate'],1)
        del encoder.policy['class_thresholds']
        self.assertEqual(encoder.predict('unused')['status'],'ok')

    def test_missing_known_is_a_failure_and_excluded_groups_are_ignored(self):
        audit = [
            {'phase':'selection','excluded_demo_group':False,'fruit':'banana','ripeness':'ripe'},
            {'phase':'selection','excluded_demo_group':False,'fruit':'grape','ripeness':'unknown'},
            {'phase':'test','excluded_demo_group':False,'fruit':'banana','ripeness':'ripe'},
            {'phase':'selection','excluded_demo_group':True,'fruit':'banana','ripeness':'ripe'}]
        p = np.full((4,len(CLASSES)),np.nan)
        p[1] = 0
        p[1,CLASSES.index('banana_ripe')] = 1
        scores = {'embedding':np.array([np.inf,.1,.1,.1])}
        policy = {'method':'embedding','thresholds':{'embedding':{'banana':.2}}}
        result = evaluate(audit,p,scores,policy,'selection')
        self.assertEqual(result['known_count'],1)
        self.assertEqual(result['unknown_count'],1)
        self.assertEqual(result['missing_detection_count'],1)
        self.assertEqual(result['known_rejection_or_missing_rate'],1)
        self.assertEqual(result['unknown_false_acceptance_rate'],1)
        self.assertEqual(result['joint_accuracy'],0)


if __name__ == '__main__':
    unittest.main()
