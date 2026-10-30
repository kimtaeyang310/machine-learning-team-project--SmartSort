"""Hybrid rejection must preserve legacy gates and enforce rescue bounds."""
import unittest
import numpy as np
from src.joint_core import CLASSES,NinePredictor
from src.rejection import acceptance_mask
from unittest.mock import Mock


class HybridRejectionTests(unittest.TestCase):
    def policy(self):
        return {'reference':np.array([[.7,np.sqrt(.51)]]),'method':'embedding',
                'thresholds':{'embedding':{'banana':.1,'mandarin':.1,'tomato':.1}}}

    def probabilities(self, confidence=.98):
        p=np.zeros((1,len(CLASSES)))
        p[0,CLASSES.index('mandarin_ripe')]=confidence
        p[0,CLASSES.index('banana_ripe')]=1-confidence
        return p

    def test_bounded_rescue_requires_strong_scores_and_near_enough_embedding(self):
        policy=self.policy()
        p=self.probabilities()
        scores={'embedding':np.array([.3])}
        self.assertFalse(acceptance_mask(p,scores,policy,CLASSES)[0])
        policy['acceptance']={'kind':'rescue','distance_multiplier':4.,'min_confidence':.95,'min_margin':.8,'min_fruit_confidence':.95}
        self.assertTrue(acceptance_mask(p,scores,policy,CLASSES)[0])
        self.assertFalse(acceptance_mask(p,{'embedding':np.array([.5])},policy,CLASSES)[0])
        self.assertFalse(acceptance_mask(self.probabilities(.6),scores,policy,CLASSES)[0])
        self.assertFalse(acceptance_mask(p,{'embedding':np.array([np.inf])},policy,CLASSES)[0])

    def test_runtime_uses_same_rescue_as_evaluation(self):
        policy=self.policy()
        policy['acceptance']={'kind':'rescue','distance_multiplier':4.,'min_confidence':.95,'min_margin':.8}
        encoder=object.__new__(NinePredictor)
        encoder.policy=policy
        encoder.raw=Mock(return_value=(self.probabilities()[0],np.array([1.,0.])))
        result=encoder.predict('unused')
        self.assertEqual(result['status'],'ok')
        self.assertGreater(result['score'],result['threshold'])
        self.assertEqual(result['acceptance_rule']['kind'],'rescue')

    def test_rescue_retains_original_acceptance_even_if_confidence_is_lower(self):
        policy=self.policy()
        policy['acceptance']={'kind':'rescue','distance_multiplier':2.,'min_confidence':.99}
        self.assertTrue(acceptance_mask(self.probabilities(.6),{'embedding':np.array([.05])},policy,CLASSES)[0])


if __name__=='__main__':
    unittest.main()
