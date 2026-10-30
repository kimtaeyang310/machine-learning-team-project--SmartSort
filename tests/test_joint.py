"""Regression tests for promotion and unknown-result suppression."""
# 모델 배포와 unknown 판정 시 숙도를 숨기는 동작을 검사합니다.
import unittest
from unittest.mock import Mock, patch
from src.joint import JointClassifier, timing_summary


class JointTests(unittest.TestCase):
    def test_default_folder_entry_uses_joint_not_legacy_models(self):
        from src.predict import predict_folder
        with patch("src.joint.predict_folder", return_value=([], {})) as joint:
            self.assertEqual(predict_folder("photos"), ([], {}))
            joint.assert_called_once_with("photos")

    def test_default_single_entry_suppresses_unknown_stage(self):
        from src.predict import predict
        with patch("src.joint.JointClassifier") as classifier:
            classifier.return_value.predict.return_value = {"status": "unknown", "ripeness": None}
            self.assertEqual(predict("photo.jpg"), "unknown")

    def predictor(self, accepted):
        classifier = object.__new__(JointClassifier)
        classifier.sha = "test"
        classifier.model_name = "yolo26n_cls_joint_nine"
        classifier.engine = Mock()
        classifier.engine.predict.return_value = {
            "status": "ok" if accepted else "unknown", "candidate_fruit": "banana",
            "candidate_ripeness": "ripe", "candidate_class": "banana_ripe",
            "score": .1, "threshold": .2, "method": "embedding",
            "probabilities": {"banana_ripe": .6, "banana_unripe": .1, "banana_overripe": .1,
                              "tomato_ripe": .2}}
        return classifier

    def test_unknown_hides_ripeness_and_sends_review(self):
        result = self.predictor(False).predict("unused.jpg")
        self.assertEqual(result["fruit"], "unknown")
        self.assertIsNone(result["ripeness"])
        self.assertIsNone(result["probabilities"])
        self.assertEqual(result["agent_decision"]["action"], "human_review")

    def test_joint_probability_is_not_renormalized(self):
        result = self.predictor(True).predict("unused.jpg")
        self.assertEqual(result["model"], "yolo26n_cls_joint_nine")
        self.assertEqual(result["probabilities"]["ripe"], .6)
        self.assertEqual(result["agent_decision"]["action"], "retake")
        self.assertEqual(result["nine_probabilities"]["tomato_ripe"], .2)

    def test_failure_time_is_included_without_counting_failure_as_throughput(self):
        result = timing_summary([{"status": "ok", "total_ms": 10},
                                 {"status": "unknown", "total_ms": 10},
                                 {"status": "failed", "total_ms": 80}])
        self.assertEqual(result["fps"], 20)
        self.assertEqual(result["failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
