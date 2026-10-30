"""Metric and isolation checks for the separate experiment only."""
# 실험 지표 계산과 원본 파일로부터의 분리 조건을 검증합니다.
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment as experiment


def perfect_rows():
    rows = []
    for fruit in experiment.FRUITS:
        for stage in experiment.STAGES:
            rows.append({"truth_fruit": fruit, "truth_ripeness": stage, "truth_class": f"{fruit}_{stage}",
                         "status": "ok", "candidate_fruit": fruit, "candidate_ripeness": stage,
                         "candidate_class": f"{fruit}_{stage}", "total_ms": 10.})
    rows.append({"truth_fruit": "grape", "truth_ripeness": "", "truth_class": "unknown",
                 "status": "unknown", "candidate_fruit": "tomato", "candidate_ripeness": "ripe",
                 "candidate_class": "tomato_ripe", "total_ms": 10.})
    return rows


class ExperimentTests(unittest.TestCase):
    def test_embedding_hook_survives_backend_copy_without_second_forward(self):
        import copy
        import threading
        from types import SimpleNamespace
        import numpy as np
        import torch

        class FakeYOLO:
            def __init__(self, checkpoint):
                head = torch.nn.Module()
                head.linear = torch.nn.Linear(4, 9, bias=False)
                torch.nn.init.zeros_(head.linear.weight)
                self.model = torch.nn.Module()
                self.model.model = torch.nn.ModuleList([head])
                self.lock = threading.Lock()
                self.forward_count = 0

            def predict(self, *args, **kwargs):
                backend = copy.deepcopy(self.model)
                features = torch.tensor([[1., 2., 3., 4.]])
                logits = backend.model[-1].linear(features)
                self.forward_count += 1
                return [SimpleNamespace(names=dict(enumerate(experiment.CLASSES)),
                        probs=SimpleNamespace(data=logits.softmax(1)[0]))]

        with patch.object(experiment, "YOLO", FakeYOLO):
            predictor = experiment.NinePredictor("fixture.pt", 224)
            probabilities, embedding = predictor.raw("not_read.jpg")
        self.assertEqual(predictor.network.forward_count, 1)
        np.testing.assert_allclose(probabilities, np.full(9, 1 / 9), rtol=1e-6)
        np.testing.assert_allclose(embedding, np.array([1, 2, 3, 4]) / np.sqrt(30), rtol=1e-6)

    def test_rejected_known_photo_is_not_dropped_from_f1(self):
        rows = perfect_rows()
        self.assertEqual(experiment.performance_metrics(rows)["nine_class_macro_f1"], 1.)
        rows[0]["status"] = "unknown"
        metrics = experiment.performance_metrics(rows)
        self.assertAlmostEqual(metrics["nine_class_macro_f1"], 8 / 9)
        self.assertEqual(metrics["known_false_rejection_count"], 1)
        self.assertAlmostEqual(metrics["known_false_rejection_rate"], 1 / 9)

    def test_wrong_fruit_with_same_stage_is_a_ripeness_failure(self):
        rows = perfect_rows()
        rows[0].update(candidate_fruit="tomato", candidate_class="tomato_unripe")
        metrics = experiment.performance_metrics(rows)
        self.assertAlmostEqual(metrics["per_fruit"]["banana"]["ripeness_macro_f1"], 2 / 3)
        self.assertAlmostEqual(metrics["joint_accuracy_including_rejections"], 8 / 9)

    def test_unknown_acceptance_and_processing_failure_are_separate(self):
        rows = perfect_rows()
        rows[-1]["status"] = "ok"
        rows[0]["status"] = "failed"
        metrics = experiment.performance_metrics(rows)
        self.assertEqual(metrics["unknown_false_acceptance_rate"], 1.)
        self.assertEqual(metrics["known_false_rejection_count"], 0)
        self.assertEqual(metrics["failure_count"], 1)
        self.assertAlmostEqual(metrics["timing"]["fps"], 100.)

    def test_materialization_never_uses_calibration_selection_or_test(self):
        with tempfile.TemporaryDirectory(dir=experiment.HERE / "runtime" / "tmp") as directory:
            root = Path(directory)
            samples = []
            for stage in experiment.STAGES:
                for i in range(2):
                    path = f"{stage}_{i}.jpg"
                    (root / path).write_bytes(b"fixture")
                    samples.append({"phase": "train", "fruit": "tomato", "ripeness": stage, "path": path})
            for phase in ("calibration", "selection", "test"):
                samples.append({"phase": phase, "fruit": "tomato", "ripeness": "ripe", "path": "DO_NOT_READ.jpg"})
            with patch.object(experiment, "ROOT", root):
                out = experiment.materialize(samples, root / "data", "tomato")
            self.assertEqual(len(list((out / "train").rglob("*.jpg"))), 6)
            self.assertEqual(len(list((out / "val").rglob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
