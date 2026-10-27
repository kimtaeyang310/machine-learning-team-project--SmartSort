import tempfile
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from src.performance_history import rebuild_evaluation_history, record_training, sample_id, EVALUATION_COLUMNS


class HistoryTests(unittest.TestCase):
    def samples(self):
        return pd.DataFrame([
            dict(image_path="a.jpg", fruit="banana", group_id="a", ripeness_label=0),
            dict(image_path="b.jpg", fruit="banana", group_id="b", ripeness_label=1),
        ])

    def test_legacy_rebuild_and_changed_test_set(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            for i, score in enumerate([0.5, 0.75, 1.0]):
                run = root / f"20261008_12000{i}_000000"
                run.mkdir()
                samples = self.samples()
                if i == 2:
                    samples.loc[0, "ripeness_label"] = 2
                samples.to_csv(run / "test_samples.csv", index=False)
                pd.DataFrame([dict(scope="overall", accuracy=score, macro_f1=score)]).to_csv(run / "metrics.csv", index=False)
            (root / "20261008_130000_000000").mkdir()  # 실패 실행: metrics 없음
            first = rebuild_evaluation_history(root)
            second = rebuild_evaluation_history(root)
            self.assertEqual(len(second), 3)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(list(second.columns), EVALUATION_COLUMNS)
            self.assertEqual(second.iloc[0].dataset_id, second.iloc[1].dataset_id)
            self.assertNotEqual(second.iloc[1].dataset_id, second.iloc[2].dataset_id)
            ElementTree.parse(root / "evaluation_history.svg")

    def test_training_runs_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            comparison = pd.DataFrame([dict(model="svm", cv_macro_f1_mean=0.8, cv_macro_f1_std=0.02)])
            one = record_training(comparison, self.samples().to_dict("records"), directory)
            comparison.loc[0, "cv_macro_f1_mean"] = 0.9
            two = record_training(comparison, self.samples().to_dict("records"), directory)
            self.assertNotEqual(one, two)
            self.assertEqual(pd.read_csv(one / "model_comparison.csv").iloc[0].cv_macro_f1_mean, 0.8)
            history = pd.read_csv(Path(directory) / "training_history.csv")
            self.assertEqual(len(history), 2)
            ElementTree.parse(Path(directory) / "training_history.svg")


    def test_sample_identity_ignores_order(self):
        self.assertEqual(sample_id(self.samples()), sample_id(self.samples().iloc[::-1]))


if __name__ == "__main__":
    unittest.main()
