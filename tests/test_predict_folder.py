import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.predict import predict_folder, infer_fruit, predict_loaded


class FolderPredictionTests(unittest.TestCase):
    def test_probability_called_once_and_class_order_respected(self):
        model = Mock()
        model.classes_ = [2, 0, 1]
        model.predict_proba.return_value = [[0.1, 0.7, 0.2]]
        bundle = {"label_mapping": {"unripe": 0, "ripe": 1, "overripe": 2}}
        with patch("src.predict.extract_image_features", return_value=[[1]]):
            result = predict_loaded(Path("photo.jpg"), bundle, model)
        self.assertEqual(result["ripeness"], "unripe")
        self.assertEqual(result["probabilities"]["unripe"], 0.7)
        model.predict_proba.assert_called_once()
        model.predict.assert_not_called()

    def test_model_without_probabilities_uses_predict(self):
        model = Mock(spec=["predict"])
        model.predict.return_value = [1]
        with patch("src.predict.extract_image_features", return_value=[[1]]):
            result = predict_loaded(Path("photo.jpg"), {"label_mapping": {"ripe": 1}}, model)
        self.assertEqual(result["ripeness"], "ripe")
        self.assertIsNone(result["probabilities"])
        model.predict.assert_called_once()

    def test_load_once_filter_and_continue_after_failure(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            for name in ("banana_a.jpg", "tomato_b.PNG", "notes.txt"):
                (root / name).touch()
            result = dict(ripeness="ripe", probabilities=None, feature_ms=20.,
                          prediction_ms=5., total_ms=25.)
            with patch("src.predict.load_model", return_value={}) as load, \
                 patch("src.predict.select_model", return_value=(Mock(), "test")), \
                 patch("src.predict.predict_loaded", side_effect=[result, ValueError("broken")]) as process, \
                 contextlib.redirect_stdout(io.StringIO()):
                rows, summary = predict_folder(root, fruit_mode="filename")
            self.assertEqual(load.call_count, 1)
            self.assertEqual(process.call_count, 2)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(rows[1]["error"], "broken")
            self.assertGreaterEqual(summary["mean_ms"], 0.)
            self.assertEqual(rows[0]["fruit"], "banana")

    def test_filename_rules(self):
        self.assertEqual(infer_fruit("20261008_mandarin_M09.jpg"), "mandarin")
        self.assertEqual(infer_fruit("Overripe_109_jpg.rf.ab12.jpg"), "banana")
        self.assertEqual(infer_fruit("TOMATO_test.png"), "tomato")
        for name in ("unknown.jpg", "ripe.jpg", "banana_tomato.jpg"):
            with self.assertRaises(ValueError):
                infer_fruit(name)

    def test_empty_folder_does_not_load_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            with patch("src.predict.load_model") as load:
                with self.assertRaises(ValueError):
                    predict_folder(directory)
                load.assert_not_called()
