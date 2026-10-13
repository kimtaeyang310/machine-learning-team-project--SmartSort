import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.model_bundle import FRUITS, select_model, validate_bundle
from src.train import fit_fruit_models
from src.evaluate import evaluate, load_inputs
from src.predict import predict
from src.performance_history import sample_id


class FruitModelTests(unittest.TestCase):
    def bundle(self):
        return {
            "models": {
                fruit: {"fruit": fruit, "model_name": fruit,
                        "model": Mock()} for fruit in FRUITS
            },
            "feature_names": ["x"],
            "preprocess_config": {},
            "label_mapping": {"unripe": 0, "ripe": 1, "overripe": 2},
        }

    def test_training_never_mixes_fruits(self):
        samples, values = [], []
        for fruit_index, fruit in enumerate(FRUITS):
            for label in range(3):
                for group in range(6):
                    samples.append(dict(fruit=fruit, ripeness_label=label,
                                        group_id=f"{fruit}_{label}_{group}"))
                    values.append(fruit_index * 100 + label)
        seen = []

        def check_cv(model, X, y, **kwargs):
            seen.append(set((X.x // 100).astype(int)))
            self.assertEqual(len(seen[-1]), 1)
            for train, valid in kwargs["cv"]:
                self.assertFalse(set(train) & set(valid))
                self.assertEqual(set(y[train]), {0, 1, 2})
            return np.array([0.5, 0.6, 0.7])

        with patch("src.train.get_models", return_value={"dummy": DummyClassifier()}), \
             patch("src.train.cross_val_score", side_effect=check_cv), \
             contextlib.redirect_stdout(io.StringIO()):
            models, comparisons = fit_fruit_models(pd.DataFrame({"x": values}), samples)
        self.assertEqual(seen, [{0}, {1}, {2}])
        self.assertEqual(set(models), set(FRUITS))
        for fruit in FRUITS:
            self.assertEqual(models[fruit]["train_count"], 18)
            self.assertEqual(comparisons[fruit].iloc[0]["model"], "dummy")

    def test_evaluation_routes_interleaved_rows(self):
        bundle = self.bundle()
        fruits = ["tomato", "banana", "mandarin", "tomato", "banana"]
        labels = [2, 0, 1, 2, 0]
        frame = pd.DataFrame({"fruit": fruits, "ripeness_label": labels,
                              "image_path": [f"{i}.jpg" for i in range(5)],
                              "group_id": [str(i) for i in range(5)]})
        for fruit, label in zip(FRUITS, range(3)):
            bundle["models"][fruit]["model"].predict.side_effect = (
                lambda X, label=label: np.full(len(X), label))
        with tempfile.TemporaryDirectory() as directory, \
             patch("src.evaluate.load_inputs", return_value=(bundle, frame)), \
             patch("src.evaluate.extract_test_features", return_value=pd.DataFrame({"x": range(5)})), \
             contextlib.redirect_stdout(io.StringIO()):
            summary, _ = evaluate(output_root=directory)
        self.assertTrue(summary.accuracy.eq(1).all())
        tomato_input = bundle["models"]["tomato"]["model"].predict.call_args.args[0]
        self.assertEqual(tomato_input.x.tolist(), [0, 3])

    def test_missing_ripeness_stops_before_training(self):
        samples = []
        for fruit in FRUITS:
            for label in ([0, 1] if fruit == "tomato" else [0, 1, 2]):
                for group in range(6):
                    samples.append(dict(fruit=fruit, ripeness_label=label,
                                        group_id=f"{fruit}_{label}_{group}"))
        with patch("src.train.cross_val_score") as score:
            with self.assertRaisesRegex(ValueError, "tomato"):
                fit_fruit_models(pd.DataFrame({"x": range(len(samples))}), samples)
            score.assert_not_called()

    def test_missing_fruit_and_incomplete_bundle_rejected(self):
        bundle = self.bundle()
        with self.assertRaises(ValueError):
            select_model(bundle, None)
        with self.assertRaises(ValueError):
            select_model(bundle, "apple")
        del bundle["models"]["tomato"]
        with self.assertRaises(ValueError):
            validate_bundle(bundle)

    def test_prediction_uses_requested_fruit(self):
        bundle = self.bundle()
        model = DummyClassifier(strategy="constant", constant=2).fit([[0], [1], [2]], [0, 1, 2])
        bundle["models"]["tomato"]["model"] = model
        with patch("src.predict.load_model", return_value=bundle), \
             patch("src.predict.resolve_image_path", return_value=Path("photo.jpg")), \
             patch("src.predict.extract_image_features", return_value=np.array([[0]])), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(predict("photo.jpg", fruit="tomato"), "overripe")
        bundle["models"]["banana"]["model"].predict.assert_not_called()

    def test_changed_split_with_same_count_rejected(self):
        train = pd.DataFrame([dict(image_path="a.jpg", fruit="banana", group_id="a",
                                   split="train", ripeness="unripe", ripeness_label=0)])
        test = pd.DataFrame([dict(image_path="b.jpg", fruit="banana", group_id="b",
                                  split="test", ripeness="ripe", ripeness_label=1)])
        bundle = self.bundle()
        # Use a real serializable model for the persistence check.
        for entry in bundle["models"].values():
            entry["model"] = DummyClassifier()
        bundle.update(train_count=1, test_count=1,
                      train_dataset_id=sample_id(train), test_dataset_id=sample_id(test))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joblib.dump(bundle, root / "model.joblib")
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            load_inputs(root / "model.joblib", root / "train.csv", root / "test.csv")
            test.loc[0, "image_path"] = "changed.jpg"
            test.to_csv(root / "test.csv", index=False)
            with self.assertRaises(ValueError):
                load_inputs(root / "model.joblib", root / "train.csv", root / "test.csv")


if __name__ == "__main__":
    unittest.main()
