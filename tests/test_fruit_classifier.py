import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.fruit_classifier import (rejection_threshold, unfamiliarity_scores, rejection_metrics,
                                 prepare_identity_image, threshold_values, FruitClassifier, FRUITS)
from src.fruit_classifier import LinearSoftmaxClassifier
from src.predict import predict, predict_folder, save_prediction_artifacts
from src.train_fruit_classifier import select_method, split_unknown, assign_identity_splits
from src.fruit_identity_experiment import specimen_weights, calibrate_per_fruit, candidate_key


class UnknownDetectionTests(unittest.TestCase):
    def test_portable_head_matches_trained_multinomial_probabilities(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(42)
        features = rng.normal(size=(90, 4))
        labels = np.array(FRUITS)[np.tile(np.arange(3), 30)]
        trained = LogisticRegression(max_iter=1000).fit(features, labels)
        portable = LinearSoftmaxClassifier.from_estimator(trained)
        probes = np.r_[features, features * 1000]
        np.testing.assert_allclose(portable.predict_proba(probes), trained.predict_proba(probes), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(portable.predict(probes), trained.predict(probes))
        self.assertTrue(np.isfinite(portable.predict_proba(probes)).all())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            import joblib
            target = Path(directory) / "portable.joblib"
            joblib.dump(portable, target, protocol=4)
            np.testing.assert_allclose(joblib.load(target).predict_proba(probes), portable.predict_proba(probes))

    def test_portable_head_refuses_ovr_and_invalid_features(self):
        with self.assertRaises(ValueError):
            LinearSoftmaxClassifier.from_estimator(Mock(solver="liblinear", multi_class="ovr"))
        model = LinearSoftmaxClassifier(np.eye(3), np.zeros(3), FRUITS)
        with self.assertRaises(ValueError):
            model.predict_proba([[float("nan"), 0, 0]])
        with self.assertRaises(ValueError):
            model.predict_proba([[1, 2]])

    def test_dates_of_same_specimen_never_cross_partitions(self):
        samples = [{"fruit": fruit, "ripeness": stage, "group": f"{fruit}:{i}", "phase": "train",
                    "date": date}
                   for fruit in FRUITS for i in range(10)
                   for stage in ("unripe", "ripe", "overripe") for date in ("2026-10-01", "2026-10-02")]
        samples += [{"fruit": fruit, "ripeness": "ripe", "group": fruit + ":heldout", "phase": "test"}
                    for fruit in FRUITS]
        assign_identity_splits(samples)
        for group in {s["group"] for s in samples}:
            self.assertEqual(len({s["phase"] for s in samples if s["group"] == group}), 1)
        self.assertTrue(all(s["phase"] == "test" for s in samples if s["group"].endswith(":heldout")))
        self.assertEqual({s["phase"] for s in samples}, {"train", "calibration", "selection", "test"})

    def test_repeated_specimen_does_not_dominate_weight(self):
        samples = [{"fruit": "tomato", "ripeness": "ripe", "group": "T01"}] * 10
        samples += [{"fruit": "tomato", "ripeness": "ripe", "group": "T02"}]
        weights = specimen_weights(samples)
        self.assertAlmostEqual(weights[:10].sum(), weights[10])

    def test_two_ripe_specimens_are_not_duplicated_to_fill_three_partitions(self):
        samples = [{"fruit": fruit, "ripeness": "ripe" if i < 2 else "unripe" if i < 5 else "overripe",
                    "group": f"{fruit}:{i}", "phase": "train"} for fruit in FRUITS for i in range(15)]
        assign_identity_splits(samples)
        for fruit in FRUITS:
            phases = {s["phase"] for s in samples if s["fruit"] == fruit and s["ripeness"] == "ripe"}
            self.assertEqual(phases, {"train", "calibration"})

    def test_letterbox_preserves_wide_image_and_foreground_crop_focuses_on_object(self):
        from PIL import Image
        raw = np.full((100, 300, 3), 255, dtype=np.uint8)
        raw[40:60, 140:160] = [220, 30, 30]
        letter = np.asarray(prepare_identity_image(Image.fromarray(raw), "letterbox"))
        foreground = np.asarray(prepare_identity_image(Image.fromarray(raw), "foreground_crop"))
        self.assertEqual(letter.shape, (224, 224, 3))
        self.assertTrue(np.all(letter[0] == 255))
        red = lambda a: (a[:, :, 0] > 150) & (a[:, :, 1] < 100)
        self.assertGreater(red(foreground).sum(), 10 * red(letter).sum())
        gray = Image.new("RGB", (100, 200), (128, 128, 128))
        self.assertEqual(prepare_identity_image(gray, "foreground_crop").size, (224, 224))

    def test_class_threshold_does_not_use_test_scores(self):
        samples, probabilities, values = [], [], []
        for i, fruit in enumerate(FRUITS):
            for j in range(20):
                samples.append({"phase": "calibration", "fruit": fruit})
                probabilities.append([0.99 if c == i else .005 for c in range(3)])
                values.append((i + 1) * .1 + j * .001)
        samples.append({"phase": "test", "fruit": "tomato"})
        probabilities.append([0, 0, 1]); values.append(100.)
        thresholds = calibrate_per_fruit({"embedding": np.array(values)}, np.array(probabilities), samples)
        self.assertLess(thresholds["embedding"]["tomato"], 1.)
        np.testing.assert_allclose(threshold_values(thresholds["embedding"], ["tomato", "banana"]), [.319, .119], atol=2e-6)

    def test_candidate_ranking_uses_selection_unknown_error_first(self):
        a = dict(preprocess_mode="center_crop", method="msp", unknown_false_acceptance_rate=.1,
                 known_balanced_error_rate=.01, tomato_false_rejection_rate=.01)
        b = dict(preprocess_mode="foreground_crop", method="embedding", unknown_false_acceptance_rate=0.,
                 known_balanced_error_rate=.1, tomato_false_rejection_rate=.1)
        self.assertLess(candidate_key(b), candidate_key(a))

    def test_calibration_counts_kind_errors_in_actual_fruit_95_percent_budget(self):
        samples = [{"fruit": f, "phase": "calibration"} for f in FRUITS for _ in range(20)]
        probabilities = np.repeat(np.eye(3), 20, axis=0)
        probabilities[-1] = [0, 1, 0]  # One real tomato misclassified as mandarin.
        values = np.tile(np.arange(20, dtype=float) / 100, 3)
        thresholds = calibrate_per_fruit({"embedding": values}, probabilities, samples)["embedding"]
        predicted = np.array(FRUITS)[probabilities.argmax(1)]
        actual = np.array([s["fruit"] for s in samples])
        passed = values <= threshold_values(thresholds, predicted)
        for fruit in FRUITS:
            self.assertGreaterEqual(np.mean((passed & (predicted == fruit))[actual == fruit]), .95)
        perturbed_passed = values + 5e-7 <= threshold_values(thresholds, predicted)
        for fruit in FRUITS:
            self.assertGreaterEqual(np.mean((perturbed_passed & (predicted == fruit))[actual == fruit]), .95)
        probabilities[-2] = [0, 1, 0]  # 90% type accuracy cannot meet the guarantee.
        with self.assertRaises(ValueError):
            calibrate_per_fruit({"embedding": values}, probabilities, samples)

    def test_runtime_uses_stored_preprocessing_and_candidate_threshold(self):
        bundle = {"version": 2, "classes": FRUITS, "encoder_state": {}, "preprocess_mode": "letterbox",
                  "classifier": Mock(classes_=np.array(FRUITS)),
                  "selected_method": "embedding", "thresholds": {"embedding": {"banana": .1, "mandarin": .2, "tomato": .3}}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "model.joblib"; path.write_bytes(b"test")
            with patch("src.fruit_classifier.joblib.load", return_value=bundle), \
                 patch("src.fruit_classifier.ImageEncoder") as encoder:
                detector = FruitClassifier(path)
                encoder.assert_called_once_with({}, "letterbox")
                detector.score = Mock(return_value=(np.array([[.01, .01, .98]]), {"embedding": np.array([.25])}))
                row = detector.classify("not_a_real_file.jpg")
                self.assertEqual(row["fruit"], "tomato")
                self.assertEqual(row["threshold"], .3)

    def test_selection_prioritizes_unknown_false_acceptance(self):
        comparison = {"msp": {"unknown_false_acceptance_rate": 0.2, "known_false_rejection_rate": 0.01},
                      "embedding": {"unknown_false_acceptance_rate": 0.1, "known_false_rejection_rate": 0.04}}
        self.assertEqual(select_method(comparison), "embedding")
        comparison["msp"]["unknown_false_acceptance_rate"] = 0.1
        self.assertEqual(select_method(comparison), "msp")

    def test_unknown_near_duplicates_do_not_cross_splits(self):
        fingerprints = [("a", 0), ("a", 0), ("b", (1 << 64) - 1),
                        ("c", int("10" * 32, 2)), ("d", int("01" * 32, 2))]
        with patch("src.train_fruit_classifier.image_fingerprint", side_effect=fingerprints):
            selection, test, groups = split_unknown(list(range(5)))
        self.assertEqual(groups[0], groups[1])
        self.assertFalse(set(groups[i] for i in selection) & set(groups[i] for i in test))

    def test_calibration_passes_at_least_95_percent_and_ties_pass(self):
        values = np.linspace(0, 1, 101)
        threshold = rejection_threshold(values)
        self.assertGreaterEqual(np.mean(values <= threshold), 0.95)
        self.assertEqual(rejection_threshold([0.5] * 20), 0.5)
        with self.assertRaises(ValueError):
            rejection_threshold([float("nan")])

    def test_high_confidence_unknown_can_be_rejected_by_distance(self):
        reference = np.array([[1., 0.]] * 6)
        scores = unfamiliarity_scores(np.array([[0.99, 0.005, 0.005]]),
                                      np.array([[0., 1.]]), reference)
        self.assertAlmostEqual(scores["msp"][0], 0.01)
        self.assertAlmostEqual(scores["embedding"][0], 1.)

    def test_error_denominators_and_threshold_equality(self):
        result = rejection_metrics([0.1, 0.5, 0.6, 0.9], [0.5, 0.8], 0.5)
        self.assertEqual(result["known_false_rejection_count"], 2)
        self.assertEqual(result["known_false_rejection_rate"], 0.5)
        self.assertEqual(result["unknown_false_acceptance_rate"], 0.5)

    def test_unknown_skips_ripeness_and_counts_separately_from_failure(self):
        detector = Mock()
        unknown = {"fruit": "unknown", "accepted": False, "model_sha256": "abc"}
        known = {"fruit": "tomato", "accepted": True, "model_sha256": "abc"}
        detector.classify.side_effect = [unknown, known, OSError("bad image")]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (root / name).touch()
            with patch("src.predict.FruitClassifier", return_value=detector), \
                 patch("src.predict.load_model", return_value={}), \
                 patch("src.predict.select_model", return_value=(Mock(), "test")) as select, \
                 patch("src.predict.predict_loaded", return_value={"ripeness": "ripe", "probabilities": None,
                       "feature_ms": 1., "prediction_ms": 1., "total_ms": 2.}) as process, \
                 contextlib.redirect_stdout(io.StringIO()):
                rows, summary = predict_folder(root, fruit_mode="image")
            self.assertEqual([r["status"] for r in rows], ["unknown", "ok", "failed"])
            self.assertIsNone(rows[0]["ripeness"])
            self.assertEqual(summary["unknown_count"], 1)
            self.assertEqual(summary["failure_count"], 1)
            select.assert_called_once_with({}, "tomato")
            process.assert_called_once()
            csv_path, manifest = save_prediction_artifacts(rows, root / "results.csv")
            self.assertIn("fruit_identity_json", csv_path.read_text(encoding="utf-8-sig"))
            self.assertIn('"abc"', manifest.read_text(encoding="utf-8"))

    def test_single_unknown_does_not_load_ripeness_model(self):
        with patch("src.predict.resolve_image_path", return_value=Path("photo.jpg")), \
             patch("src.predict.FruitClassifier") as cls, \
             patch("src.predict.load_model") as load, \
             contextlib.redirect_stdout(io.StringIO()):
            cls.return_value.classify.return_value = {"accepted": False}
            self.assertEqual(predict("photo.jpg", fruit_mode="image"), "unknown")
            load.assert_not_called()

    def test_missing_identity_model_does_not_fall_back_to_filename(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            (root / "banana.jpg").touch()
            with patch("src.predict.FruitClassifier", side_effect=FileNotFoundError("missing")), \
                 patch("src.predict.load_model") as load:
                with self.assertRaises(FileNotFoundError):
                    predict_folder(root, fruit_mode="image")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
