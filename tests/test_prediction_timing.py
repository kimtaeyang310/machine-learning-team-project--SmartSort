import unittest
from unittest.mock import patch

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.predict import configure_prediction_jobs
from src.preprocessing import preprocess
from src.feature_extraction import extract_features


class PredictionTimingTests(unittest.TestCase):
    def test_jobs_changes_only_random_forest(self):
        forest = RandomForestClassifier(n_jobs=-1)
        model = make_pipeline(StandardScaler(), forest)
        configure_prediction_jobs(model)
        self.assertEqual(forest.n_jobs, 1)
        neighbors = KNeighborsClassifier(n_jobs=-1)
        configure_prediction_jobs(neighbors)
        self.assertEqual(neighbors.n_jobs, -1)

    def test_timing_preserves_processing_and_attempts(self):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        cv2.circle(image, (32, 32), 16, (0, 100, 220), -1)
        stages = {}
        with patch("src.preprocessing.read_image", return_value=image), \
             patch("src.preprocessing.load_manual_mask", return_value=None), \
             patch("src.preprocessing.cv2.kmeans", wraps=cv2.kmeans) as kmeans:
            baseline = preprocess("unused.jpg", size=64, save_failed=False)
            measured = preprocess("unused.jpg", size=64, timings=stages, save_failed=False)
            self.assertEqual(kmeans.call_args.args[4], 3)
            compact = preprocess("unused.jpg", size=64, include_preview=False, save_failed=False)
            preprocess("unused.jpg", size=64, kmeans_attempts=1, save_failed=False)
            self.assertEqual(kmeans.call_args.args[4], 1)
        for key in ("image", "mask", "hsv"):
            np.testing.assert_array_equal(baseline[key], measured[key])
            np.testing.assert_array_equal(baseline[key], compact[key])
        self.assertEqual(extract_features(baseline), extract_features(compact))
        self.assertIsNone(compact["original_image"])
        self.assertIsNone(compact["fruit_image"])
        self.assertEqual(set(stages), {"read_ms", "denoise_ms", "resize_ms",
                                      "manual_mask_ms", "kmeans_ms", "mask_other_ms", "finish_ms"})
        self.assertTrue(all(value >= 0 for value in stages.values()))
        self.assertGreater(stages["kmeans_ms"], 0)

    def test_manual_mask_skips_kmeans_and_resets_timings(self):
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        stages = {"kmeans_ms": 999, "old": 1}
        with patch("src.preprocessing.read_image", return_value=image), \
             patch("src.preprocessing.load_manual_mask", return_value=mask), \
             patch("src.preprocessing.create_fruit_mask") as create:
            preprocess("unused.jpg", size=64, timings=stages, save_failed=False)
        create.assert_not_called()
        self.assertEqual(stages["kmeans_ms"], 0)
        self.assertNotIn("old", stages)

    def test_compact_failure_still_saves_original_for_diagnostics(self):
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        with patch("src.preprocessing.read_image", return_value=image), \
             patch("src.preprocessing.load_manual_mask", return_value=None), \
             patch("src.preprocessing.create_fruit_mask", side_effect=ValueError("no mask")), \
             patch("src.preprocessing.save_failure_debug") as save:
            with self.assertRaisesRegex(ValueError, "no mask"):
                preprocess("unused.jpg", size=64, include_preview=False)
        np.testing.assert_array_equal(save.call_args.args[3]["original"], image)
