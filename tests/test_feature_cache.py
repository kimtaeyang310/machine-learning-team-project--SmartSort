import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.feature_cache import FeatureCache
from src.train import build_feature_table


class FeatureCacheTests(unittest.TestCase):
    def test_reuse_and_invalidate_image_config_code_and_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "data/raw/photo.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"first")
            cache_dir = root / "cache"
            preprocess = Mock(return_value={})
            extract = Mock(return_value={"x": 2.0})
            with patch("src.feature_cache.PROJECT_ROOT", root):
                def run(config=None):
                    cache = FeatureCache(config or {}, cache_dir)
                    self.assertEqual(cache.get(image, preprocess, extract), {"x": 2.0})
                    return cache
                run()
                self.assertEqual(run().hits, 1)
                self.assertEqual(preprocess.call_count, 1)
                image.write_bytes(b"other")
                self.assertEqual(run().misses, 1)
                self.assertEqual(run({"size": 320}).misses, 1)
                mask = root / "outputs/manual_masks/photo.jpg.png"
                mask.parent.mkdir(parents=True)
                mask.write_bytes(b"mask")
                self.assertEqual(run().misses, 1)
                mask.write_bytes(b"changed mask")
                self.assertEqual(run().misses, 1)
                cache = FeatureCache({}, cache_dir)
                cache.signature["code"]["preprocessing.py"] = "changed"
                cache.get(image, preprocess, extract)
                self.assertEqual(cache.misses, 1)

    def test_corrupt_cache_rebuilt_and_invalid_features_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            cache = FeatureCache({}, root / "cache")
            pre = Mock(return_value={})
            extract = Mock(return_value={"x": 1.0})
            cache.get(image, pre, extract)
            entry = next(cache.directory.glob("*.json"))
            entry.write_text("broken", encoding="utf-8")
            cache.get(image, pre, extract)
            self.assertEqual(pre.call_count, 2)
            image.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                cache.get(image, pre, Mock(return_value={"x": float("nan")}))
            self.assertEqual(len(list(cache.directory.glob("*.json"))), 1)

    def test_training_reuses_features_with_fresh_labels_and_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a.jpg", root / "b.jpg"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            samples = [{"path": p, "image_path": p.name, "ripeness": "unripe"} for p in (a, b)]
            with patch("src.feature_cache.CACHE_DIR", root / "cache"), \
                 patch("src.train.preprocess", side_effect=lambda path, **kw: path.name) as pre, \
                 patch("src.feature_extraction.extract_features", side_effect=lambda name: {"x": 1 if name == "a.jpg" else 2}):
                first = build_feature_table(samples)
                samples[0]["ripeness"] = "ripe"
                second = build_feature_table(samples[::-1])
                self.assertEqual(pre.call_count, 2)
                self.assertEqual(first.x.tolist(), [1, 2])
                self.assertEqual(second.x.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
