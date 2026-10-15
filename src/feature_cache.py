"""이미지별 원시 특징 캐시. 라벨, 분할, 학습된 표준화 값은 저장하지 않습니다."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "features" / "cache"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_features(features):
    return (isinstance(features, dict) and bool(features)
            and all(isinstance(k, str) and isinstance(v, (int, float))
                    and math.isfinite(v) for k, v in features.items()))


class FeatureCache:
    def __init__(self, config, directory=None):
        self.directory = Path(directory) if directory is not None else CACHE_DIR
        self.config = dict(config)
        self.hits = self.misses = 0
        self.signature = {
            "version": 1,
            "config": self.config,
            "numpy": np.__version__, "opencv": cv2.__version__,
            "code": {name: digest(Path(__file__).with_name(name)) for name in
                     ("preprocessing.py", "feature_extraction.py", "feature_cache.py")},
        }

    def get(self, path, preprocess, extract_features):
        path = Path(path).resolve()
        mask_hash = None
        try:
            relative = path.relative_to((PROJECT_ROOT / "data" / "raw").resolve())
        except ValueError:
            pass
        else:
            mask = PROJECT_ROOT / "outputs" / "manual_masks" / relative.parent / (relative.name + ".png")
            if mask.is_file():
                mask_hash = digest(mask)
        identity = {**self.signature, "path": str(path), "image": digest(path),
                    "manual_mask": mask_hash}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        target = self.directory / (key + ".json")
        try:
            entry = json.loads(target.read_text(encoding="utf-8"))
            if entry.get("identity") == identity and valid_features(entry.get("features")):
                self.hits += 1
                return entry["features"]
        except (OSError, ValueError, AttributeError):
            pass

        features = extract_features(preprocess(path, **self.config))
        if not isinstance(features, dict) or not features:
            raise ValueError("특징은 비어 있지 않은 딕셔너리여야 합니다.")
        if any(not isinstance(k, str) or not np.isscalar(v)
               or not isinstance(v, (int, float, np.number)) for k, v in features.items()):
            raise ValueError("특징 이름은 문자열, 특징값은 숫자 하나여야 합니다.")
        features = {k: float(v) for k, v in features.items()}
        if not valid_features(features):
            raise ValueError("특징에 NaN 또는 무한대가 있습니다.")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"identity": identity, "features": features}, stream, allow_nan=False)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.misses += 1
        return features

    def report(self):
        print(f"특징 캐시: 재사용 {self.hits}장 / 새로 계산 {self.misses}장")
