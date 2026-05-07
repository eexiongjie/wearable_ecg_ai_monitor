from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_ai_monitor.dsp.features import feature_dict_to_vector

LABELS = ["NORMAL", "TACHYCARDIA", "BRADYCARDIA", "IRREGULAR_RHYTHM"]


@dataclass
class ECGWindowModel:
    pipeline: Pipeline

    def predict_feature_dicts(self, features: list[dict]) -> list[str]:
        if not features:
            return []
        x = np.array([feature_dict_to_vector(f) for f in features], dtype=float)
        return [str(v) for v in self.pipeline.predict(x)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "ECGWindowModel":
        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)


def build_default_model(random_state: int = 42) -> ECGWindowModel:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=120, max_depth=8, random_state=random_state, class_weight="balanced")),
        ]
    )
    return ECGWindowModel(pipeline=pipeline)
