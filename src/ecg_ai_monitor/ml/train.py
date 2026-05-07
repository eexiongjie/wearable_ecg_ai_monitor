from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ecg_ai_monitor.data.simulator import Scenario, simulate_ecg
from ecg_ai_monitor.dsp.features import extract_window_features, feature_dict_to_vector
from ecg_ai_monitor.dsp.filters import preprocess_ecg
from ecg_ai_monitor.dsp.rpeaks import detect_r_peaks
from ecg_ai_monitor.ml.model import build_default_model

SCENARIOS: list[Scenario] = ["normal", "tachycardia", "bradycardia", "irregular"]
SCENARIO_TO_LABEL = {
    "normal": "NORMAL",
    "tachycardia": "TACHYCARDIA",
    "bradycardia": "BRADYCARDIA",
    "irregular": "IRREGULAR_RHYTHM",
}


def make_training_dataset(n_samples_per_class: int = 40, fs: int = 250, duration_s: float = 30.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[str] = []
    rng = np.random.default_rng(seed)
    for scenario in SCENARIOS:
        for i in range(n_samples_per_class):
            sim = simulate_ecg(duration_s=duration_s, fs=fs, scenario=scenario, seed=int(rng.integers(0, 1_000_000)))
            filtered = preprocess_ecg(sim.ecg, fs=fs)
            peaks = detect_r_peaks(filtered, fs=fs)
            windows = extract_window_features(filtered, peaks, fs=fs, window_s=10.0, step_s=10.0)
            for feat in windows:
                x_rows.append(feature_dict_to_vector(feat))
                y_rows.append(SCENARIO_TO_LABEL[scenario])
    return np.array(x_rows, dtype=float), np.array(y_rows, dtype=str)


def train_model(out_path: str | Path, n_samples_per_class: int = 40, fs: int = 250, seed: int = 42) -> dict:
    x, y = make_training_dataset(n_samples_per_class=n_samples_per_class, fs=fs, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    model = build_default_model(random_state=seed)
    model.pipeline.fit(x_train, y_train)
    y_pred = model.pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    model.save(out_path)
    return {
        "model_path": str(out_path),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "labels": sorted(set(y.tolist())),
        "classification_report": report,
    }
