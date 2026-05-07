from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ecg_ai_monitor.dsp.rpeaks import rr_intervals_from_peaks


@dataclass
class ECGFeatures:
    duration_s: float
    r_peak_count: int
    mean_hr_bpm: float | None
    min_hr_bpm: float | None
    max_hr_bpm: float | None
    sdnn_ms: float | None
    rmssd_ms: float | None
    pnn50: float | None
    signal_energy: float
    signal_std: float
    irregularity_index: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_global_features(ecg_filtered: np.ndarray, r_peaks: np.ndarray, fs: int) -> ECGFeatures:
    x = np.asarray(ecg_filtered, dtype=float)
    duration_s = len(x) / fs if fs > 0 else 0.0
    rr = rr_intervals_from_peaks(r_peaks, fs)
    hr = 60.0 / rr if len(rr) else np.array([], dtype=float)

    sdnn_ms = float(np.std(rr, ddof=1) * 1000.0) if len(rr) > 1 else None
    rmssd_ms = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0) if len(rr) > 2 else None
    pnn50 = float(np.mean(np.abs(np.diff(rr)) > 0.05)) if len(rr) > 2 else None
    irregularity_index = float(np.std(rr) / np.mean(rr)) if len(rr) > 2 and np.mean(rr) > 0 else None

    return ECGFeatures(
        duration_s=float(duration_s),
        r_peak_count=int(len(r_peaks)),
        mean_hr_bpm=float(np.mean(hr)) if len(hr) else None,
        min_hr_bpm=float(np.min(hr)) if len(hr) else None,
        max_hr_bpm=float(np.max(hr)) if len(hr) else None,
        sdnn_ms=sdnn_ms,
        rmssd_ms=rmssd_ms,
        pnn50=pnn50,
        signal_energy=float(np.mean(x**2)) if len(x) else 0.0,
        signal_std=float(np.std(x)) if len(x) else 0.0,
        irregularity_index=irregularity_index,
    )


def extract_window_features(
    ecg_filtered: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> list[dict]:
    """Extract features from sliding windows."""

    x = np.asarray(ecg_filtered, dtype=float)
    n = len(x)
    win = int(window_s * fs)
    step = int(step_s * fs)
    if win <= 0 or step <= 0:
        raise ValueError("window_s and step_s must be positive")
    if n < win:
        starts = [0]
        win = n
    else:
        starts = list(range(0, n - win + 1, step))

    out: list[dict] = []
    for start in starts:
        end = min(n, start + win)
        peaks = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
        features = extract_global_features(x[start:end], peaks, fs).to_dict()
        features["start_s"] = float(start / fs)
        features["end_s"] = float(end / fs)
        out.append(features)
    return out


def feature_dict_to_vector(d: dict) -> list[float]:
    keys = [
        "duration_s",
        "r_peak_count",
        "mean_hr_bpm",
        "min_hr_bpm",
        "max_hr_bpm",
        "sdnn_ms",
        "rmssd_ms",
        "pnn50",
        "signal_energy",
        "signal_std",
        "irregularity_index",
    ]
    vector: list[float] = []
    for key in keys:
        value = d.get(key)
        vector.append(0.0 if value is None or not np.isfinite(value) else float(value))
    return vector
