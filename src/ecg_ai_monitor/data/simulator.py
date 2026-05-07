from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Scenario = Literal["normal", "tachycardia", "bradycardia", "irregular", "mixed"]


@dataclass
class SimulatedECG:
    time: np.ndarray
    ecg: np.ndarray
    rhythm_label: str
    beat_times: np.ndarray
    fs: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": self.time, "ecg": self.ecg})


def _gaussian(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def _scenario_hr_at(t: float, scenario: Scenario) -> tuple[float, str]:
    if scenario == "normal":
        return 72.0, "normal"
    if scenario == "tachycardia":
        return 125.0, "tachycardia"
    if scenario == "bradycardia":
        return 45.0, "bradycardia"
    if scenario == "irregular":
        return 78.0, "irregular"
    # mixed: normal -> tachy -> irregular -> brady -> normal
    phase = int(t // 12) % 5
    if phase == 1:
        return 125.0, "tachycardia"
    if phase == 2:
        return 82.0, "irregular"
    if phase == 3:
        return 45.0, "bradycardia"
    return 72.0, "normal"


def _build_beat_times(duration_s: float, scenario: Scenario, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 0.6
    beat_times: list[float] = []
    while t < duration_s - 0.3:
        hr, label = _scenario_hr_at(t, scenario)
        rr = 60.0 / hr
        if label == "irregular" or scenario == "irregular":
            rr *= float(rng.uniform(0.65, 1.45))
            # occasional ectopic-like short interval followed by compensatory pause
            if rng.random() < 0.12:
                beat_times.append(t)
                t += rr * 0.55
                beat_times.append(t)
                t += rr * 1.45
                continue
        else:
            rr *= float(rng.normal(1.0, 0.025))
        beat_times.append(t)
        t += max(0.32, rr)
    return np.array(beat_times, dtype=float)


def simulate_ecg(
    duration_s: float = 60.0,
    fs: int = 250,
    scenario: Scenario = "mixed",
    noise_std: float = 0.025,
    seed: int = 42,
) -> SimulatedECG:
    """Generate a synthetic single-lead ECG-like waveform.

    The simulator is intentionally lightweight. It models each beat as a sum of
    P/Q/R/S/T Gaussian components plus baseline drift, high-frequency noise and
    small amplitude variability. The output is suitable for algorithm testing,
    not for clinical validation.
    """

    if duration_s <= 5:
        raise ValueError("duration_s must be greater than 5 seconds")
    if fs < 100:
        raise ValueError("fs must be at least 100 Hz for ECG simulation")

    rng = np.random.default_rng(seed)
    time = np.arange(0.0, duration_s, 1.0 / fs)
    ecg = np.zeros_like(time)
    beat_times = _build_beat_times(duration_s, scenario, seed)

    for bt in beat_times:
        _, label = _scenario_hr_at(bt, scenario)
        amp_scale = float(rng.normal(1.0, 0.04))
        r_amp = 1.15 * amp_scale
        qrs_width = 0.018 if label != "irregular" else float(rng.uniform(0.016, 0.026))
        ecg += _gaussian(time, bt - 0.18, 0.035, 0.09 * amp_scale)  # P
        ecg += _gaussian(time, bt - 0.035, 0.010, -0.14 * amp_scale)  # Q
        ecg += _gaussian(time, bt, qrs_width, r_amp)  # R
        ecg += _gaussian(time, bt + 0.035, 0.014, -0.22 * amp_scale)  # S
        ecg += _gaussian(time, bt + 0.25, 0.075, 0.28 * amp_scale)  # T

    baseline = 0.06 * np.sin(2 * np.pi * 0.25 * time) + 0.03 * np.sin(2 * np.pi * 0.05 * time)
    mains = 0.006 * np.sin(2 * np.pi * 50.0 * time)
    noise = rng.normal(0.0, noise_std, size=time.shape)
    ecg = ecg + baseline + mains + noise

    return SimulatedECG(time=time, ecg=ecg.astype(float), rhythm_label=scenario, beat_times=beat_times, fs=fs)


def save_simulated_csv(path: str, duration_s: float, fs: int, scenario: Scenario, seed: int = 42) -> str:
    sim = simulate_ecg(duration_s=duration_s, fs=fs, scenario=scenario, seed=seed)
    sim.to_frame().to_csv(path, index=False)
    return path
