from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ECGConfig:
    """Global ECG processing configuration."""

    fs: int = 250
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    notch_hz: float = 50.0
    min_peak_distance_s: float = 0.28
    window_s: float = 10.0
    window_step_s: float = 5.0


DEFAULT_CONFIG = ECGConfig()
