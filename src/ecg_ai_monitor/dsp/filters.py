from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass_filter(ecg: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 40.0, order: int = 3) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""

    if len(ecg) < fs:
        return np.asarray(ecg, dtype=float)
    nyq = fs / 2.0
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.99)
    if low >= high:
        raise ValueError("lowcut must be lower than highcut and below Nyquist")
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, ecg)


def notch_filter(ecg: np.ndarray, fs: int, notch_hz: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Apply a zero-phase notch filter for mains interference."""

    if notch_hz <= 0 or notch_hz >= fs / 2.0:
        return np.asarray(ecg, dtype=float)
    b, a = signal.iirnotch(w0=notch_hz / (fs / 2.0), Q=quality)
    return signal.filtfilt(b, a, ecg)


def preprocess_ecg(ecg: np.ndarray, fs: int, lowcut: float = 0.5, highcut: float = 40.0, notch_hz: float | None = 50.0) -> np.ndarray:
    x = np.asarray(ecg, dtype=float)
    x = x - np.nanmedian(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if notch_hz is not None:
        x = notch_filter(x, fs=fs, notch_hz=notch_hz)
    x = bandpass_filter(x, fs=fs, lowcut=lowcut, highcut=highcut)
    return x
