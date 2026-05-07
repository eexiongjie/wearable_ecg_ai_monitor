from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ecg_csv(path: str | Path, fs: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Load ECG CSV with columns `time` and `ecg`.

    If fs is not provided, it is inferred from the median time step.
    """

    df = pd.read_csv(path)
    if "ecg" not in df.columns:
        raise ValueError("CSV must contain an `ecg` column")
    ecg = df["ecg"].to_numpy(dtype=float)

    if "time" in df.columns:
        time = df["time"].to_numpy(dtype=float)
        if fs is None:
            dt = float(np.median(np.diff(time)))
            if dt <= 0:
                raise ValueError("Invalid time column: non-positive time step")
            fs = int(round(1.0 / dt))
    else:
        if fs is None:
            raise ValueError("fs is required when CSV has no `time` column")
        time = np.arange(len(ecg), dtype=float) / fs

    if len(ecg) < int(fs * 3):
        raise ValueError("ECG recording is too short; at least 3 seconds is recommended")
    return time, ecg, int(fs)


def save_json(path: str | Path, data: dict) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
