from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from ecg_ai_monitor.api.schemas import ECGAnalyzeRequest, SimulateRequest
from ecg_ai_monitor.data.simulator import simulate_ecg
from ecg_ai_monitor.screening.engine import analyze_ecg

app = FastAPI(
    title="Wearable ECG AI Monitor API",
    description="Single-lead ECG preprocessing, R-peak detection and rhythm screening API.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    web_path = Path(__file__).resolve().parents[3] / "web" / "index.html"
    if web_path.exists():
        return web_path.read_text(encoding="utf-8")
    return "<h1>Wearable ECG AI Monitor</h1><p>Open /docs for API documentation.</p>"


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict:
    scenario = req.scenario
    if scenario not in {"normal", "tachycardia", "bradycardia", "irregular", "mixed"}:
        raise HTTPException(status_code=400, detail="Unsupported scenario")
    sim = simulate_ecg(duration_s=req.duration_s, fs=req.fs, scenario=scenario, seed=req.seed)  # type: ignore[arg-type]
    return {
        "fs": sim.fs,
        "duration_s": req.duration_s,
        "scenario": scenario,
        "time": sim.time.tolist(),
        "samples": sim.ecg.tolist(),
    }


@app.post("/analyze")
def analyze(req: ECGAnalyzeRequest) -> dict:
    result = analyze_ecg(np.array(req.samples, dtype=float), fs=req.fs, model_path=req.model_path)
    return result.to_dict()


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...), fs: int | None = None) -> dict:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # pragma: no cover - defensive API branch
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    if "ecg" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain an `ecg` column")
    ecg = df["ecg"].to_numpy(dtype=float)
    if fs is None:
        if "time" not in df.columns:
            raise HTTPException(status_code=400, detail="Please provide fs when CSV has no `time` column")
        t = df["time"].to_numpy(dtype=float)
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            raise HTTPException(status_code=400, detail="Invalid time column")
        fs = int(round(1.0 / dt))
    return analyze_ecg(ecg, fs=fs).to_dict()
