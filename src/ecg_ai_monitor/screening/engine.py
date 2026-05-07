from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from ecg_ai_monitor.dsp.features import extract_global_features, extract_window_features
from ecg_ai_monitor.dsp.filters import preprocess_ecg
from ecg_ai_monitor.dsp.rpeaks import detect_r_peaks, rr_intervals_from_peaks
from ecg_ai_monitor.ml.model import ECGWindowModel

RhythmClass = Literal["NORMAL", "TACHYCARDIA", "BRADYCARDIA", "IRREGULAR_RHYTHM", "LOW_SIGNAL_QUALITY"]


@dataclass
class Finding:
    start_s: float
    end_s: float
    label: RhythmClass
    severity: Literal["info", "warning", "critical"]
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScreeningResult:
    fs: int
    duration_s: float
    signal_quality: str
    global_features: dict
    r_peak_indices: list[int]
    findings: list[dict]
    summary_label: str
    recommendations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _signal_quality(ecg_raw: np.ndarray, ecg_filtered: np.ndarray) -> tuple[str, str | None]:
    raw_std = float(np.std(ecg_raw))
    filt_std = float(np.std(ecg_filtered))
    if raw_std < 1e-4 or filt_std < 1e-4:
        return "poor", "Signal amplitude is extremely low. Please check electrode contact."
    if np.max(np.abs(ecg_raw)) > 8.0:
        return "poor", "Signal contains unrealistically large spikes, suggesting saturation or motion artifact."
    if raw_std > 0 and filt_std / raw_std < 0.12:
        return "fair", "Most signal energy was removed by filtering; noise or baseline drift may be high."
    return "good", None


def _classify_window_by_rules(w: dict) -> Finding:
    start_s = float(w["start_s"])
    end_s = float(w["end_s"])
    mean_hr = w.get("mean_hr_bpm")
    irregularity = w.get("irregularity_index")
    rmssd = w.get("rmssd_ms")
    r_count = int(w.get("r_peak_count", 0))

    if r_count < 3 or mean_hr is None:
        return Finding(start_s, end_s, "LOW_SIGNAL_QUALITY", "warning", 0.78, "Too few valid R peaks detected in this window.")
    if mean_hr >= 110:
        return Finding(start_s, end_s, "TACHYCARDIA", "warning", min(0.98, 0.65 + (mean_hr - 110) / 70), f"Mean heart rate is high: {mean_hr:.1f} bpm.")
    if mean_hr <= 50:
        return Finding(start_s, end_s, "BRADYCARDIA", "warning", min(0.98, 0.65 + (50 - mean_hr) / 35), f"Mean heart rate is low: {mean_hr:.1f} bpm.")
    if irregularity is not None and irregularity >= 0.16:
        return Finding(start_s, end_s, "IRREGULAR_RHYTHM", "warning", min(0.95, 0.60 + irregularity), f"RR interval variability is elevated: irregularity={irregularity:.2f}.")
    if rmssd is not None and rmssd >= 180:
        return Finding(start_s, end_s, "IRREGULAR_RHYTHM", "warning", 0.72, f"RMSSD is elevated: {rmssd:.1f} ms.")
    return Finding(start_s, end_s, "NORMAL", "info", 0.70, "No obvious abnormal rhythm pattern was detected by rule-based screening.")


def _merge_summary(findings: list[Finding]) -> str:
    priority = ["LOW_SIGNAL_QUALITY", "TACHYCARDIA", "BRADYCARDIA", "IRREGULAR_RHYTHM", "NORMAL"]
    labels = {f.label for f in findings}
    for label in priority:
        if label in labels and label != "NORMAL":
            return label
    return "NORMAL"


def _recommendations(summary_label: str, signal_quality: str) -> list[str]:
    recs: list[str] = []
    if signal_quality != "good":
        recs.append("建议检查电极接触、导联连接和运动伪迹，必要时重新采集数据。")
    if summary_label == "TACHYCARDIA":
        recs.append("检测到疑似心动过速片段，建议结合症状、运动状态和医生意见进一步判断。")
    elif summary_label == "BRADYCARDIA":
        recs.append("检测到疑似心动过缓片段，建议结合睡眠、运动员体质、用药情况和医生意见进一步判断。")
    elif summary_label == "IRREGULAR_RHYTHM":
        recs.append("检测到疑似节律不齐片段，建议保留原始 ECG 片段供专业人员复核。")
    elif summary_label == "LOW_SIGNAL_QUALITY":
        recs.append("信号质量不足，当前结果可信度有限，建议重新采集。")
    else:
        recs.append("当前数据未发现明显异常筛查结果，但该结果不能替代临床诊断。")
    recs.append("本系统为工程原型，仅用于算法验证和教学展示。")
    return recs


def analyze_ecg(
    ecg: np.ndarray,
    fs: int,
    model_path: str | None = None,
    window_s: float = 10.0,
    step_s: float = 5.0,
) -> ScreeningResult:
    """Run the full ECG screening pipeline."""

    raw = np.asarray(ecg, dtype=float)
    filtered = preprocess_ecg(raw, fs=fs)
    r_peaks = detect_r_peaks(filtered, fs=fs)
    global_features = extract_global_features(filtered, r_peaks, fs).to_dict()
    windows = extract_window_features(filtered, r_peaks, fs, window_s=window_s, step_s=step_s)

    quality, quality_reason = _signal_quality(raw, filtered)
    findings = [_classify_window_by_rules(w) for w in windows]

    if model_path:
        try:
            model = ECGWindowModel.load(model_path)
            ml_labels = model.predict_feature_dicts(windows)
            for i, label in enumerate(ml_labels):
                # Keep poor-quality rule decisions, otherwise enrich with ML label.
                if findings[i].label != "LOW_SIGNAL_QUALITY" and label != "NORMAL":
                    findings[i].label = label  # type: ignore[assignment]
                    findings[i].confidence = max(findings[i].confidence, 0.80)
                    findings[i].reason += f" ML classifier also predicted {label}."
        except FileNotFoundError:
            pass

    if quality_reason is not None:
        findings.insert(0, Finding(0.0, len(raw) / fs, "LOW_SIGNAL_QUALITY", "warning", 0.80, quality_reason))

    summary = _merge_summary(findings)
    # Include global RR irregularity check to catch cases missed by windows.
    rr = rr_intervals_from_peaks(r_peaks, fs)
    if len(rr) > 3:
        irr = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0.0
        if irr >= 0.20 and summary == "NORMAL":
            summary = "IRREGULAR_RHYTHM"

    return ScreeningResult(
        fs=int(fs),
        duration_s=float(len(raw) / fs),
        signal_quality=quality,
        global_features=global_features,
        r_peak_indices=[int(i) for i in r_peaks],
        findings=[f.to_dict() for f in findings],
        summary_label=summary,
        recommendations=_recommendations(summary, quality),
    )
