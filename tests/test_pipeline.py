from ecg_ai_monitor.data.simulator import simulate_ecg
from ecg_ai_monitor.dsp.filters import preprocess_ecg
from ecg_ai_monitor.dsp.rpeaks import detect_r_peaks
from ecg_ai_monitor.screening.engine import analyze_ecg


def test_r_peak_detection_normal_signal():
    sim = simulate_ecg(duration_s=20, fs=250, scenario="normal", seed=2)
    filtered = preprocess_ecg(sim.ecg, fs=250)
    peaks = detect_r_peaks(filtered, fs=250)
    assert 18 <= len(peaks) <= 30


def test_screening_detects_tachycardia():
    sim = simulate_ecg(duration_s=25, fs=250, scenario="tachycardia", seed=3)
    result = analyze_ecg(sim.ecg, fs=250)
    assert result.summary_label in {"TACHYCARDIA", "IRREGULAR_RHYTHM"}
    assert result.global_features["mean_hr_bpm"] is not None


def test_screening_detects_bradycardia():
    sim = simulate_ecg(duration_s=25, fs=250, scenario="bradycardia", seed=4)
    result = analyze_ecg(sim.ecg, fs=250)
    assert result.summary_label == "BRADYCARDIA"
