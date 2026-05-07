from ecg_ai_monitor.data.simulator import simulate_ecg


def test_simulator_generates_expected_shape():
    sim = simulate_ecg(duration_s=10, fs=250, scenario="normal", seed=1)
    assert len(sim.time) == 2500
    assert len(sim.ecg) == 2500
    assert len(sim.beat_times) >= 8
    df = sim.to_frame()
    assert list(df.columns) == ["time", "ecg"]
