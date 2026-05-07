from pathlib import Path

from ecg_ai_monitor.ml.train import train_model


def test_train_model_creates_file(tmp_path: Path):
    out = tmp_path / "model.joblib"
    metrics = train_model(out, n_samples_per_class=2, fs=250, seed=5)
    assert out.exists()
    assert metrics["n_train"] > 0
    assert "classification_report" in metrics
