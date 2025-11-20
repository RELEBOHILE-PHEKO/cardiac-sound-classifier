from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from src.prediction import HeartbeatPredictor


def test_load_missing_model(tmp_path):
    predictor = HeartbeatPredictor(model_path=tmp_path / "missing.h5")
    with pytest.raises(FileNotFoundError):
        predictor.load()


def test_batch_predict_handles_errors(tmp_path):
    predictor = HeartbeatPredictor(model_path=tmp_path / "missing.h5")
    results = predictor.batch_predict([tmp_path / "fake.wav"])
    assert "error" in results[0]
