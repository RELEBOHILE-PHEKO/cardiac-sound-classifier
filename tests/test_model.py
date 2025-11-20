import numpy as np

from src.model import HeartbeatEnsemble


def test_ensemble_forward_pass():
    ensemble = HeartbeatEnsemble()
    model = ensemble.build_ensemble()

    assert model.output_shape[-1] == ensemble.num_classes
    assert ensemble.waveform_model is not None
    assert ensemble.spectrogram_model is not None

    waveform = np.random.randn(2, *ensemble.waveform_input_shape).astype(np.float32)
    spectrogram = np.random.randn(2, *ensemble.spectrogram_input_shape).astype(np.float32)
    outputs = model.predict([waveform, spectrogram], verbose=0)

    assert outputs.shape == (2, ensemble.num_classes)
    np.testing.assert_allclose(outputs.sum(axis=1), np.ones(2), atol=1e-5)


def test_summary_contains_model_name():
    ensemble = HeartbeatEnsemble()
    ensemble.build_ensemble()
    summary_text = ensemble.summary()
    assert "heartbeat_ensemble" in summary_text
