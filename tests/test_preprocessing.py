from pathlib import Path

import numpy as np
import soundfile as sf

from src.preprocessing import AudioPreprocessor


def _write_wave(tmp_path: Path, sr: int = 4000, seconds: float = 1.0) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 220 * t)
    file_path = tmp_path / "sample.wav"
    sf.write(file_path, waveform, sr)
    return file_path


def test_waveform_loading_and_spectrogram(tmp_path):
    audio_path = _write_wave(tmp_path)
    preprocessor = AudioPreprocessor(
        target_sample_rate=4000,
        segment_seconds=1.0,
        n_mels=64,
        random_state=0,
    )

    waveform = preprocessor.load_waveform(audio_path)
    assert waveform.shape == (preprocessor.segment_samples,)

    mel = preprocessor.to_mel_spectrogram(waveform)
    assert mel.shape[0] == preprocessor.n_mels
    assert mel.ndim == 3  # (mel, time, channel)

    assert preprocessor.validate_audio(audio_path) is True

    augmented, metadata = preprocessor.augment(waveform)
    assert augmented.shape == waveform.shape
    assert set(metadata.keys()) == {"noise_factor", "time_shift"}
