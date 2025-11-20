"""Audio preprocessing utilities for HeartBeat AI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioPreprocessor:
    """End-to-end audio preprocessing helper."""

    target_sample_rate: int = 4000
    segment_seconds: float = 5.0
    n_mels: int = 128
    mel_hop_length: int = 256
    mel_fmin: int = 20
    mel_fmax: int | None = 2000
    random_state: int | None = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)

    @property
    def segment_samples(self) -> int:
        return int(self.segment_seconds * self.target_sample_rate)

    def load_waveform(self, path: Path) -> np.ndarray:
        """Load, resample, and length-normalize a waveform."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, _ = librosa.load(
            file_path.as_posix(),
            sr=self.target_sample_rate,
            mono=True,
        )
        if waveform.size == 0:
            raise ValueError(f"Empty audio file: {path}")

        target_len = self.segment_samples
        if waveform.shape[0] < target_len:
            pad = target_len - waveform.shape[0]
            waveform = np.pad(waveform, (0, pad), mode="constant")
        else:
            waveform = waveform[:target_len]

        return waveform.astype(np.float32)

    def to_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Convert waveform to log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sample_rate,
            n_mels=self.n_mels,
            hop_length=self.mel_hop_length,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.expand_dims(mel_db, axis=-1)
        return mel_db.astype(np.float32)

    def validate_audio(self, path: Path) -> bool:
        """Check if audio is readable and meets min duration."""
        try:
            with sf.SoundFile(path) as sound_file:
                duration = len(sound_file) / sound_file.samplerate
                return duration > 0
        except (RuntimeError, FileNotFoundError):
            return False

    def augment(self, waveform: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply lightweight noise/time-shift augmentation."""
        augmented = np.copy(waveform)
        noise_factor = float(self._rng.uniform(0.0, 0.02))
        if noise_factor > 0:
            noise = self._rng.normal(0, 1, size=augmented.shape).astype(np.float32)
            augmented = augmented + noise_factor * noise

        max_shift = max(1, int(0.05 * augmented.shape[0]))
        time_shift = int(self._rng.integers(-max_shift, max_shift + 1))
        if time_shift != 0:
            augmented = np.roll(augmented, time_shift)

        augmented = np.clip(augmented, -1.0, 1.0).astype(np.float32)
        metadata = {"noise_factor": noise_factor, "time_shift": float(time_shift)}
        return augmented, metadata
