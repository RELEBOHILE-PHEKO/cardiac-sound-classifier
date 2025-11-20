import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("tensorflow")

from src.preprocessing import AudioPreprocessor
from src.train import (
    build_tf_dataset,
    plot_confusion_matrix,
    plot_training_history,
    prepare_numpy_arrays,
)


def _write_wave(directory, filename="sample.wav", sr=4000, seconds=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 110 * t)
    path = directory / filename
    sf.write(path, waveform, sr)
    return path


def test_prepare_numpy_arrays_and_dataset(tmp_path):
    class_names = ["normal_heart", "murmur"]
    train_dir = tmp_path / "train"
    for class_name in class_names:
        _write_wave(train_dir / class_name)

    preprocessor = AudioPreprocessor(segment_seconds=1.0, n_mels=32)
    waveforms, spectrograms, labels = prepare_numpy_arrays(
        train_dir,
        preprocessor,
        class_names,
    )

    assert waveforms.shape[0] == len(class_names)
    assert spectrograms.shape[0] == len(class_names)
    assert labels.shape == (len(class_names), len(class_names))

    dataset = build_tf_dataset(waveforms, spectrograms, labels, batch_size=2)
    batch = next(iter(dataset.take(1)))
    (wave_batch, spec_batch), label_batch = batch
    assert wave_batch.shape[-1] == 1
    assert spec_batch.shape[-1] == 1
    assert label_batch.shape[-1] == len(class_names)


def test_plot_helpers(tmp_path):
    history = {
        "accuracy": [0.5, 0.7],
        "val_accuracy": [0.4, 0.65],
        "loss": [1.2, 0.8],
        "val_loss": [1.4, 0.9],
    }
    history_plot = tmp_path / "history.png"
    plot_training_history(history, history_plot)
    assert history_plot.exists()

    confusion_plot = tmp_path / "cm.png"
    plot_confusion_matrix(np.eye(2), ["a", "b"], confusion_plot)
    assert confusion_plot.exists()

