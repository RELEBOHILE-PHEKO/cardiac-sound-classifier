"""Training entrypoints for HeartBeat AI."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import settings
from src.model import HeartbeatEnsemble
from src.preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)
AUDIO_EXTENSIONS = ("*.wav", "*.flac", "*.mp3")
OUTPUT_DIR = Path("outputs")


def prepare_numpy_arrays(
    split_dir: Path,
    preprocessor: AudioPreprocessor,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load audio files from split_dir and return arrays."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    waveforms, spectrograms, labels = [], [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory missing: %s", class_dir)
            continue

        audio_files = list(_iter_audio_files(class_dir))
        if not audio_files:
            logger.warning("No audio files found for class %s", class_name)
            continue

        for audio_path in audio_files:
            waveform = preprocessor.load_waveform(audio_path)
            spectrogram = preprocessor.to_mel_spectrogram(waveform)
            waveforms.append(waveform[..., np.newaxis])
            spectrograms.append(spectrogram)
            labels.append(class_idx)

    if not waveforms:
        raise ValueError(f"No audio files discovered in {split_dir}")

    waveforms = np.stack(waveforms).astype(np.float32)
    spectrograms = np.stack(spectrograms).astype(np.float32)
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    labels = labels.astype(np.float32)

    return waveforms, spectrograms, labels


def build_tf_dataset(
    waveforms: np.ndarray,
    spectrograms: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data pipeline from numpy arrays."""
    ds = tf.data.Dataset.from_tensor_slices(((waveforms, spectrograms), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(waveforms), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_pipeline(
    data_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    fine_tune: bool = False,
) -> dict[str, Any]:
    """End-to-end training pipeline."""
    data_dir = Path(data_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = AudioPreprocessor(
        target_sample_rate=settings.sample_rate,
        segment_seconds=settings.segment_seconds,
        n_mels=settings.n_mels,
    )
    ensemble = HeartbeatEnsemble()
    class_names = list(ensemble.classes)

    train_wave, train_spec, train_labels = prepare_numpy_arrays(
        data_dir / "train",
        preprocessor,
        class_names,
    )
    # validation directory may be named 'validation' or legacy 'test'
    val_dir = data_dir / "validation"
    if not val_dir.exists():
        val_dir = data_dir / "test"

    val_wave, val_spec, val_labels = prepare_numpy_arrays(
        val_dir,
        preprocessor,
        class_names,
    )

    train_ds = build_tf_dataset(train_wave, train_spec, train_labels, batch_size, True)
    val_ds = build_tf_dataset(val_wave, val_spec, val_labels, batch_size, False)

    model = ensemble.build_ensemble()
    callbacks = _default_callbacks()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    if fine_tune:
        _unfreeze_top_layers(ensemble, fraction=0.3)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(5, epochs // 3),
            callbacks=callbacks,
        )

    eval_stats = evaluate_model(model, val_ds, class_names)

    history_plot = OUTPUT_DIR / "training_history.png"
    plot_training_history(history, history_plot)
    cm_plot = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        np.array(eval_stats["confusion_matrix"]),
        class_names,
        cm_plot,
    )

    artifacts = save_artifacts(
        model=model,
        history=history.history,
        evaluation=eval_stats,
        history_plot=history_plot,
        confusion_plot=cm_plot,
    )

    return {
        "history": history.history,
        "evaluation": eval_stats,
        "artifacts": artifacts,
    }


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate model on dataset and compute metrics."""
    y_true, y_pred = [], []
    for (wave, spec), labels in dataset:
        predictions = model.predict([wave, spec], verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    return {
        "accuracy": float(report["accuracy"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_training_history(history: Any, save_path: Path) -> None:
    """Plot accuracy and loss curves."""
    save_path = Path(save_path)
    history_dict = history.history if hasattr(history, "history") else history

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get("accuracy", []), label="train")
    plt.plot(history_dict.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get("loss", []), label="train")
    plt.plot(history_dict.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    save_path: Path,
) -> None:
    """Plot confusion matrix heatmap."""
    save_path = Path(save_path)
    plt.figure(figsize=(8, 6))
    # Use float annotation format to support non-integer matrices
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_artifacts(
    *,
    model: tf.keras.Model,
    history: Dict[str, list],
    evaluation: Dict[str, Any],
    history_plot: Path,
    confusion_plot: Path,
) -> Dict[str, Path]:
    """Persist trained model, metadata, and metrics."""
    model_path = settings.model_dir / "heartbeat_model.h5"
    metadata_path = settings.metadata_file
    metrics_path = Path("monitoring/metrics.json")

    model.save(model_path, include_optimizer=False)

    metadata = {
        "updated_at": datetime.utcnow().isoformat(),
        "classes": evaluation["class_names"],
        "accuracy": evaluation["accuracy"],
        "history_keys": list(history.keys()),
        "model_path": str(model_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    metrics_payload = {
        "accuracy": evaluation["accuracy"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "history_plot": str(history_plot),
        "confusion_plot": str(confusion_plot),
        "updated_at": metadata["updated_at"],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    history_path = OUTPUT_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": metrics_path,
        "history_plot": history_plot,
        "confusion_plot": confusion_plot,
        "history_json": history_path,
    }


def _iter_audio_files(directory: Path) -> Iterable[Path]:
    for pattern in AUDIO_EXTENSIONS:
        yield from directory.rglob(pattern)


def _default_callbacks() -> list[tf.keras.callbacks.Callback]:
    checkpoint_path = settings.model_dir / "heartbeat_best.h5"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]


def _unfreeze_top_layers(ensemble: HeartbeatEnsemble, fraction: float = 0.2) -> None:
    """Unfreeze top fraction of layers for fine-tuning."""
    for branch in (ensemble.waveform_model, ensemble.spectrogram_model):
        if branch is None:
            continue
        total_layers = len(branch.layers)
        unfreeze_from = int(total_layers * (1 - fraction))
        for idx, layer in enumerate(branch.layers):
            layer.trainable = idx >= unfreeze_from


def _cli_main(argv: list[str] | None = None) -> int:
    """Simple CLI so `python -m src.train` runs training.

    Usage examples:
      python -m src.train data --epochs 5 --batch-size 16
      python -m src.train --help
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run training pipeline for HeartBeat AI")
    parser.add_argument("data_dir", nargs="?", default=str(settings.data_dir), help="Path to data directory (contains train/ and validation/)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--fine-tune", action="store_true", help="Run fine-tuning after base training")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and exit without training")

    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.dry_run:
        # Quick validation of directories
        try:
            _ = list(data_dir.rglob("*.wav"))
            print(f"Dry-run: found audio files under {data_dir}")
            return 0
        except Exception as e:
            print(f"Dry-run failed: {e}")
            return 2

    try:
        print(f"Starting training: data={data_dir}, epochs={args.epochs}, batch_size={args.batch_size}")
        result = train_pipeline(data_dir, epochs=args.epochs, batch_size=args.batch_size, fine_tune=args.fine_tune)
        print("Training finished. Artifacts:")
        for k, v in result.get("artifacts", {}).items():
            print(f" - {k}: {v}")
        return 0
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Training failed: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(_cli_main())
