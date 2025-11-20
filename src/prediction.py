"""Prediction utilities for HeartBeat AI."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from src.config import settings
from src.preprocessing import AudioPreprocessor


class HeartbeatPredictor:
    """Handle inference, confidence reporting, and explainability."""

    def __init__(self, model_path: Path, metadata_path: Path | None = None):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else settings.metadata_file
        self.model: tf.keras.Model | None = None
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=settings.sample_rate,
            segment_seconds=settings.segment_seconds,
            n_mels=settings.n_mels,
        )
        self.classes = [
            "normal_heart",
            "murmur",
            "extrasystole",
            "normal_resp",
            "wheeze",
            "crackle",
        ]
        self.instructions = {
            "normal_heart": "No abnormality detected. Continue routine monitoring.",
            "murmur": "Consult cardiologist. Schedule echocardiogram for valve assessment.",
            "extrasystole": "Review patient history, consider ECG for arrhythmia confirmation.",
            "normal_resp": "Respiratory sounds are clear. Maintain baseline checkups.",
            "wheeze": "Administer bronchodilator assessment. Evaluate for asthma/COPD.",
            "crackle": "Investigate for pneumonia or pulmonary edema. Order chest imaging.",
        }
        self.impact = {
            "normal_heart": {"risk_score": 0.1},
            "murmur": {"risk_score": 0.7, "follow_up": "cardiology"},
            "extrasystole": {"risk_score": 0.6, "follow_up": "arrhythmia"},
            "normal_resp": {"risk_score": 0.1},
            "wheeze": {"risk_score": 0.5, "follow_up": "pulmonology"},
            "crackle": {"risk_score": 0.8, "follow_up": "emergency_if_severe"},
        }

    def load(self) -> None:
        """Load trained Keras model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text())
            if "classes" in metadata:
                self.classes = metadata["classes"]

    def predict(self, audio_path: Path) -> dict[str, Any]:
        """Predict waste type from audio file."""
        self._ensure_loaded()
        start = time.perf_counter()
        waveform, spectrogram = self._prepare_inputs(audio_path)
        inputs = [waveform[np.newaxis, ...], spectrogram[np.newaxis, ...]]
        probs = self.model.predict(inputs, verbose=0)[0]
        processing_time = time.perf_counter() - start

        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        sorted_indices = np.argsort(probs)[::-1]
        top3 = [
            {"class": self.classes[idx], "confidence": float(probs[idx])}
            for idx in sorted_indices[:3]
        ]
        probabilities = {self.classes[idx]: float(probs[idx]) for idx in range(len(self.classes))}

        return {
            "predicted_class": top_class,
            "confidence": float(probs[top_idx]),
            "all_probabilities": probabilities,
            "top_3_predictions": top3,
            "instructions": self.instructions.get(top_class, "No guidance available."),
            "environmental_impact": self.impact.get(top_class, {}),
            "processing_time": processing_time,
        }

    def batch_predict(self, audio_paths: List[Path]) -> list[dict[str, Any]]:
        """Predict multiple files."""
        results = []
        for audio_path in audio_paths:
            try:
                results.append(self.predict(audio_path))
            except Exception as exc:  # pylint: disable=broad-except
                results.append({"error": str(exc), "path": str(audio_path)})
        return results

    def get_grad_cam(self, audio_path: Path, layer_name: str = "spec_conv_3") -> dict[str, np.ndarray]:
        """Generate Grad-CAM heatmap focusing on spectrogram branch."""
        self._ensure_loaded()
        waveform, spectrogram = self._prepare_inputs(audio_path)
        inputs = [
            waveform[np.newaxis, ...],
            spectrogram[np.newaxis, ...],
        ]

        conv_layer = self.model.get_layer(layer_name)
        grad_model = tf.keras.Model(self.model.inputs, [conv_layer.output, self.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(inputs, training=False)
            top_index = tf.argmax(predictions[0])
            loss = predictions[:, top_index]
        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
        cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
        heatmap = cam.numpy()
        heatmap = np.uint8(255 * heatmap)
        overlay = self._overlay_heatmap(spectrogram[..., 0], heatmap)

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "spectrogram": spectrogram[..., 0],
        }

    def _prepare_inputs(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
        waveform = self.preprocessor.load_waveform(audio_path)
        spectrogram = self.preprocessor.to_mel_spectrogram(waveform)
        waveform = waveform[..., np.newaxis]
        return waveform, spectrogram

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    @staticmethod
    def _overlay_heatmap(spectrogram: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        normalized_spec = (spectrogram - spectrogram.min()) / (spectrogram.ptp() + 1e-8)
        normalized_heatmap = heatmap / 255.0
        overlay = 0.6 * normalized_spec + 0.4 * normalized_heatmap
        return overlay
