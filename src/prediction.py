"""Prediction utilities for HeartBeat AI."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.config import settings
from src.preprocessing import AudioPreprocessor


class HeartbeatPredictor:
    """
    Handle inference, confidence reporting, and explainability.
    
    Configured for PhysioNet Challenge 2016 binary classification:
    - Normal: Healthy heart sounds
    - Abnormal: Murmurs, clicks, and other cardiac anomalies
    """

    def __init__(
        self, 
        model_path: Path | str, 
        metadata_path: Path | str | None = None,
        threshold: float = 0.5
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained Keras model (.h5 or SavedModel)
            metadata_path: Optional path to model metadata JSON
            threshold: Classification threshold (default 0.5)
                      Lower threshold = higher recall (catch more abnormals)
                      Higher threshold = higher precision (fewer false alarms)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.threshold = threshold
        self.model: Optional[tf.keras.Model] = None
        self.is_ensemble = False  # Will be detected on model load
        
        # Initialize preprocessor with settings
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=settings.sample_rate,
            segment_seconds=settings.segment_seconds,
            n_mels=settings.n_mels,
        )
        
        # Binary classification classes
        self.classes = ["normal", "abnormal"]
        
        # Clinical guidance for each class
        self.instructions = {
            "normal": {
                "summary": "No abnormality detected",
                "action": "Continue routine monitoring",
                "follow_up": "Standard annual checkup",
                "urgency": "low"
            },
            "abnormal": {
                "summary": "Potential cardiac abnormality detected",
                "action": "Consult cardiologist for further evaluation",
                "follow_up": "Schedule echocardiogram and ECG",
                "urgency": "medium-high"
            }
        }
        
        # Risk scoring for clinical decision support
        self.risk_levels = {
            "normal": {
                "risk_score": 0.1,
                "risk_level": "LOW",
                "color": "green"
            },
            "abnormal": {
                "risk_score": 0.7,
                "risk_level": "ELEVATED",
                "color": "red"
            }
        }

    def load(self) -> None:
        """Load trained Keras model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        
        # Detect if ensemble model (has multiple inputs). Use `model.inputs` which
        # is consistently a list-like sequence of Input tensors for both Functional
        # and Sequential models.
        try:
            n_inputs = len(self.model.inputs)
        except Exception:
            n_inputs = 1

        self.is_ensemble = n_inputs > 1
        if self.is_ensemble:
            print("  → Detected: Ensemble model (waveform + spectrogram)")
        else:
            print("  → Detected: Single-input model (spectrogram only)")
        
        # Load metadata if available
        if self.metadata_path and self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text())
            if "classes" in metadata:
                self.classes = metadata["classes"]
            if "threshold" in metadata:
                self.threshold = metadata["threshold"]
            print(f"  → Loaded metadata: classes={self.classes}, threshold={self.threshold}")
        
        print("✅ Model loaded successfully!")

    def predict(self, audio_path: Path | str) -> Dict[str, Any]:
        """
        Predict cardiac condition from audio file.
        
        Args:
            audio_path: Path to .wav audio file
            
        Returns:
            Dictionary containing:
            - predicted_class: "normal" or "abnormal"
            - confidence: Probability of predicted class
            - probability_abnormal: Raw probability of abnormal
            - probability_normal: Raw probability of normal
            - clinical_guidance: Medical recommendations
            - risk_assessment: Risk score and level
            - processing_time: Inference time in seconds
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start = time.perf_counter()
        
        # Prepare inputs based on model type
        if self.is_ensemble:
            waveform, spectrogram = self._prepare_ensemble_inputs(audio_path)
            inputs = [waveform[np.newaxis, ...], spectrogram[np.newaxis, ...]]
        else:
            spectrogram = self._prepare_spectrogram_input(audio_path)
            inputs = spectrogram[np.newaxis, ...]
        
        # Normalize inputs for model.predict: make sure we pass the shape the
        # model expects. Some saved models accept a single spectrogram input while
        # others are ensembles (waveform + spectrogram). Defensively select the
        # spectrogram input when the model has a single input but our code has
        # prepared multiple arrays.
        try:
            model_input_count = len(self.model.inputs)
        except Exception:
            model_input_count = 1

        if isinstance(inputs, (list, tuple)):
            inputs_list = list(inputs)
        else:
            inputs_list = [inputs]

        if model_input_count == 1 and len(inputs_list) > 1:
            # Prefer the array that matches the spectrogram shape: (1, n_mels, t, 1)
            spect_idx = None
            for i, arr in enumerate(inputs_list):
                try:
                    arr_shape = getattr(arr, 'shape', None)
                    if arr_shape is not None and len(arr_shape) == 4 and arr_shape[1] == self.preprocessor.n_mels:
                        spect_idx = i
                        break
                except Exception:
                    continue
            if spect_idx is not None:
                inputs = inputs_list[spect_idx]
            else:
                # As a fallback pick the last prepared input
                inputs = inputs_list[-1]
        else:
            # If model expects multiple inputs keep a list, otherwise a single array
            inputs = inputs_list if model_input_count > 1 else inputs_list[0]

        prob_abnormal = float(self.model.predict(inputs, verbose=0)[0][0])
        prob_normal = 1.0 - prob_abnormal
        
        processing_time = time.perf_counter() - start
        
        # Apply threshold for classification
        predicted_class = "abnormal" if prob_abnormal >= self.threshold else "normal"
        confidence = prob_abnormal if predicted_class == "abnormal" else prob_normal
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probability_abnormal": round(prob_abnormal, 4),
            "probability_normal": round(prob_normal, 4),
            "threshold_used": self.threshold,
            "clinical_guidance": self.instructions[predicted_class],
            "risk_assessment": self.risk_levels[predicted_class],
            "processing_time": round(processing_time, 4),
            "file_name": audio_path.name,
        }

    def predict_with_details(self, audio_path: Path | str) -> Dict[str, Any]:
        """
        Extended prediction with additional diagnostic information.
        
        Includes waveform and spectrogram data for visualization.
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)
        
        # Get base prediction
        result = self.predict(audio_path)
        
        # Add visualization data
        waveform = self.preprocessor.load_waveform(audio_path)
        spectrogram = self.preprocessor.to_mel_spectrogram(waveform)
        
        result["waveform"] = waveform.tolist()
        result["spectrogram"] = spectrogram[:, :, 0].tolist()
        result["sample_rate"] = self.preprocessor.target_sample_rate
        result["duration_seconds"] = self.preprocessor.segment_seconds
        
        return result

    def batch_predict(self, audio_paths: List[Path | str]) -> List[Dict[str, Any]]:
        """
        Predict multiple audio files.
        
        Args:
            audio_paths: List of paths to .wav files
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(audio_paths)
        
        for idx, audio_path in enumerate(audio_paths, 1):
            try:
                print(f"Processing {idx}/{total}: {Path(audio_path).name}")
                result = self.predict(audio_path)
                result["status"] = "success"
                results.append(result)
            except Exception as exc:
                results.append({
                    "status": "error",
                    "error": str(exc),
                    "file_name": str(audio_path)
                })
        
        # Add batch statistics
        successful = [r for r in results if r.get("status") == "success"]
        if successful:
            abnormal_count = sum(1 for r in successful if r["predicted_class"] == "abnormal")
            normal_count = len(successful) - abnormal_count
            
            batch_summary = {
                "total_files": total,
                "successful": len(successful),
                "failed": total - len(successful),
                "normal_count": normal_count,
                "abnormal_count": abnormal_count,
                "abnormal_rate": round(abnormal_count / len(successful), 4) if successful else 0
            }
        else:
            batch_summary = {"total_files": total, "successful": 0, "failed": total}
        
        return {"predictions": results, "batch_summary": batch_summary}

    def get_grad_cam(
        self, 
        audio_path: Path | str, 
        layer_name: str = "spec_conv_2"
    ) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmap for model explainability.
        
        Shows which parts of the spectrogram the model focuses on.
        
        Args:
            audio_path: Path to audio file
            layer_name: Name of convolutional layer for Grad-CAM
            
        Returns:
            Dictionary with heatmap, overlay, and original spectrogram
        """
        self._ensure_loaded()
        audio_path = Path(audio_path)
        
        # Prepare inputs
        waveform = self.preprocessor.load_waveform(audio_path)
        spectrogram = self.preprocessor.to_mel_spectrogram(waveform)
        
        if self.is_ensemble:
            waveform_input = waveform[..., np.newaxis][np.newaxis, ...]
            spec_input = spectrogram[np.newaxis, ...]
            inputs = [waveform_input, spec_input]
        else:
            inputs = spectrogram[np.newaxis, ...]
        
        # Get the target layer
        try:
            conv_layer = self.model.get_layer(layer_name)
        except ValueError:
            # Find last conv layer if specified layer not found
            conv_layers = [l for l in self.model.layers if 'conv' in l.name.lower()]
            if not conv_layers:
                raise ValueError("No convolutional layers found in model")
            conv_layer = conv_layers[-1]
            print(f"Layer '{layer_name}' not found, using '{conv_layer.name}'")
        
        # Build Grad-CAM model
        grad_model = tf.keras.Model(
            self.model.inputs, 
            [conv_layer.output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(inputs, training=False)
            loss = predictions[:, 0]  # Binary classification output
        
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            raise ValueError("Could not compute gradients")
        
        grads = grads[0]
        conv_outputs = conv_outputs[0]
        
        # Compute weights and CAM
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
        
        # Normalize
        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
        heatmap = cam.numpy()
        
        # Resize heatmap to match spectrogram
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis], 
            spectrogram.shape[:2]
        ).numpy()[:, :, 0]
        
        # Create overlay
        overlay = self._overlay_heatmap(spectrogram[:, :, 0], heatmap_resized)
        
        return {
            "heatmap": heatmap_resized,
            "heatmap_raw": heatmap,
            "overlay": overlay,
            "spectrogram": spectrogram[:, :, 0],
            "layer_used": conv_layer.name
        }

    def _prepare_ensemble_inputs(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare both waveform and spectrogram inputs for ensemble model."""
        waveform = self.preprocessor.load_waveform(audio_path)
        spectrogram = self.preprocessor.to_mel_spectrogram(waveform)
        waveform = waveform[..., np.newaxis]  # Add channel dimension
        return waveform, spectrogram

    def _prepare_spectrogram_input(self, audio_path: Path) -> np.ndarray:
        """Prepare spectrogram input for single-branch model."""
        waveform = self.preprocessor.load_waveform(audio_path)
        spectrogram = self.preprocessor.to_mel_spectrogram(waveform)
        return spectrogram

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if self.model is None:
            self.load()

    @staticmethod
    def _overlay_heatmap(spectrogram: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Overlay Grad-CAM heatmap on spectrogram."""
        # Normalize spectrogram to 0-1
        spec_min, spec_max = spectrogram.min(), spectrogram.max()
        normalized_spec = (spectrogram - spec_min) / (spec_max - spec_min + 1e-8)
        
        # Blend
        overlay = 0.6 * normalized_spec + 0.4 * heatmap
        return np.clip(overlay, 0, 1)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        self._ensure_loaded()
        
        return {
            "model_path": str(self.model_path),
            "model_type": "ensemble" if self.is_ensemble else "spectrogram_only",
            "classes": self.classes,
            "threshold": self.threshold,
            "total_parameters": self.model.count_params(),
            "input_shapes": [str(inp.shape) for inp in self.model.inputs] if isinstance(self.model.input, list) else str(self.model.input.shape),
            "output_shape": str(self.model.output.shape),
        }
    

