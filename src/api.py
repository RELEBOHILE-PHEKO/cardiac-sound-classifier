"""FastAPI backend for the cardiac sound classifier."""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")

# -------------------------------------------------------------------
# Model wrapper
# -------------------------------------------------------------------
class HeartbeatPredictor:
    """TensorFlow model interface for heartbeat classification."""
    
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        import tensorflow as tf
        
        # Configure TensorFlow for minimal memory usage
        tf.config.set_visible_devices([], 'GPU')  # Disable GPU
        
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ["normal", "abnormal"]
        logger.info("Model loaded successfully")

    def preprocess(self, wav: np.ndarray, sr: int):
        """Preprocess audio into mel-spectrogram."""
        target_len = int(sr * 5)
        if wav.shape[0] < target_len:
            wav = np.pad(wav, (0, target_len - wav.shape[0]), mode="constant")
        else:
            wav = wav[:target_len]

        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_mels=128, hop_length=256, fmin=20, fmax=2000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.expand_dims(mel_db, axis=-1)
        return np.expand_dims(mel_db, axis=0).astype("float32")

    def predict(self, wav: np.ndarray, sr: int) -> dict:
        """Run prediction on audio data."""
        x = self.preprocess(wav, sr)
        raw = self.model.predict(x, verbose=0)[0]

        if len(raw) == 1:
            prob_abnormal = float(raw[0])
            prob_normal = 1.0 - prob_abnormal
        else:
            prob_normal = float(raw[0])
            prob_abnormal = float(raw[1])

        if prob_abnormal >= 0.5:
            predicted_class = "abnormal"
            confidence = prob_abnormal
        else:
            predicted_class = "normal"
            confidence = prob_normal

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probability_normal": round(prob_normal, 4),
            "probability_abnormal": round(prob_abnormal, 4),
        }

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
MODEL_PATH = settings.model_dir / "cardiac_cnn_model.h5"
predictor: HeartbeatPredictor | None = None
prediction_history: list = []
MAX_HISTORY = 100  # Limit history size
startup_time: datetime | None = None

# -------------------------------------------------------------------
# Lifespan event
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, startup_time
    startup_time = datetime.now(timezone.utc)

    try:
        logger.info(f"Loading predictor from {MODEL_PATH}")
        predictor = HeartbeatPredictor(MODEL_PATH)
        logger.info("Predictor loaded successfully.")
    except Exception as e:
        predictor = None
        logger.exception(f"Failed to load predictor: {e}")

    yield

# -------------------------------------------------------------------
# API setup
# -------------------------------------------------------------------
app = FastAPI(
    title="HeartBeat AI API", 
    version="1.0", 
    description="Cardiac sound classification API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------
@app.get("/")
@app.head("/")
def root():
    return {
        "status": "HeartBeat AI API is running!",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "uptime": "/uptime",
            "docs": "/docs"
        }
    }

# -------------------------------------------------------------------
# Health & Status
# -------------------------------------------------------------------
@app.get("/health")
@app.head("/health")
def health():
    return {
        "status": "ok", 
        "model_loaded": predictor is not None,
        "model_available": MODEL_PATH.exists() if predictor is not None else False
    }

@app.get("/uptime")
def uptime():
    if startup_time:
        delta = datetime.now(timezone.utc) - startup_time
        return {
            "uptime_seconds": delta.total_seconds(), 
            "started_at": startup_time.isoformat()
        }
    return {"uptime_seconds": 0, "started_at": None}

# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Predict cardiac sound classification from audio file."""
    
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format. Supported: WAV, MP3, OGG, FLAC")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        content = await file.read()
        audio, sr = librosa.load(io.BytesIO(content), sr=4000, mono=True)
        result = predictor.predict(audio, sr)

        result["file_name"] = file.filename
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Store in history (with size limit)
        prediction_history.append(result.copy())
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
        
        return result

    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Predict cardiac sound classification for multiple audio files."""
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch")

    results = []
    for file in files:
        try:
            content = await file.read()
            audio, sr = librosa.load(io.BytesIO(content), sr=4000, mono=True)
            result = predictor.predict(audio, sr)

            result["file_name"] = file.filename
            result["status"] = "success"
            result["timestamp"] = datetime.now(timezone.utc).isoformat()

            prediction_history.append(result.copy())
            if len(prediction_history) > MAX_HISTORY:
                prediction_history.pop(0)
            
            results.append(result)
            
        except Exception as e:
            logger.exception(f"Error processing {file.filename}")
            results.append({
                "file_name": file.filename, 
                "status": "error", 
                "error": str(e)
            })

    return {"results": results, "total": len(files), "successful": sum(1 for r in results if r.get("status") == "success")}

# -------------------------------------------------------------------
# Metrics & History
# -------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    """Get API metrics."""
    return {
        "predictions_served": len(prediction_history),
        "model_loaded": predictor is not None,
        "uptime": uptime()["uptime_seconds"] if startup_time else 0
    }

@app.get("/prediction-history")
def get_prediction_history(limit: int = 50):
    """Get recent prediction history."""
    return {
        "history": prediction_history[-limit:],
        "total": len(prediction_history)
    }

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
