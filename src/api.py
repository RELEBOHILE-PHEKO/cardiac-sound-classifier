"""FastAPI backend for the cardiac sound classifier."""
from __future__ import annotations
import logging
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")

from src.config import settings
import json
import io

# -------------------------------------------------------------------
# Model wrapper
# -------------------------------------------------------------------

class HeartbeatPredictor:
    """Simple model interface for heartbeat classification."""
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # load model once
        self.model = tf.keras.models.load_model(model_path)

        # UPDATED → your model now has 2 classes
        self.classes = ["normal", "abnormal"]

    def preprocess(self, wav: np.ndarray, sr: int):
        # pad or cut to 5 seconds (use numpy pad/truncate to avoid librosa
        # API differences across versions)
        target_len = int(sr * 5)
        if wav.shape[0] < target_len:
            pad_width = target_len - wav.shape[0]
            wav = np.pad(wav, (0, pad_width), mode="constant")
        else:
            wav = wav[:target_len]

        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_mels=128,
            hop_length=256,
            fmin=20,
            fmax=2000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # CNN expects (batch, height, width, channels)
        mel_db = np.expand_dims(mel_db, axis=-1)
        return np.expand_dims(mel_db, axis=0).astype("float32")

    def predict(self, wav: np.ndarray, sr: int):
        x = self.preprocess(wav, sr)

        # model output shape → (1, 2)
        prob = self.model.predict(x)[0]

        idx = int(np.argmax(prob))
        return self.classes[idx], float(prob[idx])


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
MODEL_PATH = settings.model_dir / "cardiac_cnn_model.h5"
predictor: HeartbeatPredictor | None = None
prediction_history: list = []
startup_time: datetime | None = None

# -------------------------------------------------------------------
# Lifespan event
# -------------------------------------------------------------------
from contextlib import asynccontextmanager

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

app = FastAPI(title="Cardiac Sound Classifier API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev mode
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Model init (uses settings.model_dir)
# -------------------------------------------------------------------

# Predictor will be initialized during application startup so errors are
# visible in the uvicorn logs and don't occur at import time.
MODEL_PATH = settings.model_dir / "cardiac_cnn_model.h5"
predictor: HeartbeatPredictor | None = None
prediction_history: list = []
startup_time: datetime | None = None


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": getattr(predictor, 'model', None) is not None}


@app.get("/uptime")
def uptime():
    if startup_time:
        delta = datetime.now(timezone.utc) - startup_time
        return {"uptime_seconds": delta.total_seconds(), "started_at": startup_time.isoformat()}
    return {"uptime_seconds": 0, "started_at": None}

@app.post("/retrain")
# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format.")

    if predictor is None or getattr(predictor, 'model', None) is None:
        raise HTTPException(status_code=503, detail="Predictor not available or model not loaded.")

    try:
        data, sr = librosa.load(file.file, sr=4000, mono=True)
        label, conf = predictor.predict(data, sr)

        result = {
            "predicted_class": label,
            "confidence": float(conf),
            "file_name": file.filename,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        prediction_history.append(result.copy())
        return result

    except Exception as e:
        logger.exception("Error during /predict processing")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Predict multiple audio files at once."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        try:
            content = await file.read()
            audio_data, sr = librosa.load(io.BytesIO(content), sr=4000, mono=True)
            label, conf = predictor.predict(audio_data, sr)

            result = {
                "predicted_class": label,
                "confidence": float(conf),
                "file_name": file.filename,
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            prediction_history.append(result.copy())
            results.append(result)
        except Exception as e:
            logger.exception(f"Error processing {file.filename}")
            results.append({"file_name": file.filename, "status": "error", "error": str(e)})

    return {"results": results, "total": len(files)}

async def trigger_retrain():
    """Trigger model retraining."""
    return {"message": "Retraining triggered. Check /training-status for progress.", "status": "started"}

@app.post("/upload-training-data")
async def upload_training_data(files: list[UploadFile] = File(...), target_class: str = "normal"):
    """Upload new training data."""
    if target_class not in ["normal", "abnormal"]:
        raise HTTPException(status_code=400, detail="Invalid target_class")
    
    upload_dir = Path(f"data/uploads/{target_class}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for file in files:
        if file.filename.lower().endswith((".wav", ".mp3", ".flac")):
            content = await file.read()
            (upload_dir / file.filename).write_bytes(content)
            saved_count += 1
    
    return {"message": f"Uploaded {saved_count} files to {target_class}", "count": saved_count}

@app.get("/metrics")
def metrics():
    metrics_file = Path("monitoring/metrics.json")
    if metrics_file.exists():
        try:
            return json.loads(metrics_file.read_text())
        except Exception:
            pass
    return {"accuracy": None, "latency_ms": None, "predictions_served": 0}


@app.get("/visualizations/prediction-history")
def get_prediction_history():
    return {"history": prediction_history}


@app.get("/visualizations/class-distribution")
def class_distribution():
    distribution: dict[str, int] = {}
    train_dir = Path("data/train")
    if train_dir.exists():
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                distribution[class_dir.name] = len(list(class_dir.glob("*.wav")))
    return {"distribution": distribution}


@app.get("/training-status")
def training_status():
    return {"status": "idle", "progress": 0, "message": ""}

# -------------------------------------------------------------------
# Entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)






