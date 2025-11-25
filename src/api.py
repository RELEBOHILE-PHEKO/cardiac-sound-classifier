"""FastAPI backend for the cardiac sound classifier using TFLite."""

from __future__ import annotations

import io
import json
import logging
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
# TFLite Model wrapper
# -------------------------------------------------------------------
class HeartbeatPredictorTFLite:
    """TFLite model interface for heartbeat classification."""
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")

        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.classes = ["normal", "abnormal"]

    def preprocess(self, wav: np.ndarray, sr: int):
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

    def predict(self, wav: np.ndarray, sr: int):
        x = self.preprocess(wav, sr)

        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        prob = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        if len(prob) == 1:
            prob_abnormal = float(prob[0])
            prob_normal = 1.0 - prob_abnormal
        else:
            prob_normal = float(prob[0])
            prob_abnormal = float(prob[1])

        if prob_abnormal >= 0.5:
            return "abnormal", prob_abnormal
        else:
            return "normal", prob_normal

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
MODEL_PATH = settings.model_dir / "cardiac_cnn_model.tflite"
predictor: HeartbeatPredictorTFLite | None = None
prediction_history: list = []
startup_time: datetime | None = None

# -------------------------------------------------------------------
# Lifespan event
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, startup_time
    startup_time = datetime.now(timezone.utc)

    try:
        logger.info(f"Loading TFLite predictor from {MODEL_PATH}")
        predictor = HeartbeatPredictorTFLite(MODEL_PATH)
        logger.info("TFLite predictor loaded successfully.")
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Health & Status
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.get("/uptime")
def uptime():
    if startup_time:
        delta = datetime.now(timezone.utc) - startup_time
        return {"uptime_seconds": delta.total_seconds(), "started_at": startup_time.isoformat()}
    return {"uptime_seconds": 0, "started_at": None}

# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format.")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        content = await file.read()
        audio_data, sr = librosa.load(io.BytesIO(content), sr=4000, mono=True)
        label, conf = predictor.predict(audio_data, sr)

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

# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "HeartBeat AI API is running with TFLite!",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "docs": "/docs"
        }
    }

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
