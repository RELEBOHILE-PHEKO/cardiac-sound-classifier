""" Memory-optimized FastAPI backend for cardiac sound classifier."""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import gc

import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")

# -------------------------------------------------------------------
# Model wrapper with lazy loading
# -------------------------------------------------------------------
class HeartbeatPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None  # Don't load immediately
        self.classes = ["normal", "abnormal"]
        self.threshold = 0.5

    def _load_model(self):
        """Lazy load model only when needed."""
        if self.model is None:
            import tensorflow as tf
            # Configure TensorFlow for minimal memory usage
            tf.config.set_visible_devices([], 'GPU')  # Disable GPU
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Model loaded successfully")

    def _unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            logger.info("Model unloaded to free memory")

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
        result = np.expand_dims(mel_db, axis=0).astype("float32")
        
        # Clean up intermediate arrays
        del wav, mel, mel_db
        gc.collect()
        
        return result

    def predict(self, wav: np.ndarray, sr: int) -> dict:
        # Load model on demand
        self._load_model()
        
        try:
            x = self.preprocess(wav, sr)
            raw = self.model.predict(x, verbose=0)[0]

            if len(raw) == 1:
                prob_abnormal = float(raw[0])
                prob_normal = 1.0 - prob_abnormal
            else:
                prob_normal = float(raw[0])
                prob_abnormal = float(raw[1])

            if prob_abnormal >= self.threshold:
                predicted_class = "abnormal"
                confidence = prob_abnormal
            else:
                predicted_class = "normal"
                confidence = prob_normal

            result = {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "probability_normal": round(prob_normal, 4),
                "probability_abnormal": round(prob_abnormal, 4),
            }
            
            # Clean up
            del x, raw
            gc.collect()
            
            return result
        finally:
            # Unload model after prediction to free memory
            self._unload_model()

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
MODEL_PATH = settings.model_dir / "cardiac_cnn_model.h5"
predictor: HeartbeatPredictor | None = None
prediction_history: list = []
MAX_HISTORY = 50  # Limit history to save memory
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
        logger.info(f"Initializing predictor (lazy loading)")
        # Don't load model at startup - just create the predictor
        predictor = HeartbeatPredictor(MODEL_PATH)
        logger.info("Predictor initialized successfully.")
    except Exception as e:
        predictor = None
        logger.exception(f"Failed to initialize predictor: {e}")

    yield
    
    # Cleanup on shutdown
    if predictor:
        predictor._unload_model()
    gc.collect()

# -------------------------------------------------------------------
# App initialization
# -------------------------------------------------------------------
app = FastAPI(title="HeartBeat AI API", version="1.0", lifespan=lifespan)
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
@app.get("/")
def root():
    return {"status": "HeartBeat AI API is running!"}

@app.get("/health")
def health():
    return {"status": "ok", "predictor_ready": predictor is not None}

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
        raise HTTPException(status_code=400, detail="Invalid audio format")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    try:
        content = await file.read()
        audio, sr = librosa.load(io.BytesIO(content), sr=4000, mono=True)
        raw_result = predictor.predict(audio, sr)
        
        # Clean up audio data
        del content, audio
        gc.collect()
        
        logger.info(f"predict_audio: raw_result type={type(raw_result)}")

        out = raw_result.copy() if isinstance(raw_result, dict) else {"prediction": raw_result}
        out["file_name"] = file.filename
        out["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Maintain limited history
        prediction_history.append(out.copy())
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)
        
        return out
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []
    for file in files:
        try:
            content = await file.read()
            audio, sr = librosa.load(io.BytesIO(content), sr=4000, mono=True)
            raw_result = predictor.predict(audio, sr)
            
            # Clean up
            del content, audio
            gc.collect()
            
            logger.info(f"batch_predict: raw_result type={type(raw_result)}")

            out = raw_result.copy() if isinstance(raw_result, dict) else {"prediction": raw_result}
            out["file_name"] = file.filename
            out["status"] = "success"
            
            prediction_history.append(out.copy())
            if len(prediction_history) > MAX_HISTORY:
                prediction_history.pop(0)
            
            results.append(out)
        except Exception as e:
            results.append({"file_name": file.filename, "status": "error", "error": str(e)})

    return {"results": results, "total": len(files)}

# -------------------------------------------------------------------
# Metrics & Visualization
# -------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    f = Path("monitoring/metrics.json")
    if f.exists():
        try:
            return json.loads(f.read_text())
        except:
            pass
    return {"accuracy": None, "predictions_served": len(prediction_history)}

@app.get("/visualizations/prediction-history")
def get_prediction_history():
    return {"history": prediction_history[-50:]}  # Return max 50

@app.get("/visualizations/class-distribution")
def class_distribution():
    dist = {"normal": 0, "abnormal": 0}
    for d in [Path("data/train/training"), Path("data/train")]:
        if d.exists():
            for c in d.iterdir():
                if c.is_dir() and c.name in dist:
                    dist[c.name] += len(list(c.glob("*.wav")))
    return {"distribution": dist}

# -------------------------------------------------------------------
# Training (disabled for memory constraints)
# -------------------------------------------------------------------
@app.get("/training-status")
def training_status():
    return {"status": "disabled", "message": "Training disabled due to memory constraints"}

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    raise HTTPException(status_code=503, detail="Training disabled on this instance due to memory constraints")

@app.post("/upload-training-data")
async def upload_training_data(files: list[UploadFile] = File(...), target_class: str = "normal"):
    if target_class not in ["normal", "abnormal"]:
        raise HTTPException(status_code=400, detail="Invalid class")

    upload_dir = Path(f"data/uploads/{target_class}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for file in files:
        if file.filename.lower().endswith((".wav", ".mp3", ".flac")):
            content = await file.read()
            (upload_dir / file.filename).write_bytes(content)
            count += 1

    return {"message": f"Uploaded {count} files", "count": count}

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        workers=1  # Single worker to minimize memory
    )
