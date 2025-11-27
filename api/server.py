"""Lightweight inference API exposing POST /predict for audio files.

This wraps the repository `HeartbeatPredictor` so Locust can benchmark model inference.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import traceback
import os

# Disable oneDNN informational message when TensorFlow is loaded by the
# predictor (keeps container/demo output quiet).
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = FastAPI(title="Heartbeat Inference API")


class PredictorWrapper:
    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        self.predictor = None

    def load(self):
        from src.prediction import HeartbeatPredictor

        self.predictor = HeartbeatPredictor(self.model_path)
        self.predictor.load()

    def predict_file(self, file_path: Path) -> dict:
        if self.predictor is None:
            raise RuntimeError("Predictor not loaded")

        # Prefer detailed output when available
        if hasattr(self.predictor, "predict_with_details"):
            return self.predictor.predict_with_details(file_path)
        return self.predictor.predict(file_path)


MODEL_PATH = Path("models/cardiac_cnn_model.h5")
predictor_wrapper = PredictorWrapper(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    try:
        predictor_wrapper.load()
        print("Loaded predictor for API")
    except Exception:
        traceback.print_exc()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != 'audio':
        raise HTTPException(status_code=400, detail="Uploaded file must be audio")

    try:
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        result = predictor_wrapper.predict_file(tmp_path)

        try:
            tmp_path.unlink()
        except Exception:
            pass

        return JSONResponse(result)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}
