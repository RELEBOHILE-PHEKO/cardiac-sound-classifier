"""FastAPI surface for HeartBeat AI."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.prediction import HeartbeatPredictor
from src.train import train_pipeline

app = FastAPI(
    title="HeartBeat AI",
    description="Cardiac & respiratory sound diagnostics",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = HeartbeatPredictor(model_path=settings.model_dir / "heartbeat_model.h5")
training_status = {"status": "idle", "progress": 0, "message": ""}
prediction_history: List[dict[str, Any]] = []
LOG_FILE = Path("monitoring/logs/predictions.jsonl")
UPLOAD_ROOT = Path("data/uploaded")
MAX_HISTORY = 100


@app.on_event("startup")
async def _load_predictor() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        predictor.load()
    except FileNotFoundError:
        # Model will be loaded after first training run.
        pass


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": predictor.model is not None,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    file_path = _save_temp_file(file)
    try:
        result = predictor.predict(file_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file_path.unlink(missing_ok=True)
    _record_prediction(result)
    return JSONResponse(result)


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)) -> dict[str, Any]:
    temp_paths = []
    for upload in files:
        temp_paths.append(_save_temp_file(upload))
    try:
        results = predictor.batch_predict(temp_paths)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
    for res in results:
        if "predicted_class" in res:
            _record_prediction(res)
    return {"results": results}


@app.post("/upload-training-data")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    target_class: str = Form(...),
) -> dict[str, Any]:
    class_dir = UPLOAD_ROOT / target_class
    class_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for upload in files:
        dest = class_dir / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved_files.append(str(dest))
    return {"uploaded": saved_files, "count": len(saved_files)}


@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks) -> dict[str, str]:
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Retraining already in progress.")

    training_status.update({"status": "running", "progress": 0, "message": "Starting retraining"})
    background_tasks.add_task(_run_retraining_task)
    return {"message": "Retraining triggered"}


@app.get("/training-status")
async def get_training_status() -> dict[str, Any]:
    return training_status


@app.get("/metrics")
async def get_model_metrics() -> dict[str, Any]:
    metrics_file = Path("monitoring/metrics.json")
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    raise HTTPException(status_code=404, detail="Metrics not available.")


@app.get("/visualizations/class-distribution")
async def get_class_distribution() -> dict[str, Any]:
    distribution = {}
    train_dir = Path("data/train")
    for class_dir in train_dir.iterdir() if train_dir.exists() else []:
        if class_dir.is_dir():
            distribution[class_dir.name] = len(list(class_dir.glob("*.wav")))
    return {"distribution": distribution}


@app.get("/visualizations/prediction-history")
async def get_prediction_history() -> dict[str, Any]:
    return {"history": prediction_history[-MAX_HISTORY:]}


def _save_temp_file(upload: UploadFile) -> Path:
    temp_dir = Path("data/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    dest = temp_dir / f"{uuid.uuid4().hex}_{upload.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _record_prediction(result: dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "result": result,
    }
    prediction_history.append(entry)
    if len(prediction_history) > MAX_HISTORY:
        del prediction_history[0]
    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def _run_retraining_task() -> None:
    try:
        training_status.update({"status": "running", "message": "Preparing data", "progress": 10})
        results = train_pipeline(
            data_dir=Path("data"),
            epochs=15,
            batch_size=32,
        )
        training_status.update(
            {
                "status": "completed",
                "message": f"Accuracy {results['evaluation']['accuracy']:.2f}",
                "progress": 100,
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        training_status.update({"status": "failed", "message": str(exc), "progress": 0})
