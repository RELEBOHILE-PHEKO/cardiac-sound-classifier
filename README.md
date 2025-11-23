# HeartBeat AI - Cardiac Sound Abnormality Detection

# ❤️ HeartBeat AI - Cardiac Sound Classifier

An AI-powered clinical decision support system for automated classification of cardiac sounds using deep learning. This system helps healthcare professionals quickly identify abnormal heart sounds from audio recordings.

##  Features

- **Real-time Audio Classification**: Upload cardiac sound recordings and get instant predictions
- **Batch Processing**: Analyze multiple audio files simultaneously
- **Interactive Dashboard**: User-friendly Streamlit interface with real-time monitoring
- **Prediction History**: Track and review past predictions with timestamps
- **RESTful API**: FastAPI backend for easy integration with other systems
- **Model Performance Metrics**: View accuracy, precision, recall, F1 score, and AUC-ROC
- **Support for Multiple Formats**: WAV, MP3, FLAC, and OGG audio files

# HeartBeat AI — Cardiac Sound Classifier

An end-to-end project that classifies cardiac sound recordings (heartbeats) into "normal" or "abnormal" using a Convolutional Neural Network (CNN). The repository provides a FastAPI backend, a Streamlit dashboard frontend, utilities for batch testing, and scripts for training and preprocessing audio.

**This README covers:** quick setup, running the API + dashboard, API contract, training, testing, Docker, troubleshooting, and developer notes.

**Repository layout**

```
cardiac-sound-classifier/
├── src/                     # Backend and ML code
│   ├── api.py               # FastAPI app and endpoints
│   ├── config.py            # Configuration (paths, settings)
│   ├── model.py             # Model wrapper utilities
│   ├── preprocessing.py     # Audio preprocessing helpers
│   └── train.py             # Training script
├── frontend/                # Streamlit dashboard
│   └── app.py
├── models/                  # Model artifacts (not checked in)
│   └── cardiac_cnn_model.h5
├── data/                    # Dataset and uploads
├── monitoring/              # Monitoring artifacts (metrics.json)
├── tools/                   # Helper scripts (tests, inspectors)
├── notebook/                # EDA notebooks
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

**Quick start (local development)**

- Requirements: Python 3.11+, Git, and a machine with enough CPU memory for TensorFlow (GPU optional).

1. Clone and enter the repo

```powershell
git clone <your-repo-url>
cd cardiac-sound-classifier
```

2. Create & activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

4. Place or verify the trained model

Ensure `models/cardiac_cnn_model.h5` exists. If not available, run training (see Training section) or copy the model into `models/`.

5. Run the API server (development)

```powershell
.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

6. Run the Streamlit dashboard (in a new terminal)

```powershell
.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501 --server.headless true
```

Open the dashboard at: http://localhost:8501

**API summary**

- `GET /health` — health and model-loaded status
- `GET /uptime` — server uptime
- `POST /predict` — single-file prediction (multipart form, key `file`)
- `POST /batch-predict` — multi-file prediction (multipart form, key `files` repeated)
- `GET /metrics` — monitoring metrics (from `monitoring/metrics.json` or fallback)
- `GET /visualizations/prediction-history` — last predictions
- `GET /visualizations/class-distribution` — counts from `data/train`

Example: single prediction (Python)

```python
import requests

resp = requests.post('http://127.0.0.1:8000/predict', files={'file': open('heartbeat.wav','rb')})
print(resp.json())
```

Example: batch prediction (Python)

```python
files = [('files', open('a.wav','rb')), ('files', open('b.wav','rb'))]
resp = requests.post('http://127.0.0.1:8000/batch-predict', files=files)
print(resp.json())
```

**Data & model**

- `data/train/` — training audio organized by class (e.g. `training-a/` etc.)
- `models/cardiac_cnn_model.h5` — trained Keras model used by the API
- Preprocessing converts audio to a 5-second clip sampled at 4000Hz and extracts 128-band mel-spectrograms.

**Training**

1. Prepare labeled audio under `data/train/normal/` and `data/train/abnormal/` (or adapt `train.py` dataset loader).
2. Edit training hyperparameters in `src/config.py` if needed.
3. Run training:

```powershell
python src/train.py
```

Output model will be saved to `models/cardiac_cnn_model.h5` by default.

**Testing & utilities**

- Unit tests: run `pytest` if present (see `tests/` folder).
- Batch test helper: `tools/run_batch_test.py` — posts sample audio to `/batch-predict` and falls back to `/predict` per file.
- Route inspector: `tools/inspect_routes.py` can print the FastAPI app routes for debugging.

**Docker (quick)**

Build and run with docker-compose (containerized API + dashboard if configured):

```powershell
docker-compose build
docker-compose up
```

Adjust Dockerfile and docker-compose.yml for production deployment (set environment variables, mount model, configure secrets).

**Common troubleshooting**

- Model not loaded / 503: ensure `models/cardiac_cnn_model.h5` exists and is compatible with installed TensorFlow. Check server logs for stack traces.
- Audio "format not recognized": API reads uploaded bytes with `librosa` using `io.BytesIO` — ensure the uploaded files are valid audio files and supported formats (WAV, MP3, FLAC, OGG).
- `--reload` causing stale imports: if you see duplicate startup handlers or NameError for `datetime`, stop reloader and restart uvicorn without `--reload` while editing `src/api.py` carefully.
- Large TensorFlow logs: set `TF_CPP_MIN_LOG_LEVEL=2` in your environment to reduce verbosity.

**Developer notes**

- Backend entrypoint: `src.api:app` (Uvicorn/ASGI)
- Predictor is lazy-loaded during app lifespan; errors during load are logged to `uvicorn.error` and the `/health` endpoint shows `model_loaded: false` when not loaded.
- The dashboard gracefully falls back to per-file `/predict` if `/batch-predict` is unavailable.

**Contribution & license**

Contributions welcome — open an issue or submit a PR. This project is provided for research/education; adapt licensing and data governance for clinical use.

---

If you'd like, I can:
- run the repository tests and the batch test now and paste the outputs,
- commit and push the updated README and notebook to your remote, or
- add a short `CONTRIBUTING.md` and `.gitignore` entries for model artifacts.

Which would you like next?

## Load testing / Locust results

I ran a short Locust load test against the API and saved the full time-series CSV under `outputs/locust/`.

Top-line results (run `outputs/locust/run1`):

- Total requests: ~4,200 (see `outputs/locust/run1_stats_history.csv` for the exact time-series)
- Failures: 0
- Peak requests/s: ~50 req/s
- Median response time (end of run): ~10–12 ms
- Average response time (end of run): ~12–32 ms
- Max observed response time: ~176 ms

Generated artifacts (in the repo):

- `outputs/locust/locust_requests.png` — Requests/sec vs time
- `outputs/locust/locust_response_times.png` — Median and average response times vs time
- `outputs/locust/summary.txt` — short textual summary of the run

How to reproduce locally

1. Start the API server (see Quick start above).
2. Run a headless Locust test (example):

```powershell
Set-Location 'C:\Users\PC\.vscode\cardiac-sound-classifier'
# Example: 100 users, spawn-rate 50 users/sec, 2 minutes
python -m locust -f tests\locustfile.py --headless -u 100 -r 50 -t 2m --host http://127.0.0.1:8000 --csv outputs\locust\run1
```

3. The CSV and PNG plots will be stored under `outputs/locust/`.

Notes

- The API handled ~50 requests/sec under this local test with no recorded failures. Response times were low for the majority of requests, with occasional spikes up to ~176 ms. For a proper production evaluation, repeat tests on cloud-deployed instances and compare performance across different numbers of containers/replicas.
