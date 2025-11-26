# HeartBeat AI — Cardiac Sound Classifier

Concise: a Streamlit-first web app that loads a Keras/TensorFlow heart sound classifier, provides single- and batch-prediction, allows uploading labeled audio for retraining, and exposes a simple monitoring dashboard.

This README documents how to get the project running locally, how to retrain the model from uploaded examples, where model and notebook artifacts live, and how to reproduce a basic Locust load test.

**Repo layout (important files)**

- `frontend/app.py` — Streamlit application (UI: prediction, visualizations, monitoring, retrain)
- `src/` — core code: `preprocessing.py`, `model.py`, `prediction.py`, `train.py`
- `models/` — saved model artifacts (e.g., `cardiac_cnn_model.h5`)

- `data/` — dataset folders (`train/`, `validation/`, `uploads/`)
- `notebook/heartbeat_ai_eda.ipynb` — exploratory notebook and evaluation (open and inspect)
- `tools/retrain_from_uploads.py` — helper to prepare uploaded files for training and launch training

- `tools/run_batch_local.py` — local batch prediction runner
- `tests/locustfile.py` — Locust scenario used for smoke testing

## System Architecture

This section describes the high-level system architecture for HeartBeat AI, how components interact, and recommended deployment/topology for production or grading demonstrations.

1) Core components
- **Streamlit Frontend (`frontend/app.py`)**: single-page UI that provides prediction, batch upload, monitoring, and retraining controls. Keeps a lightweight in-memory session state for recent predictions.
- **Prediction & Preprocessing (`src/`)**: contains `preprocessing.py` (audio loader & mel-spectrogram conversion), `model.py` (model builders/ensemble wrappers), and `prediction.py` (loading model, preparing inputs, inference helper). The same code is used by UI and tools.
- **Training Pipeline (`src/train.py`)**: end-to-end training entrypoint that builds datasets, trains the model with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint), evaluates metrics, and writes artifacts to `models/`, `outputs/`, and `monitoring/`.
- **Model Artifacts (`models/`)**: Keras/TensorFlow `.h5` model files used for inference and as pre-trained starting points for retraining.
- **Tools (`tools/`)**: helper scripts such as `run_batch_local.py` (batch inference), `retrain_from_uploads.py` (prepare uploaded files and launch training), and small utilities used in demos and grading.

2) Data stores and file layout
- **Uploaded examples**: `data/uploads/<class>` — user-uploaded audio intended for retraining.
- **Training data**: `data/train/<class>` and `data/validation/<class>` — used by `src.train`.
- **Model & artifacts**: `models/heartbeat_model.h5`, `outputs/` (training history, confusion matrix), `monitoring/metrics.json` for saved evaluation metrics.

3) Request/Control flows
- **Single prediction (UI)**: user uploads file → Streamlit writes a temporary file → `src.prediction.HeartbeatPredictor` loads file and runs preprocessing → model.predict() → UI displays results and saves to session history.
- **Batch prediction**: multiple files uploaded → UI iterates and calls predictor → aggregates CSV results → provides download.
- **Retraining trigger**: UI saves uploaded files to `data/uploads` → user clicks retrain → Streamlit spawns a background process that invokes `python -m src.train` (or you can run `tools/retrain_from_uploads.py` to prepare a canonical train/validation split and launch training).
- **Monitoring**: `src.train` writes evaluation artifacts (`monitoring/metrics.json`, `outputs/*`) that the Streamlit monitoring page reads to display metrics and plots.

4) Simple ASCII diagram

```
	[User Browser]
			 |
			 |  (HTTP)  Streamlit UI (frontend/app.py)
			 v
	[Streamlit App container]
			 |-- uses --> [src.prediction] -> loads model from `models/` -> returns prediction
			 |-- writes --> data/uploads/  (for retraining)
			 |-- triggers -> python -m src.train  (background)
			 v
	[Training process] -> reads data/train & data/validation -> writes models/ + outputs/ + monitoring/
```

5) Recommended production topology (for grading / scaled demos)
- For light demos: Streamlit Community Cloud (single process) hosting the repository is acceptable.
- For scaled inference / load testing: split responsibilities into services:
	- A lightweight API service (FastAPI/Flask) that wraps `HeartbeatPredictor` and exposes a `POST /predict` endpoint accepting audio files. Containerize the API and scale horizontally behind a load balancer.
	- A separate worker/runner for retraining (runs `src.train` in a batch job or scheduled job). Avoid retraining inside the UI process in production.
	- Object storage (S3-compatible) for large datasets and model artifacts instead of committing large files to the Git repository.
	- Metrics collection (Prometheus/Grafana) or centralized monitoring for logging training runs, inference latency, and throughput.

6) Scaling & load-testing guidance
- Use `tests/locustfile.py` to test a health endpoint or an API wrapper. For true inference benchmarking, implement a small API that accepts `POST /predict` (multipart) and returns JSON; Streamlit is not designed for high-concurrency POST inference.
- To simulate scale, run multiple API containers and distribute load with a simple reverse proxy (NGINX) or cloud load balancer.

7) Security & privacy notes
- Do not commit private or patient audio files to Git. Use `.gitignore` to exclude `data/` and uploaded content for public repos.
- Sanitize and validate uploaded files before training; current helper scripts copy files conservatively and do not execute arbitrary content.

8) Where to extend
- Add a small `api/` service (FastAPI) for inference and integrate model loading with a shared model registry.
- Add CI workflows to validate the model loading and basic inference on push (smoke tests) and to run the notebook tests.


## Quick start (local development)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```powershell
& .venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open http://127.0.0.1:8501 in your browser.

3. If you prefer to run only a quick local prediction from the command line, use the batch runner:

## Prediction (what to expect)

- The app accepts single audio files (`wav`, `mp3`, `flac`, `ogg`) via the Prediction page and shows: predicted class, probabilities, confidence, and technical JSON output.
- Batch uploads are supported in the Prediction page (downloadable CSV of results).
- The `HeartbeatPredictor` in `src/prediction.py` is used by both the UI and the batch runner.

## Retraining workflow

Two options to retrain the model with newly uploaded data:

1) From the Streamlit UI (Monitoring page)
	- Upload labeled audio files under Monitoring → Upload training audio files. Files will be saved under `data/uploads/<class>`.
	- Click "🔁 Retrain Model" to spawn a background process that runs the training pipeline (`src.train`). This is convenient for demos but not intended for long-running production training.

2) From the command line using the helper `tools/retrain_from_uploads.py` (recommended for reproducible runs):

```powershell
& .venv\Scripts\python.exe tools/retrain_from_uploads.py --uploads-dir data/uploads --target-data data --validation-split 0.2
```

- The helper copies files into `data/train/<class>` and `data/validation/<class>` (it does not delete uploads) and then launches `python -m src.train` by default.
- To preview actions without copying or training, run with `--dry-run`.

To train from scratch (full training pipeline):

```powershell
& .venv\Scripts\python.exe -m src.train data --epochs 30 --batch-size 32
```

Training artifacts (saved by `src.train`):
- Model file: `models/heartbeat_model.h5` (or `models/cardiac_cnn_model.h5` depending on code)
- Metrics/metadata: `monitoring/metrics.json` and `outputs/` images (training history, confusion matrix)

## Notebook and evaluation

- The notebook `notebook/heartbeat_ai_eda.ipynb` contains exploratory analysis and should include preprocessing, model training/testing examples, and evaluation metrics (accuracy, loss, F1, precision, recall). If your grader requires explicit metrics, ensure the notebook includes evaluation cells that print these metrics and save image artifacts under `outputs/`.

If you want, I can add/annotate notebook cells to compute and save the required metrics and figures.

## Locust load testing (reproduce smoke test)

We used Locust to run a headless smoke test targeting the deployed app health endpoint. Reproduce locally:

```powershell
& .venv\Scripts\python.exe -m locust -f tests/locustfile.py --headless -u 10 -r 2 --run-time 1m --host https://heartbeat-ai-classifier.streamlit.app
```

- Results from the last run are saved in `locust_results/locust_summary.json` and include average latency and throughput. To load-test inference endpoints (POST with audio), you'll need a lightweight API endpoint; Streamlit apps are not ideal for high-throughput POST benchmarking.

## Video demo (required for submission)

- Please record a short camera-on demo (5–8 minutes) showing:
	- Single-file prediction
	- Uploading training files (Monitoring → Upload training audio files)
	- Triggering retraining ("🔁 Retrain Model")
	- Showing monitoring outputs or `outputs/` images
- After uploading to YouTube (unlisted is fine), paste the URL here:

Video Demo: <ADD_YOUTUBE_LINK_HERE>

Need a script? See `tools/demo_script.md` and `tools/record_demo_instructions.md` for step-by-step guidance.

## Troubleshooting & tips

- If Streamlit shows "No module named 'src'" on deploy, ensure the repo root is on `sys.path` (the app currently inserts the repo root at startup).
- If model loading fails, check `models/` contains the expected `.h5` file and that your Python environment includes the packages listed in `requirements.txt` (notably `tensorflow`, `librosa`, `streamlit`).
- For large training datasets, keep training data out of Git history. Use `.gitignore` to ignore `data/` or large folders. If you accidentally committed large files, use `git rm --cached <file>` and consider `git filter-repo` or BFG for history rewrite (destructive — backup first).

## What to submit

- GitHub repository URL with the project (this repo).
- A short camera-on YouTube demo link in the README.
- Notebook with preprocessing, model training and evaluation cells (`notebook/heartbeat_ai_eda.ipynb`).
- Model file (in `models/`) and retraining script (`src/train.py` and `tools/retrain_from_uploads.py`).
- Locust results and reproduction instructions (we added `locust_results/locust_summary.json`).

## Help / Next steps

Tell me which of these you want me to do next and I will implement it:

- Patch the notebook to explicitly compute and save Accuracy, Loss, F1, Precision and Recall and export `outputs/` images.
- Add a `Makefile` / PowerShell helper to run common tasks (start app, retrain, run locust).
- Add a small example API wrapper to accept `POST /predict` with audio for better load-testing.

If you'd like me to add or change anything in this README (for example update the demo link), tell me and I'll patch it.

# HeartBeat AI — Streamlit-first Deployment

This repository contains a cardiac sound classifier. The project runs as a single Streamlit application that bundles the UI, model loading, prediction, monitoring, and retraining controls.

If you want to deploy the app on GitHub, the recommended workflow is:

- Push the repository to GitHub (you must have a remote named `origin`).
- Deploy to Streamlit Community Cloud by connecting your GitHub repo (fastest option for Streamlit apps).
- Optionally add a GitHub Actions workflow to build a Docker image and publish it to GitHub Container Registry (GHCR) or another container registry for alternative deployment.

## Quick start (local)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```powershell
& .venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

Open http://127.0.0.1:8501 in your browser.

## Deploy to Streamlit Community Cloud (recommended)

1. Commit and push your repo to GitHub:

```powershell
git add -A
git commit -m "Streamlit-first: deploy-ready updates"
git push origin main
```

2. Go to https://streamlit.io/cloud and connect your GitHub repository. Configure the app to run `frontend/app.py` and set the Python version to match your `.venv` (recommended: 3.11).

Streamlit Cloud will build and deploy the app automatically on push.

## Deploy via GitHub Actions / Docker (advanced)

If you prefer deploying from a container image, add a GitHub Actions workflow to build and push a Docker image to GHCR and then deploy to your chosen host (e.g., DigitalOcean, AWS ECS, Azure Container Instances). Example steps:

- Build image with `docker build -t ghcr.io/<OWNER>/<REPO>:latest .`
- Push to GHCR with `docker push ghcr.io/<OWNER>/<REPO>:latest`
- Deploy the image to your hosting provider.

I can add a sample GitHub Actions workflow file if you'd like — tell me which registry or host you prefer.

## Local batch predictions (no API)

Use the provided local batch runner which calls the predictor directly:

```powershell
& .venv\Scripts\python.exe tools/run_batch_local.py --folder data/validation --limit 5
```

### Retraining from uploaded files

If you use the Streamlit Monitoring → Upload training audio files UI, uploaded files are saved to `data/uploads/<class>`.

You can prepare those uploads for retraining using the helper script:

```powershell
& .venv\Scripts\python.exe tools/retrain_from_uploads.py --uploads-dir data/uploads --target-data data --validation-split 0.2
```

By default the script copies files into `data/train/<class>` and `data/validation/<class>` and then launches `python -m src.train`.

You can run a dry-run to preview actions:

```powershell
& .venv\Scripts\python.exe tools/retrain_from_uploads.py --uploads-dir data/uploads --dry-run
```

### Video demo (required for submission)

Please record a short camera-on demo (5–8 minutes) showing:

- Single prediction using the UI
- Uploading new training files via Monitoring → Upload training audio files
- Triggering retraining using the "🔁 Retrain Model" button
- Viewing monitoring metrics or `outputs/` artifacts

After uploading the video to YouTube (unlisted is fine), add the link here:

Video Demo: <ADD_YOUTUBE_LINK_HERE>

If you need a recording script or checklist, see `tools/record_demo_instructions.md`.

### Locust load test (reproduce headless smoke test)

We used Locust to run a headless smoke test against the deployed app's health endpoint. To reproduce locally (in the venv):

```powershell
& .venv\Scripts\python.exe -m locust -f tests/locustfile.py --headless -u 10 -r 2 --run-time 1m --host https://heartbeat-ai-classifier.streamlit.app
```

A summary of the last run is saved in `locust_results/locust_summary.json`.

## Notes
- The Streamlit frontend uses `src.prediction.HeartbeatPredictor` for inference.
- Training can be triggered from the Monitoring page; retraining runs in a background subprocess.
- You mentioned `src/api.py` was deleted — the repository is Streamlit-first and does not require the separate API.
