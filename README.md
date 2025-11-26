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

## Notes
- The Streamlit frontend uses `src.prediction.HeartbeatPredictor` for inference.
- Training can be triggered from the Monitoring page; retraining runs in a background subprocess.
- You mentioned `src/api.py` was deleted — the repository is Streamlit-first and does not require the separate API.
