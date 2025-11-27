Streamlit Deployment Notes
==========================

Recommended runtime for Streamlit Community Cloud
------------------------------------------------

- Use Python 3.11.x for deployments (e.g. `3.11.13`) — many ML wheels (TensorFlow, PyTorch)
  provide prebuilt binaries for Python 3.11/3.12 but not for 3.13. Setting the app
  runtime to Python 3.11 avoids lengthy or failing builds when installing heavy
  packages like `tensorflow-cpu`.

How to set Python version on Streamlit Cloud
--------------------------------------------

1. In the Streamlit Community Cloud dashboard, open your app's settings.
2. Under "Advanced" or "Runtime", set the Python version to a 3.11.x release.
3. Redeploy the app.

Alternative: deploy with a minimal runtime requirements file
-----------------------------------------------------------

If your Streamlit frontend does not need the full ML stack (training/inference),
use a lightweight `requirements.txt` for the deployed app and keep the full
dependencies in a separate `requirements-dev.txt` for development and training.

Suggested minimal `requirements.txt` for the frontend only:

```
streamlit==1.26.0
requests==2.31.0
plotly==5.16.1
pandas==2.1.0
pillow==9.5.0
psutil==5.9.5
# librosa optional if you preprocess audio in the frontend
# librosa==0.10.1
```

Then keep the full ML/development deps in `requirements-dev.txt` (unchanged).

Why this matters
-----------------

- Streamlit Cloud environments may use newer Python versions; many prebuilt
  binary wheels for TF/torch aren't available for the newest ABIs and pip will
  try to build from source (slow and often failing).
- Setting the correct Python runtime or slimming `requirements.txt` avoids
  installation failures and makes deployments fast and reliable.

If you'd like, I can:
- Create `requirements-dev.txt` with the current full dependency list and replace
  `requirements.txt` with a minimal file (safe option for Streamlit Cloud).
- Or just keep this README and ask you to change the Streamlit runtime to 3.11.
