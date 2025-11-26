"""Locust load test that posts audio files to POST /predict.

Usage (headless):
  locust -f tests/locust_predict.py --headless -u 20 -r 2 --run-time 1m --host http://localhost:8000

This script chooses sample files from `tests/locust_samples` if present, otherwise `data/validation`.
"""
from locust import HttpUser, task, between
from pathlib import Path
import random


SAMPLES_DIRS = [Path("tests/locust_samples"), Path("data/validation")]


def _collect_samples():
    for d in SAMPLES_DIRS:
        if d.exists():
            files = list(d.rglob("*.wav"))
            if files:
                return files
    return []


SAMPLES = _collect_samples()


class PredictUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        if not SAMPLES:
            # fallback to health check if no samples available
            self.client.get("/health")
            return

        sample = random.choice(SAMPLES)
        with open(sample, "rb") as fh:
            files = {"file": (sample.name, fh, "audio/wav")}
            self.client.post("/predict", files=files, timeout=30)
