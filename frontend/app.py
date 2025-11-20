import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="HeartBeat AI", page_icon="❤", layout="wide")

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def fetch_json(endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return {}


def upload_file(endpoint: str, file_data: Dict[str, Any], data: Dict[str, Any] | None = None):
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.post(url, files=file_data, data=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        return {}


def show_home_page():
    col1, col2 = st.columns([2, 1])
    metrics = fetch_json("metrics")
    with col1:
        st.subheader("System Overview")
        health = fetch_json("health")
        st.metric("API Status", health.get("status", "unknown"))
        st.metric("Model Loaded", str(health.get("model_loaded", False)))
        st.metric("Last Update", metrics.get("updated_at", "N/A"))
    with col2:
        st.subheader("Quick Stats")
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}" if metrics else "N/A")
        history = fetch_json("visualizations/prediction-history")
        st.metric("Predictions Logged", len(history.get("history", [])))

    st.markdown("### Recent Predictions")
    history_df = pd.DataFrame(history.get("history", []))
    if not history_df.empty:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        st.dataframe(history_df.sort_values("timestamp", ascending=False).head(10))
    else:
        st.info("No predictions logged yet.")


def show_prediction_page():
    st.markdown("### Single Audio Prediction")
    file = st.file_uploader("Upload auscultation audio (.wav)", type=["wav", "mp3", "flac"])
    if st.button("Predict", disabled=file is None):
        if file:
            result = upload_file("predict", {"file": (file.name, file, file.type)})
            if result:
                st.success(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
                st.json(result)

    st.markdown("---")
    st.markdown("### Batch Prediction")
    batch_files = st.file_uploader(
        "Upload multiple files",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
        key="batch",
    )
    if st.button("Run Batch Prediction", disabled=not batch_files):
        files_payload = [
            ("files", (f.name, io.BytesIO(f.read()), f.type or "application/octet-stream"))
            for f in batch_files
        ]
        url = f"{API_BASE.rstrip('/')}/batch-predict"
        try:
            resp = requests.post(url, files=files_payload, timeout=180)
            resp.raise_for_status()
            batch_results = resp.json()["results"]
            st.dataframe(pd.DataFrame(batch_results))
        except requests.RequestException as exc:
            st.error(f"Batch request failed: {exc}")


def show_visualizations_page():
    st.markdown("### Class Distribution")
    distribution = fetch_json("visualizations/class-distribution").get("distribution", {})
    if distribution:
        dist_df = pd.DataFrame({"class": list(distribution.keys()), "count": list(distribution.values())})
        fig = px.bar(dist_df, x="class", y="count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No class distribution available yet.")

    metrics = fetch_json("metrics")
    if metrics:
        st.markdown("### Training Curves")
        hist_img = metrics.get("history_plot")
        if hist_img and Path(hist_img).exists():
            st.image(hist_img, caption="Training History")
        cm_img = metrics.get("confusion_plot")
        if cm_img and Path(cm_img).exists():
            st.image(cm_img, caption="Confusion Matrix")
    else:
        st.info("Train the model to generate visualizations.")


def show_upload_retrain_page():
    st.markdown("### Upload Labeled Audio")
    target_class = st.selectbox(
        "Target class",
        ["normal_heart", "murmur", "extrasystole", "normal_resp", "wheeze", "crackle"],
    )
    files = st.file_uploader(
        "Upload audio files",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
        key="upload",
    )
    if st.button("Upload Files", disabled=not files):
        files_payload = [
            ("files", (f.name, io.BytesIO(f.read()), f.type or "application/octet-stream"))
            for f in files
        ]
        url = f"{API_BASE.rstrip('/')}/upload-training-data"
        try:
            resp = requests.post(url, files=files_payload, data={"target_class": target_class}, timeout=120)
            resp.raise_for_status()
            st.success(f"Uploaded {resp.json().get('count', 0)} files.")
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    st.markdown("---")
    st.markdown("### Retraining Controls")
    if st.button("Trigger Retraining"):
        resp = fetch_json("retrain", method="POST")
        if resp:
            st.success(resp.get("message", "Retraining started."))

    status = fetch_json("training-status")
    st.progress(status.get("progress", 0) / 100)
    st.write(status.get("message", "Idle"))


def show_monitoring_page():
    st.markdown("### System Metrics")
    metrics = fetch_json("metrics")
    if metrics:
        cols = st.columns(3)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        cols[1].metric("Updated At", metrics.get("updated_at", "N/A"))
        cols[2].metric("Confusion Matrix Size", len(metrics.get("confusion_matrix", [])))

    st.markdown("### Prediction History")
    history = fetch_json("visualizations/prediction-history").get("history", [])
    if history:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df.tail(20))
    else:
        st.info("No predictions recorded yet.")


def main():
    st.sidebar.image("https://img.icons8.com/?size=512&id=63189&format=png", width=80)
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Prediction",
            "📊 Visualizations",
            "📤 Upload & Retrain",
            "📈 Monitoring",
        ],
    )
    st.title("HeartBeat AI Clinical Dashboard")

    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Prediction":
        show_prediction_page()
    elif page == "📊 Visualizations":
        show_visualizations_page()
    elif page == "📤 Upload & Retrain":
        show_upload_retrain_page()
    elif page == "📈 Monitoring":
        show_monitoring_page()


if __name__ == "__main__":
    main()
