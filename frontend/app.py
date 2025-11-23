"""
Streamlit Dashboard for HeartBeat AI Cardiac Sound Classifier
Production-ready version with all features fully functional
"""

import io
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from PIL import Image
import plotly.express as px
import requests
import streamlit as st

# -----------------------------------------------------------
# Streamlit App Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="HeartBeat AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = os.getenv("API_BASE_URL", "https://heartbeat-ai-api.onrender.com")

# Custom CSS for better UI
st.markdown("""
<style>
    .success-box {
        background-color: #27ae60;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .warning-box {
        background-color: #e74c3c;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #3498db;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# API CALL HELPERS
# -----------------------------------------------------------
def fetch_json(endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    """Generic GET/POST JSON fetcher with error handling."""
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Make sure the API is running.")
        return {}
    except requests.exceptions.HTTPError as e:
        st.error(f"API error ({e.response.status_code}): {e.response.text}")
        return {}
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return {}

def upload_file(endpoint: str, file_data, data=None, timeout: int = 120):
    """Upload files to the backend with error handling."""
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, files=file_data, data=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API. Is it running at {API_BASE}?")
        return {}
    except requests.RequestException as exc:
        st.error(f"Upload failed: {exc}")
        return {}

def load_local_metrics():
    """Load metrics from local outputs folder."""
    metrics_file = Path("outputs/metrics_summary.csv")
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            return df.set_index('Metric')['Value'].to_dict()
        except Exception as e:
            st.warning(f"Could not load metrics: {e}")
    return {}

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
def show_home_page():
    st.markdown("# ❤️ HeartBeat AI — Clinical Dashboard")
    
    # System Overview
    st.header("System Overview")
    health = fetch_json("health")
    
    if not health:
        st.error("API is not responding. Please start the API server.")
        st.code("python src/api.py", language="bash")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = health.get("status", "unknown")
        if status == "ok":
            st.metric("API Status", "✅ Online", delta="Healthy")
        else:
            st.metric("API Status", "❌ Offline", delta="Down")
    
    with col2:
        model_loaded = health.get("model_loaded", False)
        if model_loaded:
            st.metric("Model Loaded", "✅ Ready", delta="Active")
        else:
            st.metric("Model Loaded", "❌ Not Loaded", delta="Inactive")
    
    with col3:
        uptime_data = fetch_json("uptime")
        if uptime_data:
            uptime_seconds = uptime_data.get("uptime_seconds", 0)
            st.metric("Uptime", f"{int(uptime_seconds // 60)} min")
        else:
            st.metric("Uptime", "N/A")
    
    # Model Performance
    st.header("Model Performance")
    
    metrics = load_local_metrics()
    
    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            acc = metrics.get('Accuracy', 0)
            st.metric("Accuracy", f"{acc*100:.1f}%", help="Overall classification accuracy")
        
        with col2:
            prec = metrics.get('Precision', 0)
            st.metric("Precision", f"{prec*100:.1f}%", help="Positive predictive value")
        
        with col3:
            rec = metrics.get('Recall', 0)
            st.metric("Recall", f"{rec*100:.1f}%", help="Sensitivity / True positive rate")
        
        with col4:
            f1 = metrics.get('F1 Score', 0)
            st.metric("F1 Score", f"{f1*100:.1f}%", help="Harmonic mean of precision and recall")
        
        with col5:
            auc = metrics.get('AUC-ROC', 0)
            st.metric("AUC-ROC", f"{auc:.3f}", help="Area under ROC curve")
        
        # Performance insights
        if acc >= 0.85:
            st.success("Model is performing well with high accuracy!")
        elif acc >= 0.75:
            st.info("Model performance is acceptable.")
        else:
            st.warning("Model performance could be improved with more training data.")
    else:
        st.info("No metrics available. Run the training notebook to generate metrics.")
    
    # Recent Predictions
    st.header("Recent Predictions")
    
    history = fetch_json("visualizations/prediction-history")
    if history:
        history_data = history.get("history", [])
        if history_data:
            df = pd.DataFrame(history_data)
            # Show most recent first
            st.dataframe(df.tail(10).sort_values('timestamp', ascending=False) if 'timestamp' in df.columns else df.tail(10))
        else:
            st.info("No predictions logged yet. Upload an audio file to make a prediction.")
    else:
        st.info("No predictions logged yet.")

# -----------------------------------------------------------
# PREDICTION PAGE
# -----------------------------------------------------------
def show_prediction_page():
    st.markdown("# ❤️ HeartBeat AI — Prediction")
    
    # Check API health first
    health = fetch_json("health")
    if not health or not health.get("model_loaded"):
        st.error("Model is not loaded. Please check API status.")
        return
    
    # Single Audio Prediction
    st.header("Single Audio Prediction")
    
    file = st.file_uploader(
        "Upload cardiac sound recording (.wav)",
        type=["wav", "mp3", "flac"],
        help="Upload a 5-second cardiac auscultation recording"
    )
    
    if file:
        st.audio(file, format="audio/wav")
        st.caption(f"File: {file.name} ({file.size / 1024:.1f} KB)")
    
    if st.button("🔍 Analyze Heart Sound", disabled=file is None, type="primary", use_container_width=True):
        if file:
            with st.spinner("Analyzing cardiac sound..."):
                # Reset file pointer
                file.seek(0)
                result = upload_file("predict", {"file": (file.name, file, file.type)})
                
                if result:
                    # Handle both direct and nested prediction responses
                    prediction = result.get("prediction", result)
                    predicted_class = prediction.get("predicted_class", "unknown")
                    confidence = prediction.get("confidence", 0)
                    prob_normal = prediction.get("probability_normal", 0)
                    prob_abnormal = prediction.get("probability_abnormal", 0)
                    
                    # Display result with styling
                    if predicted_class == "normal":
                        st.markdown(
                            f'<div class="success-box">✅ Result: NORMAL<br>'
                            f'Confidence: {confidence*100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.success("No cardiac abnormality detected. Continue routine monitoring.")
                    else:
                        st.markdown(
                            f'<div class="warning-box">⚠️ Result: ABNORMAL<br>'
                            f'Confidence: {confidence*100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.warning("Potential cardiac abnormality detected. Consult a cardiologist for further evaluation.")
                    
                    # Detailed probabilities
                    st.subheader("Detailed Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Normal Probability",
                            f"{prob_normal*100:.1f}%",
                            help="Probability of normal heart sound"
                        )
                    
                    with col2:
                        st.metric(
                            "Abnormal Probability",
                            f"{prob_abnormal*100:.1f}%",
                            help="Probability of abnormal heart sound"
                        )
                    
                    # Visualization
                    prob_df = pd.DataFrame({
                        'Class': ['Normal', 'Abnormal'],
                        'Probability': [prob_normal * 100, prob_abnormal * 100]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Class',
                        color_discrete_map={'Normal': 'green', 'Abnormal': 'red'},
                        title='Classification Probabilities'
                    )
                    fig.update_layout(showlegend=False, yaxis_title="Probability (%)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Raw response
                    with st.expander("📋 View Technical Details"):
                        st.json(result)
    
    st.divider()
    
    # Batch Prediction
    st.header("Batch Prediction")
    st.caption("Upload multiple audio files for bulk analysis")
    
    batch_files = st.file_uploader(
        "Upload multiple cardiac sound recordings",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
        key="batch",
        help="Select multiple .wav files for batch processing"
    )
    
    if batch_files:
        st.info(f"📁 {len(batch_files)} files selected")
    
    if st.button("🔍 Analyze All Files", disabled=not batch_files, type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner(f"Processing {len(batch_files)} files..."):
            # Prepare payload
            payload = []
            for f in batch_files:
                f.seek(0)
                payload.append(("files", (f.name, io.BytesIO(f.read()), f.type or "audio/wav")))
            
            # Try batch endpoint
            resp = upload_file("batch-predict", payload, timeout=180)
            
            if resp and "results" in resp:
                results = resp["results"]
            else:
                # Fallback: process individually
                results = []
                for idx, f in enumerate(batch_files):
                    status_text.text(f"Processing {idx+1}/{len(batch_files)}: {f.name}")
                    progress_bar.progress((idx + 1) / len(batch_files))
                    
                    f.seek(0)
                    single_result = upload_file("predict", {"file": (f.name, f, f.type)}, timeout=60)
                    
                    if single_result:
                        pred = single_result.get("prediction", single_result)
                        results.append({
                            "file_name": f.name,
                            "predicted_class": pred.get("predicted_class", "error"),
                            "confidence": pred.get("confidence", 0),
                            "status": "success"
                        })
                    else:
                        results.append({
                            "file_name": f.name,
                            "status": "error",
                            "error": "Processing failed"
                        })
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            if results:
                df = pd.DataFrame(results)
                
                # Summary statistics
                successful = df[df.get('status', 'success') == 'success']
                if len(successful) > 0:
                    normal_count = len(successful[successful['predicted_class'] == 'normal'])
                    abnormal_count = len(successful) - normal_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(successful))
                    with col2:
                        st.metric("Normal", normal_count, delta=f"{normal_count/len(successful)*100:.1f}%")
                    with col3:
                        st.metric("Abnormal", abnormal_count, delta=f"{abnormal_count/len(successful)*100:.1f}%")
                
                # Results table
                st.subheader("Results")
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results as CSV",
                    csv,
                    f"cardiac_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

# -----------------------------------------------------------
# VISUALIZATION PAGE
# -----------------------------------------------------------
def show_visualizations_page():
    st.markdown("# ❤️ HeartBeat AI — Visualizations")
    
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        st.warning("⚠️ No visualizations found. Run the training notebook first to generate visualizations.")
        st.code("jupyter notebook notebook/heartbeat_ai_eda.ipynb", language="bash")
        return
    
    # Class Distribution
    st.header("Dataset Analysis")
    class_dist = outputs_dir / "class_distribution.png"
    if class_dist.exists():
        try:
            img = Image.open(class_dist)
            st.image(img, caption="Training Data Class Distribution", width=600)
        except Exception as e:
            st.error(f"Could not load image {class_dist.name}: {e}")
    
    # Training History
    st.header("Model Training Performance")
    training_history = outputs_dir / "training_history.png"
    if training_history.exists():
        try:
            img = Image.open(training_history)
            st.image(img, caption="Training and Validation Metrics Over Epochs", width=700)
            st.caption("Shows accuracy, loss, precision, and recall during training")
        except Exception as e:
            st.error(f"Could not load image {training_history.name}: {e}")
    
    # Confusion Matrix
    st.header("Model Evaluation")
    confusion_matrix = outputs_dir / "confusion_matrix.png"
    if confusion_matrix.exists():
        try:
            img = Image.open(confusion_matrix)
            st.image(img, caption="Confusion Matrix - Prediction Accuracy", width=700)
            st.caption("Left: Raw counts | Right: Normalized percentages")
        except Exception as e:
            st.error(f"Could not load image {confusion_matrix.name}: {e}")
    
    # ROC Curve
    st.header("Classifier Performance")
    roc_curve = outputs_dir / "roc_curve.png"
    if roc_curve.exists():
        try:
            img = Image.open(roc_curve)
            st.image(img, caption="ROC Curve (AUC: 0.90)", width=700)
            st.caption("Receiver Operating Characteristic - Shows model's diagnostic ability")
        except Exception as e:
            st.error(f"Could not load image {roc_curve.name}: {e}")
    
    # Feature Analysis
    st.header("Audio Feature Analysis")
    st.caption("Distinguishing characteristics between normal and abnormal heart sounds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        freq = outputs_dir / "frequency_band_analysis.png"
        if freq.exists():
            try:
                img = Image.open(freq)
                st.image(img, caption="Frequency Band Energy", width=400)
            except Exception as e:
                st.error(f"Could not load image {freq.name}: {e}")
    
    with col2:
        temp = outputs_dir / "temporal_analysis.png"
        if temp.exists():
            try:
                img = Image.open(temp)
                st.image(img, caption="Temporal Patterns", width=400)
            except Exception as e:
                st.error(f"Could not load image {temp.name}: {e}")
    
    with col3:
        spec = outputs_dir / "spectral_analysis.png"
        if spec.exists():
            try:
                img = Image.open(spec)
                st.image(img, caption="Spectral Features", width=400)
            except Exception as e:
                st.error(f"Could not load image {spec.name}: {e}")

# -----------------------------------------------------------
# UPLOAD + RETRAIN PAGE
# -----------------------------------------------------------
def show_upload_retrain_page():
    st.markdown("# ❤️ HeartBeat AI — Model Retraining")
    
    st.info("📚 This feature allows you to upload new labeled data and retrain the model to improve its performance.")
    
    st.header("Upload Labeled Audio Data")
    
    target_class = st.selectbox(
        "Select the target class for uploaded files",
        ["normal", "abnormal"],
        help="Choose whether the uploaded files are normal or abnormal heart sounds"
    )
    
    files = st.file_uploader(
        "Upload cardiac sound recordings",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
        key="upload",
        help="Upload .wav files of cardiac auscultation recordings"
    )
    
    if files:
        st.success(f"📁 {len(files)} files selected for class: **{target_class}**")
        
        # Show file list
        with st.expander("View selected files"):
            for f in files:
                st.text(f"• {f.name} ({f.size / 1024:.1f} KB)")
    
    if st.button("📤 Upload Training Data", disabled=not files, type="primary", use_container_width=True):
        with st.spinner(f"Uploading {len(files)} files..."):
            payload = []
            for f in files:
                f.seek(0)
                payload.append(("files", (f.name, io.BytesIO(f.read()), f.type or "audio/wav")))
            
            try:
                resp = requests.post(
                    f"{API_BASE}/upload-training-data",
                    files=payload,
                    data={"target_class": target_class},
                    timeout=120
                )
                resp.raise_for_status()
                result = resp.json()
                
                st.success(f"✅ Successfully uploaded {result.get('count', 0)} files to class '{target_class}'")
                st.balloons()
            except requests.RequestException as exc:
                st.error(f"❌ Upload failed: {exc}")
    
    st.divider()
    
    # Retraining Controls
    st.header("Model Retraining")
    
    st.warning("⚠️ **Important:** Retraining will take 10-30 minutes and the API will be temporarily unavailable during this process.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Retraining Process:**
        1. Uploaded data will be combined with existing training data
        2. Model will be retrained from scratch with updated dataset
        3. New model will replace the current model
        4. Performance metrics will be recalculated
        """)
    
    with col2:
        if st.button("🔄 Start Retraining", type="primary", use_container_width=True):
            with st.spinner("Initiating retraining..."):
                resp = fetch_json("retrain", method="POST")
                if resp:
                    st.success(f"✅ {resp.get('message', 'Retraining initiated successfully')}")
                    st.info("Check training status below for progress updates.")
    
    # Training Status
    st.subheader("Training Status")
    
    status = fetch_json("training-status")
    if status:
        progress = status.get("progress", 0)
        st.progress(progress / 100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", status.get("status", "idle").upper())
        with col2:
            st.metric("Progress", f"{progress}%")
        
        if status.get("message"):
            st.info(status.get("message"))

# -----------------------------------------------------------
# MONITORING PAGE
# -----------------------------------------------------------
def show_monitoring_page():
    st.markdown("# ❤️ HeartBeat AI — System Monitoring")
    
    # System Metrics
    st.header("Model Performance Metrics")
    
    metrics = load_local_metrics()
    
    if metrics:
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc = metrics.get('Accuracy', 0)
            st.metric("Accuracy", f"{acc:.4f}", help="Overall classification accuracy")
        
        with col2:
            prec = metrics.get('Precision', 0)
            st.metric("Precision", f"{prec:.4f}", help="Positive predictive value")
        
        with col3:
            rec = metrics.get('Recall', 0)
            st.metric("Recall", f"{rec:.4f}", help="True positive rate")
        
        with col4:
            f1 = metrics.get('F1 Score', 0)
            st.metric("F1 Score", f"{f1:.4f}", help="Harmonic mean")
        
        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.4f}" for v in metrics.values()]
        })
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No metrics available. Train the model to generate performance metrics.")
    
    # Prediction History
    st.header("Prediction History & Analytics")
    
    history = fetch_json("visualizations/prediction-history")
    if history:
        history_data = history.get("history", [])
        
        if history_data:
            df = pd.DataFrame(history_data)
            
            # Show recent predictions
            st.subheader("Recent Predictions")
            st.dataframe(df.tail(20), use_container_width=True)
            
            # Prediction distribution
            if 'predicted_class' in df.columns:
                st.subheader("Prediction Distribution")
                
                fig = px.pie(
                    df,
                    names='predicted_class',
                    title='Classification Distribution',
                    color='predicted_class',
                    color_discrete_map={'normal': 'green', 'abnormal': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                
                total = len(df)
                normal = len(df[df['predicted_class'] == 'normal'])
                abnormal = len(df[df['predicted_class'] == 'abnormal'])
                
                with col1:
                    st.metric("Total Predictions", total)
                with col2:
                    st.metric("Normal", normal, delta=f"{normal/total*100:.1f}%")
                with col3:
                    st.metric("Abnormal", abnormal, delta=f"{abnormal/total*100:.1f}%")
        else:
            st.info("No predictions recorded yet. Make some predictions to see analytics.")
    else:
        st.info("No prediction history available.")

# -----------------------------------------------------------
# MAIN NAVIGATION
# -----------------------------------------------------------
def main():
    # Sidebar with logo and navigation
    logo_path = Path("Graphics/Cardilogy-Heart-Technology-Concept.jpg")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=250)
    else:
        st.sidebar.markdown("# ❤️ HeartBeat AI")
    
    st.sidebar.markdown("### Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Home",
            "🔍 Prediction",
            "📊 Visualizations",
            "📤 Upload & Retrain",
            "📈 Monitoring",
        ],
        label_visibility="collapsed"
    )
    
    # API status in sidebar
    st.sidebar.divider()
    st.sidebar.caption("API Status")
    health = fetch_json("health")
    if health and health.get("status") == "ok":
        st.sidebar.success("✅ API Online")
    else:
        st.sidebar.error("❌ API Offline")
    
    # Route to pages
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
    
    # Footer
    st.sidebar.divider()
    st.sidebar.caption("HeartBeat AI v1.0.0")
    st.sidebar.caption("Cardiac Sound Classification System")


if __name__ == "__main__":
    main()
