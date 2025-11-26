"""
HeartBeat AI - Unified Streamlit Application
All-in-one: ML model + UI + Dashboard (No separate API needed)
"""

import io
import gc
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import librosa
import numpy as np

# -----------------------------------------------------------
# Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="HeartBeat AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
# Model / Predictor Loading (cached)
# Use the central predictor implementation from `src.prediction`
# -----------------------------------------------------------
@st.cache_resource
def load_predictor():
    """Instantiate and load `src.prediction.HeartbeatPredictor`.

    The predictor expects a file-based model path; we use the packaged
    model at `models/cardiac_cnn_model.h5`.
    """
    try:
        from src.prediction import HeartbeatPredictor
    except Exception as e:
        st.error(f"Could not import predictor from src.prediction: {e}")
        return None

    model_path = Path("models/cardiac_cnn_model.h5")
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        return None

    predictor = HeartbeatPredictor(model_path)
    try:
        with st.spinner("Loading AI model..."):
            predictor.load()
        return predictor
    except Exception as e:
        st.error(f"❌ Failed to load predictor: {e}")
        return None

def preprocess_audio(wav: np.ndarray, sr: int):
    """Preprocess audio into mel-spectrogram."""
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
    
    # Cleanup
    del wav, mel, mel_db
    gc.collect()
    
    return result

def predict_audio(predictor, uploaded_file):
    """Run prediction using `HeartbeatPredictor` on an uploaded file-like object.

    The predictor works with file paths, so we write the uploaded bytes to a
    temporary file and call `predict_with_details` to get rich outputs.
    """
    import tempfile
    try:
        # Ensure file pointer at start
        uploaded_file.seek(0)
        suffix = Path(uploaded_file.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        # Use predictor to get detailed prediction if available, otherwise fall back
        try:
            if hasattr(predictor, "predict_with_details"):
                result = predictor.predict_with_details(tmp_path)
            else:
                # older predictor implementations may only expose `predict`
                base = predictor.predict(tmp_path)
                # Ensure consistent keys expected by the UI
                result = base.copy()
                result.setdefault("waveform", None)
                result.setdefault("spectrogram", None)
                result.setdefault("sample_rate", None)
                result.setdefault("duration_seconds", None)
        except Exception as ex:
            # Try a safer fallback: force single-input spectrogram prediction if the
            # predictor is an ensemble and the ensemble path fails at runtime.
            try:
                if hasattr(predictor, "is_ensemble") and predictor.is_ensemble:
                    predictor.is_ensemble = False
                    base = predictor.predict(tmp_path)
                    result = base.copy()
                    result.setdefault("waveform", None)
                    result.setdefault("spectrogram", None)
                    result.setdefault("sample_rate", None)
                    result.setdefault("duration_seconds", None)
                    result["warning"] = "Ensemble inference failed; used spectrogram-only fallback"
                else:
                    raise
            except Exception as ex2:
                st.error(f"Prediction failed: {ex2}")
                return None

        # Cleanup temporary file
        try:
            tmp_path.unlink()
        except Exception:
            pass

        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -----------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "startup_time" not in st.session_state:
    st.session_state.startup_time = datetime.now()

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
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
# Sidebar Navigation
# -----------------------------------------------------------
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
        "📈 Monitoring",
    ],
    label_visibility="collapsed"
)

# System Status in Sidebar
st.sidebar.divider()
st.sidebar.caption("System Status")

# Load model status
predictor = load_predictor()
if predictor is not None:
    st.sidebar.success("✅ Model Ready")
else:
    st.sidebar.error("🔴 Model Not Loaded")

# Uptime
uptime = datetime.now() - st.session_state.startup_time
hours = int(uptime.total_seconds() // 3600)
minutes = int((uptime.total_seconds() % 3600) // 60)
st.sidebar.metric("Uptime", f"{hours}h {minutes}m")

# Predictions count
st.sidebar.metric("Predictions", len(st.session_state.prediction_history))

st.sidebar.divider()
st.sidebar.caption("HeartBeat AI v1.0.0")
st.sidebar.caption("Cardiac Sound Classification System")

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
def show_home_page():
    st.markdown("# ❤️ HeartBeat AI — Clinical Dashboard")
    
    # System Overview
    st.header("System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if predictor is not None:
            st.metric("System Status", "✅ Online", delta="Healthy")
        else:
            st.metric("System Status", "❌ Offline", delta="Error")
    
    with col2:
        if predictor is not None:
            st.metric("Model Status", "✅ Loaded", delta="Ready")
        else:
            st.metric("Model Status", "❌ Not Loaded", delta="Inactive")
    
    with col3:
        uptime_seconds = (datetime.now() - st.session_state.startup_time).total_seconds()
        st.metric("Uptime", f"{int(uptime_seconds // 60)} min")
    
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
            st.metric("Recall", f"{rec*100:.1f}%", help="Sensitivity")
        
        with col4:
            f1 = metrics.get('F1 Score', 0)
            st.metric("F1 Score", f"{f1*100:.1f}%", help="Harmonic mean")
        
        with col5:
            auc = metrics.get('AUC-ROC', 0)
            st.metric("AUC-ROC", f"{auc:.3f}", help="Area under ROC curve")
        
        # Performance insights
        if acc >= 0.85:
            st.success("✅ Model is performing well with high accuracy!")
        elif acc >= 0.75:
            st.info("ℹ️ Model performance is acceptable.")
        else:
            st.warning("⚠️ Model performance could be improved.")
    else:
        st.info("📊 No metrics available. Run training to generate metrics.")
    
    # Recent Predictions
    st.header("Recent Predictions")
    
    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("📭 No predictions yet. Upload an audio file to make a prediction.")

# -----------------------------------------------------------
# PREDICTION PAGE
# -----------------------------------------------------------
def show_prediction_page():
    st.markdown("# 🔍 Cardiac Sound Analysis")
    
    if predictor is None:
        st.error("❌ Model not loaded. Please check the model file.")
        return
    
    # Single Prediction
    st.header("Single Audio Prediction")
    
    file = st.file_uploader(
        "Upload cardiac sound recording",
        type=["wav", "mp3", "flac", "ogg"],
        help="Upload a cardiac auscultation recording"
    )
    
    if file:
        st.audio(file, format=f"audio/{file.name.split('.')[-1]}")
        st.caption(f"📁 File: {file.name} ({file.size / 1024:.1f} KB)")
    
    if st.button("🔬 Analyze Heart Sound", disabled=file is None):
        if file:
            with st.spinner("Analyzing cardiac sound..."):
                start_time = time.time()
                
                # Reset file pointer
                file.seek(0)
                result = predict_audio(predictor, file)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if result:
                    # Store in history
                    result["file_name"] = file.name
                    result["timestamp"] = datetime.now().isoformat()
                    result["processing_time"] = processing_time
                    st.session_state.prediction_history.append(result)
                    
                    # Display result
                    predicted_class = result["predicted_class"]
                    confidence = result["confidence"]
                    prob_normal = result["probability_normal"]
                    prob_abnormal = result["probability_abnormal"]
                    
                    # Styled result box
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
                        st.warning("Potential cardiac abnormality detected. Consult a cardiologist.")
                    
                    # Detailed Analysis
                    st.subheader("Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Normal Probability", f"{prob_normal*100:.1f}%")
                        st.metric("Abnormal Probability", f"{prob_abnormal*100:.1f}%")
                    
                    with col2:
                        # Probability chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=["Normal", "Abnormal"],
                                y=[prob_normal * 100, prob_abnormal * 100],
                                marker_color=["green", "red"],
                                text=[f"{prob_normal*100:.1f}%", f"{prob_abnormal*100:.1f}%"],
                                textposition="auto"
                            )
                        ])
                        fig.update_layout(
                            yaxis_title="Probability (%)",
                            showlegend=False,
                            height=250
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"⏱️ Processing time: {processing_time:.2f} seconds")
                    
                    # Technical details
                    with st.expander("📋 View Technical Details"):
                        st.json(result)
    
    st.divider()
    
    # Batch Prediction
    st.header("Batch Prediction")
    st.caption("Upload multiple audio files for bulk analysis")
    
    batch_files = st.file_uploader(
        "Upload multiple cardiac sound recordings",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        key="batch"
    )
    
    if batch_files:
        st.info(f"📁 {len(batch_files)} files selected")
    
    if st.button("🔍 Analyze All Files", disabled=not batch_files):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []

        for idx, f in enumerate(batch_files):
            status_text.text(f"Processing {idx+1}/{len(batch_files)}: {f.name}")
            progress_bar.progress((idx + 1) / len(batch_files))

            f.seek(0)
            result = predict_audio(predictor, f)
            
            if result:
                result["file_name"] = f.name
                result["timestamp"] = datetime.now().isoformat()
                result["status"] = "success"
                st.session_state.prediction_history.append(result)
                results.append(result)
            else:
                results.append({
                    "file_name": f.name,
                    "status": "error",
                    "error": "Processing failed"
                })
        
        status_text.text("✅ Processing complete!")
        
        if results:
            df = pd.DataFrame(results)
            
            # Summary
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
            
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                f"cardiac_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

# -----------------------------------------------------------
# VISUALIZATIONS PAGE
# -----------------------------------------------------------
def show_visualizations_page():
    st.markdown("# 📊 Model Visualizations")
    
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        st.warning("⚠️ No visualizations found. Run the training notebook to generate visualizations.")
        return
    
    # Class Distribution
    st.header("Dataset Analysis")
    class_dist = outputs_dir / "class_distribution.png"
    if class_dist.exists():
        img = Image.open(class_dist)
        st.image(img, caption="Training Data Class Distribution", use_container_width=True)
    
    # Training History
    st.header("Model Training Performance")
    training_history = outputs_dir / "training_history.png"
    if training_history.exists():
        img = Image.open(training_history)
        st.image(img, caption="Training and Validation Metrics", use_container_width=True)
    
    # Confusion Matrix
    st.header("Model Evaluation")
    confusion_matrix = outputs_dir / "confusion_matrix.png"
    if confusion_matrix.exists():
        img = Image.open(confusion_matrix)
        st.image(img, caption="Confusion Matrix", use_container_width=True)
    
    # ROC Curve
    st.header("Classifier Performance")
    roc_curve = outputs_dir / "roc_curve.png"
    if roc_curve.exists():
        img = Image.open(roc_curve)
        st.image(img, caption="ROC Curve", use_container_width=True)

# -----------------------------------------------------------
# MONITORING PAGE
# -----------------------------------------------------------
def show_monitoring_page():
    st.markdown("# 📈 System Monitoring")
    
    # Model Performance Metrics
    st.header("Model Performance Metrics")
    
    metrics = load_local_metrics()
    
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc = metrics.get('Accuracy', 0)
            st.metric("Accuracy", f"{acc:.4f}")
        
        with col2:
            prec = metrics.get('Precision', 0)
            st.metric("Precision", f"{prec:.4f}")
        
        with col3:
            rec = metrics.get('Recall', 0)
            st.metric("Recall", f"{rec:.4f}")
        
        with col4:
            f1 = metrics.get('F1 Score', 0)
            st.metric("F1 Score", f"{f1:.4f}")
        
        # Detailed table
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.4f}" for v in metrics.values()]
        })
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("No metrics available.")
    
    # Prediction Analytics
    st.header("Prediction Analytics")
    
    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        
        # Recent predictions
        st.subheader("Recent Predictions")
        st.dataframe(df.tail(20), use_container_width=True)
        
        # Distribution pie chart
        if 'predicted_class' in df.columns:
            st.subheader("Classification Distribution")
            
            fig = px.pie(
                df,
                names='predicted_class',
                color='predicted_class',
                color_discrete_map={'normal': 'green', 'abnormal': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            total = len(df)
            normal = len(df[df['predicted_class'] == 'normal'])
            abnormal = total - normal
            
            with col1:
                st.metric("Total Predictions", total)
            with col2:
                st.metric("Normal", normal, delta=f"{normal/total*100:.1f}%")
            with col3:
                st.metric("Abnormal", abnormal, delta=f"{abnormal/total*100:.1f}%")
    else:
        st.info("No predictions recorded yet.")

    st.divider()
    st.header("Model Management")
    st.caption("Retrain model or upload new training examples")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔁 Retrain Model"):
            st.info("Starting retraining in background...")
            import subprocess, sys
            cwd = Path(__file__).resolve().parents[1]
            try:
                subprocess.Popen([sys.executable, '-m', 'src.train'], cwd=str(cwd))
                st.success("Retraining started")
            except Exception as e:
                st.error(f"Failed to start retrain: {e}")

    with col2:
        upload_class = st.selectbox("Target class for uploads", ["normal", "abnormal"], index=0)
        uploaded = st.file_uploader("Upload training audio files", type=["wav", "mp3", "flac"], accept_multiple_files=True)
        if uploaded and st.button("📤 Upload to training dataset"):
            saved = 0
            upload_dir = Path(f"data/uploads/{upload_class}")
            upload_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                try:
                    f.seek(0)
                    (upload_dir / f.name).write_bytes(f.read())
                    saved += 1
                except Exception as e:
                    st.warning(f"Failed to save {f.name}: {e}")
            st.success(f"Saved {saved} files to {upload_dir}")

# -----------------------------------------------------------
# Route to Selected Page
# -----------------------------------------------------------
if page == "🏠 Home":
    show_home_page()
elif page == "🔍 Prediction":
    show_prediction_page()
elif page == "📊 Visualizations":
    show_visualizations_page()
elif page == "📈 Monitoring":
    show_monitoring_page()