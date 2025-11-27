#❤️HeartBeat AI — Cardiac Sound Classification System

## Demo

### Video Demonstration
Watch Full Demo Video: https://youtu.be/lFsv3v2-Lb8


### Live Deployment
**[Access Live Application](https://heartbeat-ai-classifier.streamlit.app)** (Streamlit Community Cloud)

> Production-ready ML pipeline for detecting cardiac abnormalities from heart sound recordings using deep learning.

A complete end-to-end machine learning system that classifies cardiac sounds as **normal** or **abnormal** using a lightweight CNN model. Features interactive web UI, RESTful API, automated retraining, comprehensive monitoring, and load-tested scalability.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [Usage Guide](#usage-guide)
- [Retraining Workflow](#retraining-workflow)
- [Load Testing](#load-testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## Features

### Core Capabilities
- **Single Audio Prediction** - Upload .wav file and get instant classification with confidence scores
- **Batch Processing** - Analyze multiple files simultaneously with downloadable CSV results
- **Automated Retraining** - Upload new labeled data and retrain model with one click
- **Real-time Monitoring** - Track predictions, metrics, system health, and model performance
- **Interactive Dashboard** - User-friendly Streamlit interface with visualizations
- **RESTful API** - FastAPI backend with auto-generated Swagger documentation

### Technical Highlights
- **Optimized Performance** - 30-50ms inference time per prediction
- **Lightweight Model** - 2.5MB model size (75% smaller than baseline)
- **High Accuracy** - 85.5% accuracy on validation set
- **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Load Tested** - Handles 100+ concurrent users
- **Docker Ready** - Containerized deployment option available

---

## System Architecture

<img width="940" height="1614" alt="image" src="https://github.com/user-attachments/assets/308bd879-1145-4003-ba23-5ea7d87015c8" />

### Architecture Overview

The HeartBeat AI system follows a **4-layer architecture**:

#### 1. User Interface Layer
- **Streamlit Dashboard** (Port 8501): Interactive web UI for predictions, visualizations, and monitoring
- **FastAPI Backend** (Port 8000): RESTful API with auto-generated docs at `/docs`
- **Browser Client**: User access point supporting file upload and real-time results

#### 2. Application Layer
- **Prediction Engine**: `HeartbeatPredictor` class handles inference and preprocessing
- **Upload Manager**: Manages file validation and storage for retraining
- **Monitoring Service**: Tracks metrics, uptime, and prediction history
- **Retraining Orchestrator**: Triggers and manages background training jobs

#### 3. Processing Layer
- **Audio Preprocessor**: Loads audio, resamples to 4kHz, extracts 128x79x1 mel-spectrograms
- **CNN Model**: Lightweight 3-layer CNN (614K parameters, 2.5MB)
- **Training Pipeline**: End-to-end training with data augmentation and callbacks
- **Evaluation Module**: Computes comprehensive performance metrics

#### 4. Data Layer
- **Training Data**: PhysioNet Challenge 2016 dataset (3,240 samples, 6 subsets)
- **Model Registry**: Stored models (.h5), metadata, and checkpoints
- **Upload Storage**: User-uploaded files organized by class
- **Metrics Store**: JSON logs, CSVs, and visualizations

### Data Flow Diagrams

**Prediction Flow:**
```
User Upload → Audio Preprocessing → Mel-Spectrogram → CNN Inference → Classification → Results Display
```

**Retraining Flow:**
```
Upload Labeled Data → Store in data/uploads/ → Trigger Training → Train on Combined Data → Update Model & Metrics
```

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM (minimum)
- 2GB disk space

### Installation

**Step 1: Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/cardiac-sound-classification.git
cd cardiac-sound-classification
```

**Step 2: Create Virtual Environment**
```bash
# Windows (PowerShell)
python -m venv .venv
& .venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Verify Installation**
```bash
python -c "import tensorflow; import librosa; import streamlit; print('Installation successful!')"
```

### Running the System

**Option 1: Streamlit Dashboard (Recommended)**
```bash
streamlit run frontend/app.py
```
Navigate to **http://localhost:8501**

**Option 2: FastAPI Backend + Streamlit**
```bash
# Terminal 1: Start API
python src/api.py

# Terminal 2: Start Dashboard
streamlit run frontend/streamlit_app.py
```

**Option 3: Automated Startup**
```bash
python start_system.py
```

### Quick Test

1. Open dashboard at http://localhost:8501
2. Navigate to **Prediction** page
3. Upload a test audio file from `data/test/`
4. Click **"Analyze Heart Sound"**
5. View results with confidence scores

---

## Model Performance

### Evaluation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 85.49% | Overall classification correctness |
| **Precision** | 64.89% | Positive predictive value (abnormal detection) |
| **Recall** | 63.91% | Sensitivity / True positive rate |
| **F1-Score** | 64.39% | Harmonic mean of precision and recall |
| **AUC-ROC** | 89.98% | Area under ROC curve |

### Confusion Matrix

|  | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **Actual Normal** | 468 (91%) | 47 (9%) |
| **Actual Abnormal** | 48 (36%) | 85 (64%) |

### Model Architecture

**Lightweight CNN Specifications:**
- **Input Shape**: 128x79x1 (mel-spectrogram)
- **Architecture**: 3 Conv2D blocks + GlobalAveragePooling + Dense layers
- **Parameters**: 614,433 (75% smaller than baseline)
- **Model Size**: 2.5 MB
- **Inference Time**: 30-50ms per sample
- **Training Time**: ~15 minutes on GPU

### Dataset Details

**PhysioNet/CinC Challenge 2016 Dataset:**
- **Total Samples**: 3,240 cardiac sound recordings
- **Classes**: Normal (79.5%), Abnormal (20.5%)
- **Training Sets**: 6 subsets (a, b, c, d, e, f)
- **Sample Rate**: 4,000 Hz
- **Duration**: ~5 seconds per recording
- **Format**: WAV, mono channel

---

## Usage Guide

### Single Prediction

**Via Web Dashboard:**
1. Navigate to **Prediction** page
2. Upload audio file (.wav, .mp3, .flac)
3. Click **"Analyze Heart Sound"**
4. View classification result and probabilities

**Via API (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/audio.wav"
```

**Via API (Python):**
```python
import requests

with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Prediction

**Via Dashboard:**
1. Go to **Prediction** → **Batch Prediction**
2. Upload multiple audio files
3. Click **"Analyze All Files"**
4. Download results as CSV

**Via Command Line:**
```bash
python tools/run_batch_local.py --folder data/validation --limit 10
```

### Viewing Visualizations

Navigate to **Visualizations** to view:
- Training history (accuracy, loss)
- Confusion matrix
- ROC curve
- Class distribution
- Frequency band analysis
- Temporal and spectral features

### Monitoring Dashboard

Access **Monitoring** page to view:
- Model performance metrics
- Prediction history
- System uptime
- API health status
- Prediction distribution analytics

---

## Retraining Workflow

### Upload New Training Data

**Via Dashboard:**
1. Navigate to **Upload & Retrain** page
2. Select target class (**normal** or **abnormal**)
3. Upload audio files (.wav, .mp3, .flac)
4. Click **"Upload Training Data"**
5. Files saved to `data/uploads/<class>/`

### Trigger Retraining

**Method 1: Via Dashboard**
1. Go to **Upload & Retrain**
2. Click **"Start Retraining"**
3. Monitor training status in real-time
4. New model automatically is saved

**Method 2: Via Command Line (Recommended)**
```bash
# Prepare uploads and launch training
python tools/retrain_from_uploads.py \
  --uploads-dir data/uploads \
  --target-data data \
  --validation-split 0.2

# Or train from scratch
python -m src.train data --epochs 20 --batch-size 32
```

**Method 3: Quick Training Script**
```bash
python train_lightweight.py
```

### Retraining Process

1. **Data Collection**: Aggregates uploaded files from `data/uploads/`
2. **Preprocessing**: Applies same preprocessing pipeline as training data
3. **Model Loading**: Loads existing model as pre-trained base
4. **Fine-tuning**: Trains on combined data with lower learning rate
5. **Evaluation**: Validates on held-out test set
6. **Model Update**: Replaces old model if performance improves
7. **Metrics Logging**: Updates `monitoring/metrics.json` and visualizations

---

## Load Testing

### Setup Locust

```bash
pip install locust
```

### Run Load Test

**Headless Mode (Automated):**
```bash
locust -f tests/locustfile.py \
  --headless \
  --users 100 \
  --spawn-rate 10 \
  --run-time 2m \
  --host http://localhost:8000
```

**Interactive Mode:**
```bash
locust -f tests/locustfile.py --host http://localhost:8000
```
Open **http://localhost:8089** to configure test parameters

### Load Test Results

#### Single Container Performance

| Users | RPS | Avg Latency | P95 Latency | Failure Rate |
|-------|-----|-------------|-------------|--------------|
| 10 | 8.5 | 120ms | 180ms | 0% |
| 50 | 35 | 450ms | 780ms | 0% |
| 100 | 55 | 1200ms | 2100ms | 2% |

#### Scaled Deployment (3 Containers)

| Users | RPS | Avg Latency | P95 Latency | Failure Rate |
|-------|-----|-------------|-------------|--------------|
| 10 | 9.2 | 110ms | 160ms | 0% |
| 50 | 42 | 280ms | 520ms | 0% |
| 100 | 78 | 650ms | 1100ms | 0% |

**Observations:**
- 42% throughput improvement with 3 containers
- 46% latency reduction under high load
- Zero failures with proper scaling

### Test Scenarios

The Locust script tests:
- Health check endpoint (`GET /health`)
- Single prediction (`POST /predict`)
- Batch prediction (`POST /batch-predict`)
- Metrics retrieval (`GET /metrics`)
- Prediction history (`GET /visualizations/prediction-history`)

---

## Deployment

### Streamlit Community Cloud (Easiest)

1. Push repository to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub repository
4. Configure to run `frontend/app.py`
5. Deploy automatically on push

### Docker Deployment

**Build Image:**
```bash
docker build -t heartbeat-ai:latest .
```

**Run Container:**
```bash
docker run -p 8501:8501 -p 8000:8000 heartbeat-ai:latest
```

**Docker Compose:**
```bash
docker-compose up --build
```

**Scale API:**
```bash
docker-compose up --scale api=3
```

### Manual Deployment

**Production Server:**
```bash
# Start API with Gunicorn
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Start Streamlit
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Project Structure

```
CARDIAC-SOUND-CLASSIFICATION/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Multi-container setup
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core package
│   ├── __init__.py
│   ├── api.py                  # FastAPI backend
│   ├── config.py               # Configuration settings
│   ├── model.py                # Model architectures
│   ├── preprocessing.py        # Audio preprocessing
│   ├── prediction.py           # Inference engine
│   ├── train.py                # Training pipeline
│   └── data_loader.py          # Data loading utilities
│
├── frontend/                    # User Interface
│   ├── app.py                  # Streamlit dashboard
│   └── streamlit_app.py        # Alternative entry point
│
├── data/                        # Dataset (excluded from git)
│   ├── train/
│   │   └── training/           # PhysioNet subsets (a-f)
│   ├── validation/             # Validation set
│   └── uploads/                # User-uploaded data
│       ├── normal/
│       └── abnormal/
│
├── models/                      # Trained models
│   ├── cardiac_cnn_model.h5    # Main model (2.5 MB)
│   └── model_metadata.json     # Model configuration
│
├── outputs/                     # Visualizations & Reports
│   ├── class_distribution.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── metrics_summary.csv
│
├── monitoring/                  # Logs & Metrics
│   ├── logs/
│   │   ├── api.log
│   │   └── streamlit.log
│   └── metrics.json
│
├── notebook/                    # Jupyter Notebooks
│   └── heartbeat_ai_eda.ipynb # EDA & Training notebook
│
├── tests/                       # Testing Suite
│   ├── locustfile.py           # Load testing
│   ├── test_system.py          # Integration tests
│   └── test_preprocessing.py   # Unit tests
│
└── tools/                       # Utility Scripts
    ├── generate_sample_data.py
    ├── run_batch_local.py
    ├── retrain_from_uploads.py
    └── record_demo_instructions.md
```

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/cardiac-sound-classification.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code quality
flake8 src/
black src/
```

---

## Acknowledgments

- **Dataset**: [PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/)
- **Libraries**: TensorFlow, Keras, Librosa, FastAPI, Streamlit, and the open-source community

---
