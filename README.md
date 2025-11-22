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

##  Architecture

```
cardiac-sound-classifier/
├── src/
│   ├── api.py              # FastAPI backend
│   ├── config.py           # Configuration settings
│   └── train.py            # Model training script
├── frontend/
│   └── app.py              # Streamlit dashboard
├── models/
│   └── cardiac_cnn_model.h5  # Trained CNN model
├── data/
│   ├── train/              # Training data
│   └── test/               # Test data
└── monitoring/
    └── metrics.json        # Model performance metrics
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd cardiac-sound-classifier
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model file exists**
```bash
# Ensure cardiac_cnn_model.h5 is in the models/ directory
```

### Running the Application

#### Start the API Server
```bash
# Windows PowerShell
.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Linux/Mac
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

#### Start the Dashboard (in a new terminal)
```bash
# Windows PowerShell
.venv\Scripts\streamlit run frontend/app.py

# Linux/Mac
streamlit run frontend/app.py
```

Access the dashboard at: **http://localhost:8501**

## 📊 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Mel-spectrogram representations of cardiac audio (128 mel bands, 5-second clips)
- **Output**: Binary classification (Normal vs Abnormal)
- **Sample Rate**: 4000 Hz
- **Performance Metrics**:
  - Accuracy: 85.5%
  - Precision: 64.9%
  - Recall: 63.9%
  - F1 Score: 64.4%
  - AUC-ROC: 0.900

## 🔌 API Endpoints

### Health & Status
- `GET /health` - Check API and model status
- `GET /uptime` - Get server uptime information

### Predictions
- `POST /predict` - Classify a single audio file
- `POST /batch-predict` - Classify multiple audio files

### Monitoring
- `GET /metrics` - Get model performance metrics
- `GET /visualizations/prediction-history` - Get recent predictions
- `GET /visualizations/class-distribution` - Get training data distribution

### Training
- `GET /training-status` - Check training status
- `POST /retrain` - Trigger model retraining
- `POST /upload-training-data` - Upload new training data

## API Usage Examples

### Single Prediction
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("heartbeat.wav", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Batch Prediction
```python
import requests

url = "http://localhost:8000/batch-predict"
files = [
    ("files", open("heartbeat1.wav", "rb")),
    ("files", open("heartbeat2.wav", "rb"))
]
response = requests.post(url, files=files)
print(response.json())
```

##  Training Your Own Model

1. **Prepare your data**
   - Place audio files in `data/train/normal/` and `data/train/abnormal/`
   - Supported formats: WAV, MP3, FLAC

2. **Run training**
```bash
python src/train.py
```

3. **Model will be saved to**
```
models/cardiac_cnn_model.h5
```

##  Deployment

### Deploy API (Render/Railway/Fly.io)

**requirements.txt** should include:
```
fastapi
uvicorn
tensorflow
librosa
numpy
python-multipart
```

**Start command**:
```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

### Deploy Dashboard (Streamlit Community Cloud)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `frontend/app.py` as the main file
5. Add API URL in Streamlit secrets:
```toml
API_BASE_URL = "https://your-api-url.com"
```

##  Configuration

Edit `src/config.py` to customize:
- Model directory path
- Data directory paths
- Training parameters
- API settings

## Dependencies

### Core
- FastAPI - Web API framework
- Streamlit - Dashboard interface
- TensorFlow - Deep learning framework
- Librosa - Audio processing
- NumPy - Numerical computations

### Full list
See `requirements.txt`

##  Testing

```bash
# Test API health
curl http://localhost:8000/health

# Test prediction
curl -X POST -F "file=@test_audio.wav" http://localhost:8000/predict
```

##  Monitoring

The dashboard provides:
- Real-time API status
- Model load status
- Server uptime
- Prediction history (last 100 predictions)
- Performance metrics
- Class distribution visualization



- Training data: PhysioNet2016



## 🔮 Future Enhancements

- [ ] Multi-class classification (additional heart conditions)
- [ ] Real-time audio streaming support
- [ ] Mobile app integration
- [ ] Enhanced visualization tools
- [ ] Model explainability features
- [ ] Integration with EHR systems

---

**Note**: This system is intended for research and educational purposes. Always consult qualified healthcare professionals for medical diagnoses.
