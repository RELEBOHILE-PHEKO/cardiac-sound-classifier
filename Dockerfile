FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploaded models logs

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD uvicorn src.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
