# Use Python slim image for smaller footprint
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONMALLOC=malloc \
    MALLOC_TRIM_THRESHOLD_=100000 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with memory-efficient flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploaded models logs monitoring

# Expose only the port being used (backend OR frontend, not both)
EXPOSE 8000

# Health check for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form and single process
# IMPORTANT: Run ONLY the backend on Render (not both services)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
