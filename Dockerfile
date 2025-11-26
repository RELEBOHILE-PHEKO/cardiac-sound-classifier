# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by Librosa & TensorFlow
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
