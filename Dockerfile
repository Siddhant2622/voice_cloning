FROM python:3.11-slim

LABEL maintainer="VoiceGuard Research Prototype"
LABEL description="Real-time voice clone detection demo"

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU (stable for 3.11 in container)
# Switch to cu128 nightly build if running on GPU node
RUN pip install --no-cache-dir \
    torch==2.4.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/       ./src/
COPY demo/      ./demo/
COPY detect.py  ./detect.py
COPY evaluate.py ./evaluate.py
COPY config/    ./config/
COPY data/      ./data/

# Create directories
RUN mkdir -p models logs results

# Expose demo port
EXPOSE 8000

# Default: run the streaming demo server
CMD ["python", "-m", "uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "8000"]
