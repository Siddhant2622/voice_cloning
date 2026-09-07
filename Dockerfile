# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 libportaudio2 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU (swap index-url for cu121/cu128 on GPU node)
RUN pip install --no-cache-dir \
    torch==2.4.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    pydantic-settings>=2.0.0 \
    huggingface_hub>=0.22.0 \
    slowapi>=0.1.9 \
    httpx>=0.27.0

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="VoiceGuard — AI Voice Clone Detection"
LABEL description="Production FastAPI server for real-time AI voice detection"
LABEL version="1.0.0"

# Copy system libs from builder
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root user for security
RUN groupadd -r voiceguard && useradd -r -g voiceguard -m -d /app voiceguard

WORKDIR /app

# Copy application code
COPY --chown=voiceguard:voiceguard src/           ./src/
COPY --chown=voiceguard:voiceguard api/           ./api/
COPY --chown=voiceguard:voiceguard training/      ./training/
COPY --chown=voiceguard:voiceguard data/          ./data/
COPY --chown=voiceguard:voiceguard config/        ./config/
COPY --chown=voiceguard:voiceguard detect.py      ./detect.py
COPY --chown=voiceguard:voiceguard evaluate.py    ./evaluate.py
COPY --chown=voiceguard:voiceguard index.html     ./index.html
COPY --chown=voiceguard:voiceguard .env.example   ./.env.example

# Runtime directories (models are volume-mounted in production)
RUN mkdir -p models logs results && chown -R voiceguard:voiceguard models logs results

USER voiceguard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Production: run the FastAPI server
# Mount model checkpoint: -v /path/to/models:/app/models
# Set env vars via: --env-file .env
CMD ["python", "-m", "uvicorn", "api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
