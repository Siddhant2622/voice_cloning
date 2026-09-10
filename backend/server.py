"""
api/server.py
=============
VoiceGuard Production FastAPI Application

Endpoints:
  GET  /                          → Health redirect
  GET  /api/v1/health             → Model load status + uptime
  POST /api/v1/analyze            → Analyze an uploaded audio file (REST)
  GET  /api/v1/benchmark          → Cached benchmark results (EER / datasets)
  WS   /ws/stream                 → Real-time streaming analysis
  GET  /api/v1/docs               → OpenAPI docs (auto-generated)

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
  # or:
  python api/server.py
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import struct
import sys
import time
import uuid
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import (
    Depends, FastAPI, File, HTTPException, Request,
    UploadFile, WebSocket, WebSocketDisconnect, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.auth import require_api_key
from backend.config import Settings, get_settings

logger = logging.getLogger("voiceguard.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_models: Dict[str, Any] = {}
_startup_time: float = 0.0
_model_ready: bool = False


def _load_models(settings: Settings) -> None:
    global _model_ready
    logger.info("Loading VoiceGuard models...")
    t0 = time.time()

    try:
        # Pull from HF Hub if configured and checkpoint not local
        ckpt = Path(settings.cm_checkpoint)
        if not ckpt.exists():
            for candidate in ["models/cm_detect2b_v4.pt", "models/cm_detect2b_v3.pt", "models/cm_detect2b_v2.pt", "models/cm.pt"]:
                if Path(candidate).exists():
                    ckpt = Path(candidate)
                    break
        if not ckpt.exists() and settings.hf_repo_id:
            logger.info("Pulling checkpoint from HF Hub: %s", settings.hf_repo_id)
            from data.download_datasets import pull_checkpoint_from_hub
            ckpt = pull_checkpoint_from_hub(settings.hf_repo_id, ckpt, settings.hf_token or None)

        from src.models.cm_classifier import CMScorerWrapper
        from src.models.liveness import LivenessScorer
        from src.fusion import FusionModel          # was wrongly called FusionScorer
        from src.risk_engine import PolicyEngine    # was wrongly called RiskEngine

        _models["cm"]     = CMScorerWrapper(checkpoint_path=ckpt if ckpt.exists() else None)
        _models["lv"]     = LivenessScorer()
        _models["fusion"] = FusionModel()
        _models["risk"]   = PolicyEngine()

        elapsed = time.time() - t0
        logger.info("Models loaded in %.2f s", elapsed)
        _model_ready = True
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        _model_ready = False


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    settings = get_settings()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models, settings)
    yield
    logger.info("VoiceGuard API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "VoiceGuard — Real-Time AI Voice Clone Detection API.\n\n"
            "Supports REST file-upload analysis and live WebSocket streaming.\n"
            "Built on ASVspoof 2019 LA trained CM classifier + WavLM embeddings."
        ),
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Simple request logger
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.time()
        response = await call_next(request)
        ms = (time.time() - t0) * 1000
        logger.info("%s %s → %d  (%.0f ms)", request.method, request.url.path, response.status_code, ms)
        return response

    return app


app = create_app()

ROOT_DIR = Path(__file__).parent.parent
INDEX_HTML = ROOT_DIR / "index.html"
SAMPLES_DIR = ROOT_DIR / "public" / "samples"

if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status:       str
    model_ready:  bool
    uptime_s:     float
    version:      str
    timestamp:    str


class AnalyzeResponse(BaseModel):
    filename:      str
    duration_s:    float
    cm_score:      float
    liveness_score: float
    fusion_score:  float
    decision:      str
    latency_ms:    float
    analyst:       Optional[Dict] = None
    timestamp:     str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    return RedirectResponse("/api/v1/health")


@app.get("/index.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    if INDEX_HTML.exists():
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {
        "status":      "ready" if _model_ready else "loading",
        "model_ready": _model_ready,
        "uptime_s":    round(time.time() - _startup_time, 1),
        "version":     get_settings().app_version,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze an audio file for AI voice cloning",
)
async def analyze_file(
    file: UploadFile = File(..., description="Audio file (WAV, FLAC, MP3, OGG, M4A)"),
    _auth = Depends(require_api_key),
):
    if not _model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading. Retry in a few seconds.",
        )

    ALLOWED = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED)}",
        )

    t0 = time.time()
    audio_bytes = await file.read()

    try:
        import torch
        import torchaudio
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        try:
            data, sr = sf.read(buf, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)
        except Exception:
            buf.seek(0)
            waveform, sr = torchaudio.load(buf)
        duration_s = waveform.shape[-1] / sr

        from src.preprocessing import preprocess_waveform
        wav = preprocess_waveform(waveform, sr)

        loop = asyncio.get_event_loop()
        feats   = await loop.run_in_executor(None, lambda: _score_waveform(wav))
        latency = (time.time() - t0) * 1000

        analyst_dict = None
        try:
            from src.quality_latency_analyst import VoiceQualityLatencyAnalyst
            analyst = VoiceQualityLatencyAnalyst()
            analyst_res = analyst.analyze(wav, sr=16000)
            analyst_dict = analyst_res.to_dict()
        except Exception as a_err:
            logger.debug("Analyst computation skipped: %s", a_err)

        return {
            "filename":       file.filename or "unknown",
            "duration_s":     round(duration_s, 2),
            "cm_score":       feats["cm_score"],
            "liveness_score": feats["liveness_score"],
            "fusion_score":   feats["fusion_score"],
            "decision":       feats["decision"],
            "latency_ms":     round(latency, 1),
            "analyst":        analyst_dict,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.exception("Analysis failed for %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio processing failed: {exc}",
        )


def _score_waveform(wav) -> Dict:
    """Run CM + liveness + fusion synchronously (called in executor).
    Returns global + per-frame scores (Resemble AI DETECT-2B style).
    """
    import numpy as np
    import torch
    from src.features import extract_all
    from src.fusion import ScoreBundle
    from src.risk_engine import RiskContext

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(wav, torch.Tensor):
        wav = torch.from_numpy(np.asarray(wav, dtype=np.float32))

    # Extract all features (WavLM + Wav2Vec2 + log-mel)
    feats = extract_all(wav, device=device, extract_wav2vec2=True)

    # Global + frame-level CM scoring (DETECT-2B style)
    cm_score = _models["cm"].score(feats)
    try:
        global_score, frame_scores = _models["cm"].score_with_frames(feats)
        # Identify frames with spoof probability > 0.6
        suspicious_frames = [i for i, s in enumerate(frame_scores) if s > 0.6]
    except Exception:
        global_score = cm_score
        frame_scores = []
        suspicious_frames = []

    lv_result = {}
    liveness_score = None
    try:
        lv_result = _models["lv"].score(wav)
        liveness_score = lv_result.get("liveness_score")
    except Exception as exc:
        logger.warning("Liveness analysis failed: %s", exc)

    bundle = ScoreBundle(
        cm_score=cm_score,
        sv_score=None,
        liveness_score=liveness_score,
        flatness_score=lv_result.get("flatness_score"),
        jitter_score=lv_result.get("jitter_score"),
        contrast_score=lv_result.get("contrast_score"),
        bandwidth_score=lv_result.get("bandwidth_score"),
    )
    fs = _models["fusion"].score(bundle)
    ctx = RiskContext(fusion_score=fs)
    decision = _models["risk"].decide(ctx)

    # Downsample frame scores to max 50 for network efficiency
    if len(frame_scores) > 50:
        step = len(frame_scores) // 50
        frame_scores_ds = [round(frame_scores[i], 3) for i in range(0, len(frame_scores), step)][:50]
    else:
        frame_scores_ds = [round(s, 3) for s in frame_scores]

    return {
        "cm_score":         round(cm_score, 4),
        "liveness_score":   round(liveness_score, 4) if liveness_score is not None else 0.05,
        "fusion_score":     round(fs, 4),
        "decision":         decision.action.value,
        "reason":           decision.reason,
        "frame_scores":     frame_scores_ds,
        "suspicious_frames": suspicious_frames[:20],  # max 20 suspicious frame indices
    }


@app.get("/api/v1/benchmark", tags=["Research"])
async def get_benchmark():
    """Return cached benchmark EER results for the Benchmark Explorer UI."""
    bench_path = Path(get_settings().benchmark_json)
    if not bench_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark results not yet generated. Run: python training/eval_eer.py",
        )
    with open(bench_path) as f:
        return JSONResponse(content=json.load(f))


@app.post("/api/challenge", tags=["Challenge"])
@app.post("/challenge", include_in_schema=False)
async def new_challenge(request: Request):
    from src.challenge import ChallengeGenerator
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    mode = body.get("mode", "digits") if isinstance(body, dict) else "digits"
    gen  = ChallengeGenerator()
    ch   = gen.generate(mode=mode)
    return {
        "challenge_id": ch.challenge_id,
        "phrase":       ch.phrase,
        "digits":       ch.digits,
        "expires_in_s": ch.expires_in_s,
    }


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    Real-time streaming analysis over WebSocket.

    Client sends raw PCM audio chunks (16-bit LE, 16 kHz, mono).
    Server responds with JSON score objects.
    """
    await ws.accept()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket client connected: %s (session %s)", ws.client, session_id)

    WINDOW_SAMPLES = 16_000 * 1    # 1.0-second analysis window
    HOP_SAMPLES    = 8_000         # 0.5-second hop
    audio_buffer = np.array([], dtype=np.float32)
    window = 0
    session_ema = 0.05

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
                continue

            if msg.get("type") == "websocket.disconnect":
                break

            # Handle text control frames (e.g. reset or ping)
            if "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    cmd = ctrl.get("cmd")
                    if cmd == "reset":
                        audio_buffer = np.array([], dtype=np.float32)
                        window = 0
                        session_ema = 0.05
                        await ws.send_json({"type": "info", "message": "Session reset."})
                        continue
                    elif cmd == "ping":
                        await ws.send_json({"type": "pong"})
                        continue
                except Exception:
                    pass

            data = msg.get("bytes") or b""
            if not data:
                continue

            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # Process when we have a full window
            while len(audio_buffer) >= WINDOW_SAMPLES:
                window_np = audio_buffer[:WINDOW_SAMPLES]
                audio_buffer = audio_buffer[HOP_SAMPLES:]

                t0 = time.time()
                window += 1

                # Energy gating for ambient silence
                raw_rms = float(np.sqrt(np.mean(window_np ** 2)))
                max_amp = float(np.max(np.abs(window_np)))

                if raw_rms < 0.005 and max_amp < 0.025:
                    session_ema = 0.80 * session_ema + 0.20 * 0.05
                    await ws.send_json({
                        "type":           "score",
                        "timestamp":      datetime.now(timezone.utc).isoformat(),
                        "window":         window,
                        "window_index":   window,
                        "cm_score":       0.04,
                        "liveness_score": 0.05,
                        "sv_score":       None,
                        "fusion_score":   round(session_ema, 4),
                        "decision":       "ALLOW",
                        "reason":         "Silence / ambient room acoustics.",
                        "latency_ms":     0.5,
                    })
                    continue

                try:
                    import torch
                    from src.preprocessing import rms_normalize
                    wav = torch.from_numpy(window_np)
                    wav = rms_normalize(wav)

                    loop = asyncio.get_event_loop()
                    scores = await loop.run_in_executor(None, _score_waveform, wav)

                    fusion = scores["fusion_score"]
                    session_ema = 0.65 * fusion + 0.35 * session_ema
                    latency = (time.time() - t0) * 1000.0

                    await ws.send_json({
                        "type":              "score",
                        "timestamp":         datetime.now(timezone.utc).isoformat(),
                        "window":            window,
                        "window_index":      window,
                        "fusion_score":      round(session_ema, 4),
                        "cm_score":          scores["cm_score"],
                        "liveness_score":    scores["liveness_score"],
                        "sv_score":          None,
                        "decision":          scores["decision"],
                        "reason":            scores.get("reason", ""),
                        "latency_ms":        round(latency, 1),
                        "frame_scores":      scores.get("frame_scores", []),
                        "suspicious_frames": scores.get("suspicious_frames", []),
                    })
                except Exception as exc:
                    logger.warning("Scoring error: %s", exc, exc_info=True)
                    await ws.send_json({"type": "error", "detail": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", ws.client)
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
    finally:
        logger.info("WebSocket session ended. Windows scored: %d", window)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "backend.server:app",
        host=s.host,
        port=s.port,
        reload=s.debug,
        log_level="info",
    )
