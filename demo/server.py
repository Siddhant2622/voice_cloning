"""
demo/server.py
Phase 2 — Real-time streaming detection demo.

FastAPI server with:
  - GET  /             → serves index.html
  - GET  /health       → health check + model status
  - GET  /recent-logs  → last 20 decision log entries
  - WS   /ws/stream    → accepts chunked 16-bit PCM audio, streams back scores
  - POST /challenge    → generate a challenge prompt
  - POST /challenge/verify → verify a challenge response

WebSocket protocol:
  Client → Server: JSON control frame OR raw binary PCM (16-bit, 16 kHz)
  Server → Client: JSON score update every ~1 s

Score update schema:
  {
    "type": "score",
    "window_index": 3,
    "cm_score": 0.72,
    "liveness_score": 0.61,
    "sv_score": null,
    "fusion_score": 0.68,
    "decision": "STEP_UP",
    "reason": "...",
    "contributing_signals": {...},
    "latency_ms": 145.2,
    "timestamp": "2024-01-01T12:00:00Z"
  }

Architecture reference: Layers 0, 2, 4a, 4c, 5, 6, 7 in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("demo.server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Voice Clone Detection Demo", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=STATIC_DIR / "samples"), name="samples")

# ---------------------------------------------------------------------------
# Lazy model loading (singleton per process)
# ---------------------------------------------------------------------------
_models_loaded = False
_cm_scorer   = None
_lv_scorer   = None
_sv          = None
_fusion_model = None
_policy      = None
_dlogger     = None

WINDOW_SAMPLES   = 16_000 * 2    # 2-second analysis window
HOP_SAMPLES      = 16_000 * 1    # 1-second hop (50% overlap)
SAMPLE_RATE      = 16_000


def _load_models():
    global _models_loaded, _cm_scorer, _lv_scorer, _sv, _fusion_model, _policy, _dlogger
    if _models_loaded:
        return

    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed — models unavailable.")
        _models_loaded = True
        return

    from src.models.cm_classifier import CMScorerWrapper
    from src.models.liveness import LivenessScorer
    from src.models.speaker_verify import SpeakerVerifier
    from src.fusion import FusionModel
    from src.risk_engine import PolicyEngine
    from src.decision_log import DecisionLogger

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading models on %s...", device)

    cm_ckpt = None
    for candidate in ["models/cm_detect2b_v3.pt", "models/cm_detect2b_v2.pt", "models/cm.pt"]:
        if Path(candidate).exists():
            cm_ckpt = candidate
            break
    _cm_scorer    = CMScorerWrapper(checkpoint_path=cm_ckpt, device=device)
    _lv_scorer    = LivenessScorer()
    _sv           = SpeakerVerifier(device=device)
    _fusion_model = FusionModel()
    for fp in [Path("models/fusion.joblib"), Path("models/fusion.pkl")]:
        if fp.exists():
            try:
                _fusion_model = FusionModel.load(fp)
                logger.info("Loaded trained fusion model: %s", fp)
                break
            except Exception as e:
                logger.warning("Could not load %s: %s", fp, e)
    _policy  = PolicyEngine(config_path="config/policy.yaml")
    _dlogger = DecisionLogger()
    _models_loaded = True
    logger.info("Models ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models)


@app.get("/", response_class=HTMLResponse)
async def root():
    root_index = Path(__file__).parent.parent / "index.html"
    if root_index.exists():
        return HTMLResponse(root_index.read_text(encoding="utf-8"))
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Voice Clone Detection Demo</h1><p>index.html not found.</p>")


@app.get("/health")
async def health():
    import importlib
    torch_ok = importlib.util.find_spec("torch") is not None
    return JSONResponse({
        "status": "ok",
        "models_loaded": _models_loaded,
        "torch_available": torch_ok,
        "cm_ready": _cm_scorer is not None,
    })


@app.get("/recent-logs")
async def recent_logs():
    if _dlogger is None:
        return JSONResponse({"logs": []})
    return JSONResponse({"logs": _dlogger.read_recent(20)})


@app.post("/challenge")
async def new_challenge(request: Request):
    from src.challenge import ChallengeGenerator
    body = await request.json()
    mode = body.get("mode", "digits")
    gen  = ChallengeGenerator()
    ch   = gen.generate(mode=mode)
    return JSONResponse({
        "challenge_id": ch.challenge_id,
        "phrase": ch.phrase,
        "digits": ch.digits,
        "expires_in_s": ch.expires_in_s,
    })


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WS session started: %s", session_id)

    _load_models()

    audio_buffer = np.array([], dtype=np.float32)
    window_index = 0
    speaker_id: Optional[str] = None
    enrolled    = False

    try:
        while True:
            raw = await websocket.receive()

            # ── Control frame (JSON text) ────────────────────────────────
            if "text" in raw:
                ctrl = json.loads(raw["text"])
                cmd  = ctrl.get("cmd")

                if cmd == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                if cmd == "enroll":
                    speaker_id = ctrl.get("speaker_id", "caller")
                    # Enrollment audio must be sent in the next binary frame
                    await websocket.send_text(json.dumps({
                        "type": "info",
                        "message": f"Ready to enroll speaker '{speaker_id}'. Send reference audio.",
                    }))
                    continue

                if cmd == "reset":
                    audio_buffer = np.array([], dtype=np.float32)
                    window_index = 0
                    enrolled     = False
                    _session_ema.pop(session_id, None)
                    await websocket.send_text(json.dumps({"type": "info", "message": "Session reset."}))
                    continue

            # ── Binary audio data ────────────────────────────────────────
            if "bytes" in raw:
                chunk_bytes = raw["bytes"]
                # Expect 16-bit little-endian PCM at SAMPLE_RATE Hz
                n_samples = len(chunk_bytes) // 2
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # Process when we have a full window
                while len(audio_buffer) >= WINDOW_SAMPLES:
                    window_np  = audio_buffer[:WINDOW_SAMPLES]
                    audio_buffer = audio_buffer[HOP_SAMPLES:]   # slide by hop

                    score_update = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _score_window,
                        window_np,
                        session_id,
                        window_index,
                        speaker_id if enrolled else None,
                    )
                    score_update["window_index"] = window_index
                    await websocket.send_text(json.dumps(score_update))
                    window_index += 1

    except WebSocketDisconnect:
        _session_ema.pop(session_id, None)
        logger.info("WS session ended: %s (%d windows processed)", session_id, window_index)
    except Exception as exc:
        _session_ema.pop(session_id, None)
        logger.error("WS session error (%s): %s", session_id, exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


_session_ema: dict[str, float] = {}


def _score_window(
    window_np: np.ndarray,
    session_id: str,
    window_index: int,
    speaker_id: Optional[str],
) -> dict:
    """
    Run the full inference pipeline on one audio window.
    Called in a thread executor (not on the async event loop).
    Includes silence/energy gating and session EWMA smoothing.
    """
    import torch
    t0 = time.perf_counter()

    from src.preprocessing import rms_normalize
    from src.features import extract_all
    from src.fusion import ScoreBundle
    from src.risk_engine import RiskContext
    from src.decision_log import DecisionEvent

    # 1. Energy gating: check if window is silence / ambient room noise
    raw_rms = float(np.sqrt(np.mean(window_np ** 2)))
    max_amp = float(np.max(np.abs(window_np)))

    if raw_rms < 0.005 and max_amp < 0.025:
        # Reset or decay EMA towards genuine on silence
        _session_ema[session_id] = 0.10
        return {
            "type":         "score",
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "window_index": window_index,
            "cm_score":     0.05,
            "liveness_score": 0.05,
            "sv_score":     None,
            "fusion_score": 0.05,
            "decision":     "ALLOW",
            "reason":       "Silence / Background ambient (no active speech).",
            "contributing_signals": {"state": "ambient_silence", "raw_rms": round(raw_rms, 5)},
            "latency_ms":   0.5,
        }

    # Preprocess speech window
    waveform = torch.from_numpy(window_np)
    waveform = rms_normalize(waveform)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = {
        "type":         "score",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "cm_score":     0.5,
        "liveness_score": None,
        "sv_score":     None,
        "fusion_score": 0.5,
        "decision":     "ALLOW",
        "reason":       "Models not loaded.",
        "contributing_signals": {},
        "latency_ms":   0.0,
    }

    if not _models_loaded or _cm_scorer is None:
        result["reason"] = "Models still loading..."
        return result

    try:
        # Feature extraction
        features  = extract_all(waveform, device=device)
        cm_score  = _cm_scorer.score(features)

        liveness_result = {}
        liveness_score  = None
        try:
            liveness_result = _lv_scorer.score(waveform)
            liveness_score  = liveness_result.get("liveness_score")
        except Exception:
            pass

        sv_score = None
        if speaker_id and _sv and _sv.available:
            sv_score = _sv.score_as_spoof_probability(waveform, speaker_id=speaker_id)

        bundle = ScoreBundle(
            cm_score=cm_score,
            sv_score=sv_score,
            liveness_score=liveness_score,
            flatness_score=liveness_result.get("flatness_score"),
            jitter_score=liveness_result.get("jitter_score"),
            contrast_score=liveness_result.get("contrast_score"),
        )
        instant_fusion = _fusion_model.score(bundle)

        # 2. Session Temporal Smoothing (EWMA):
        # Prevents single-consonant / ambient acoustic false alarms
        prev_ema = _session_ema.get(session_id, instant_fusion)
        smoothed_fusion = 0.60 * instant_fusion + 0.40 * prev_ema
        _session_ema[session_id] = smoothed_fusion

        ctx = RiskContext(
            fusion_score=smoothed_fusion,
            session_id=session_id,
        )
        decision = _policy.decide(ctx)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Structured log
        _dlogger.log(DecisionEvent(
            session_id=session_id,
            window_index=window_index,
            cm_score=cm_score,
            sv_score=sv_score,
            liveness_score=liveness_score,
            fusion_score=smoothed_fusion,
            final_risk_score=decision.final_risk_score,
            action=decision.action.value,
            reason=decision.reason,
            contributing_signals=decision.contributing_signals,
            inference_latency_ms=latency_ms,
        ))

        result.update({
            "cm_score":      round(cm_score, 4),
            "liveness_score": round(liveness_score, 4) if liveness_score is not None else None,
            "sv_score":      round(sv_score, 4) if sv_score is not None else None,
            "fusion_score":  round(smoothed_fusion, 4),
            "decision":      decision.action.value,
            "reason":        decision.reason,
            "contributing_signals": decision.contributing_signals,
            "latency_ms":    round(latency_ms, 1),
        })


    except Exception as exc:
        logger.error("Scoring error: %s", exc, exc_info=True)
        result["reason"] = f"Scoring error: {exc}"

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "demo.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
