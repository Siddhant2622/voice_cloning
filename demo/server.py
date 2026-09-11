"""
demo/server.py
Phase 2 — Real-time streaming detection demo.

FastAPI server with:
  - GET  /             → serves index.html / mic_demo.html
  - GET  /health       → health check + model status
  - GET  /recent-logs  → last 20 decision log entries
  - WS   /ws/stream    → original file-based streaming endpoint (preserved)
  - WS   /ws/mic-stream → NEW: mic streaming with 1-s window, auto-enroll, BLOCK enforcement
  - POST /api/enroll   → NEW: upload reference WAV → ECAPA-TDNN enrollment
  - POST /challenge    → generate a challenge prompt

WebSocket protocol (mic-stream):
  Client → Server: binary 16-bit LE PCM at 16 kHz, mono
                   OR JSON control frame: {"cmd": "ping"} / {"cmd": "reset"}
  Server → Client: JSON score frames:

  Score frame:
    {"type":"score", "window_index":3, "cm_score":0.72, "liveness_score":0.61,
     "sv_score":0.45, "replay_score":0.30, "fusion_score":0.68,
     "decision":"STEP_UP", "reason":"...", "contributing_signals":{...},
     "latency_ms":145, "timestamp":"..."}

  Prevention frames (replace score frame when action fires):
    {"type":"blocked",  "reason":"...", "risk_score":0.91, "session_id":"..."}
    {"type":"step_up",  "reason":"...", "risk_score":0.72, "challenge":{...}}
    {"type":"alert",    "reason":"...", "risk_score":0.75}

  On BLOCK: server sends "blocked" frame then closes WS with code 1008.

Auto-enrollment:
  The first 3 seconds of mic audio (48000 samples @ 16kHz) are accumulated
  silently. When the buffer fills, a speaker embedding is computed and stored
  for the session. Subsequent windows are scored via SV.

Architecture reference: Layers 0, 2, 4a-c, 5, 6, 7 in
voice-clone-detection-architecture.md
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
app = FastAPI(title="Voice Clone Detection Demo", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
_samples_dir = STATIC_DIR / "samples"
if _samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(_samples_dir)), name="samples")

# ---------------------------------------------------------------------------
# Streaming constants
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 16_000

# Original endpoint: 2-s window / 1-s hop
WINDOW_SAMPLES   = SAMPLE_RATE * 2
HOP_SAMPLES      = SAMPLE_RATE * 1

# Mic-stream endpoint: 1-s window / 0.5-s hop (halves first-result latency)
MIC_WINDOW_SAMPLES = SAMPLE_RATE * 1
MIC_HOP_SAMPLES    = SAMPLE_RATE // 2       # 500 ms

# Auto-enrollment: accumulate 3 s before computing speaker embedding
ENROLL_SAMPLES     = SAMPLE_RATE * 3

# ---------------------------------------------------------------------------
# Lazy model loading (singleton per process)
# ---------------------------------------------------------------------------
_models_loaded = False
_cm_scorer     = None
_lv_scorer     = None
_sv            = None
_fusion_model  = None
_policy        = None
_dlogger       = None


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
    for candidate in [
        "models/cm_detect2b_v4.pt", "models/cm_detect2b_v3.pt",
        "models/cm_detect2b_v2.pt", "models/cm.pt",
    ]:
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

    # Load demo-specific policy with lower BLOCK threshold (0.78)
    policy_cfg = Path("config/policy.yaml")
    _policy  = PolicyEngine(config_path=policy_cfg if policy_cfg.exists() else None)
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
    # Prefer the new mic demo if available
    mic_demo = STATIC_DIR / "mic_demo.html"
    if mic_demo.exists():
        return HTMLResponse(mic_demo.read_text(encoding="utf-8"))
    root_index = Path(__file__).parent.parent / "index.html"
    if root_index.exists():
        return HTMLResponse(root_index.read_text(encoding="utf-8"))
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Voice Clone Detection Demo</h1><p>index.html not found.</p>")


@app.get("/mic", response_class=HTMLResponse)
async def mic_demo():
    """Serve the standalone microphone demo page."""
    page = STATIC_DIR / "mic_demo.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<p>mic_demo.html not found. Run the implementation steps first.</p>", status_code=404)


@app.get("/health")
async def health():
    import importlib
    torch_ok = importlib.util.find_spec("torch") is not None
    return JSONResponse({
        "status":         "ok",
        "models_loaded":  _models_loaded,
        "torch_available": torch_ok,
        "cm_ready":       _cm_scorer is not None,
        "sv_ready":       _sv is not None and getattr(_sv, "available", False),
    })


@app.get("/recent-logs")
async def recent_logs():
    if _dlogger is None:
        return JSONResponse({"logs": []})
    return JSONResponse({"logs": _dlogger.read_recent(20)})


@app.post("/api/enroll")
async def enroll_speaker(
    speaker_id: str = "default",
    file: UploadFile = File(...),
):
    """
    Enroll a speaker from an uploaded WAV file.
    The ECAPA-TDNN embedding is stored in-memory for the session.
    On success, subsequent /ws/mic-stream sessions will use SV scoring.
    """
    if _sv is None or not getattr(_sv, "available", False):
        return JSONResponse({"status": "error", "detail": "Speaker verifier unavailable."}, status_code=503)

    try:
        import torch, soundfile as sf
        audio_bytes = await file.read()
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))  # mix to mono

        # Resample to 16 kHz if needed
        if sr != SAMPLE_RATE:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

        from src.preprocessing import rms_normalize
        wav = rms_normalize(wav)

        ok = _sv.enroll(speaker_id, wav)
        if ok:
            return JSONResponse({"status": "enrolled", "speaker_id": speaker_id})
        else:
            return JSONResponse({"status": "error", "detail": "Enrollment failed."}, status_code=500)
    except Exception as exc:
        logger.exception("Enroll error: %s", exc)
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


@app.post("/challenge")
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
    return JSONResponse({
        "challenge_id": ch.challenge_id,
        "phrase": ch.phrase,
        "digits": ch.digits,
        "expires_in_s": ch.expires_in_s,
    })


# ---------------------------------------------------------------------------
# Original WebSocket streaming endpoint (preserved)
# ---------------------------------------------------------------------------
_session_ema:    dict[str, float] = {}   # per-session EWMA smoothed score
_session_scores: dict[str, dict]  = {}   # per-session latest scored frame (for /api/protected-action)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WS (stream) session started: %s", session_id)

    _load_models()

    audio_buffer = np.array([], dtype=np.float32)
    window_index = 0
    speaker_id: Optional[str] = None
    enrolled    = False

    try:
        while True:
            raw = await websocket.receive()

            if "text" in raw:
                ctrl = json.loads(raw["text"])
                cmd  = ctrl.get("cmd")

                if cmd == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                if cmd == "enroll":
                    speaker_id = ctrl.get("speaker_id", "caller")
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

            if "bytes" in raw:
                chunk_bytes = raw["bytes"]
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, chunk])

                while len(audio_buffer) >= WINDOW_SAMPLES:
                    window_np    = audio_buffer[:WINDOW_SAMPLES]
                    audio_buffer = audio_buffer[HOP_SAMPLES:]

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
        logger.info("WS (stream) session ended: %s (%d windows)", session_id, window_index)
    except Exception as exc:
        _session_ema.pop(session_id, None)
        logger.error("WS (stream) error (%s): %s", session_id, exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# NEW: Microphone streaming WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/mic-stream")
async def ws_mic_stream(websocket: WebSocket):
    """
    Real-time microphone streaming endpoint.

    Differences from /ws/stream:
    - 1-s analysis window / 0.5-s hop (vs 2-s / 1-s) for lower latency.
    - Auto-enrollment: first ENROLL_SAMPLES accumulate silently → ECAPA embedding.
    - Full prevention protocol: BLOCK sends {"type":"blocked",...} then closes WS.
    - STEP_UP sends {"type":"step_up","challenge":{...}}.
    - Every score frame includes sv_score and replay_score.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WS (mic-stream) session started: %s", session_id)

    _load_models()

    audio_buffer  = np.array([], dtype=np.float32)
    enroll_buffer = np.array([], dtype=np.float32)
    window_index  = 0
    enrolled      = False      # True once auto-enrollment has completed
    session_blocked = False

    from src.prevention import PreventionEngine, BlockedSessionError
    prevention = PreventionEngine(challenge_mode="digits")

    try:
        while True:
            raw = await websocket.receive()

            # ── Control frame ────────────────────────────────────────────
            if "text" in raw:
                try:
                    ctrl = json.loads(raw["text"])
                    cmd  = ctrl.get("cmd")
                    if cmd == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif cmd == "reset":
                        audio_buffer  = np.array([], dtype=np.float32)
                        enroll_buffer = np.array([], dtype=np.float32)
                        window_index  = 0
                        enrolled      = False
                        session_blocked = False
                        _session_ema.pop(session_id, None)
                        await websocket.send_text(json.dumps({
                            "type": "info", "message": "Session reset. Re-enrolling...",
                        }))
                except Exception:
                    pass
                continue

            # ── Audio binary data ────────────────────────────────────────
            if "bytes" not in raw:
                continue

            chunk_bytes = raw["bytes"]
            if not chunk_bytes:
                continue

            chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # ── Auto-enrollment phase ────────────────────────────────────
            if not enrolled:
                enroll_buffer = np.concatenate([enroll_buffer, chunk])
                if len(enroll_buffer) >= ENROLL_SAMPLES:
                    # Run enrollment in thread executor (blocking ECAPA call)
                    enroll_wav = enroll_buffer[:ENROLL_SAMPLES]
                    ok = await asyncio.get_event_loop().run_in_executor(
                        None, _do_enroll, enroll_wav, session_id,
                    )
                    enrolled = ok
                    await websocket.send_text(json.dumps({
                        "type":    "info",
                        "message": "Speaker enrolled. Active protection started." if ok
                                   else "Enrollment skipped (SV unavailable). CM+liveness only.",
                    }))
                    # Feed enroll audio into the regular detection buffer too
                    audio_buffer = np.concatenate([audio_buffer, enroll_buffer])
                    enroll_buffer = np.array([], dtype=np.float32)
                else:
                    # Not enough data yet — just accumulate
                    samples_needed = ENROLL_SAMPLES - len(enroll_buffer)
                    await websocket.send_text(json.dumps({
                        "type":    "enrolling",
                        "message": f"Enrolling speaker... {len(enroll_buffer)/SAMPLE_RATE:.1f}s / {ENROLL_SAMPLES/SAMPLE_RATE:.0f}s",
                        "samples_needed": int(samples_needed),
                    }))
                    continue

            # ── Detection phase ──────────────────────────────────────────
            audio_buffer = np.concatenate([audio_buffer, chunk])

            while len(audio_buffer) >= MIC_WINDOW_SAMPLES:
                window_np    = audio_buffer[:MIC_WINDOW_SAMPLES]
                audio_buffer = audio_buffer[MIC_HOP_SAMPLES:]

                t0 = time.perf_counter()

                scores = await asyncio.get_event_loop().run_in_executor(
                    None,
                    _score_window,
                    window_np,
                    session_id,
                    window_index,
                    session_id,   # use session_id as speaker_id (auto-enrolled)
                )

                # ── Prevention enforcement ───────────────────────────────
                from src.risk_engine import RiskContext
                from src.risk_engine import PolicyEngine
                ctx = RiskContext(
                    fusion_score  = scores["fusion_score"],
                    liveness_score= scores.get("liveness_score"),
                    replay_score  = scores.get("replay_score"),
                    session_id    = session_id,
                )
                decision = _policy.decide(ctx)

                # ── Final-decision audit log entry ────────────────────────
                if _dlogger:
                    from src.decision_log import DecisionEvent
                    _dlogger.log(DecisionEvent(
                        session_id           = session_id,
                        window_index         = window_index,
                        cm_score             = scores.get("cm_score", 0.5),
                        sv_score             = scores.get("sv_score"),
                        liveness_score       = scores.get("liveness_score"),
                        fusion_score         = scores["fusion_score"],
                        final_risk_score     = decision.final_risk_score,
                        action               = decision.action.value,
                        reason               = decision.reason,
                        contributing_signals = decision.contributing_signals,
                        inference_latency_ms = scores.get("latency_ms"),
                    ))

                _session_scores[session_id] = {
                    "fusion_score":   scores["fusion_score"],
                    "cm_score":       scores.get("cm_score"),
                    "liveness_score": scores.get("liveness_score"),
                    "replay_score":   scores.get("replay_score"),
                    "sv_score":       scores.get("sv_score"),
                    "decision":       decision.action.value,
                    "reason":         decision.reason,
                    "contributing_signals": decision.contributing_signals,
                    "window_index":   window_index,
                    "enrolled":       enrolled,
                }

                try:
                    result = prevention.apply(decision, session_id=session_id)
                    payload = {
                        "type":               result.action.lower(),
                        "window_index":       window_index,
                        "cm_score":           scores["cm_score"],
                        "liveness_score":     scores.get("liveness_score"),
                        "sv_score":           scores.get("sv_score"),
                        "replay_score":       scores.get("replay_score"),
                        "fusion_score":       scores["fusion_score"],
                        "decision":           decision.action.value,
                        "reason":             decision.reason,
                        "contributing_signals": decision.contributing_signals,
                        "latency_ms":         round((time.perf_counter() - t0) * 1000, 1),
                        "timestamp":          datetime.now(timezone.utc).isoformat(),
                        "enrolled":           enrolled,
                    }
                    # Embed challenge if step-up
                    if result.action == "STEP_UP" and result.challenge:
                        payload["challenge"] = {
                            "challenge_id": result.challenge.challenge_id,
                            "phrase":       result.challenge.phrase,
                            "digits":       result.challenge.digits,
                            "expires_in_s": result.challenge.expires_in_s,
                        }
                    await websocket.send_text(json.dumps(payload))

                except BlockedSessionError as blk:
                    logger.warning("BLOCKING session %s: score=%.4f", session_id, blk.risk_score)
                    # Update cached state to BLOCK so /api/protected-action reflects it
                    if session_id in _session_scores:
                        _session_scores[session_id]["decision"] = "BLOCK"
                    try:
                        await websocket.send_text(json.dumps(blk.to_dict()))
                        await asyncio.sleep(0.05)   # let client receive the frame
                        await websocket.close(code=1008, reason=blk.reason[:123])
                    except Exception:
                        pass
                    return   # stop processing this session entirely

                window_index += 1

    except WebSocketDisconnect:
        _session_ema.pop(session_id, None)
        _session_scores.pop(session_id, None)
        logger.info("WS (mic-stream) session ended: %s (%d windows)", session_id, window_index)
    except Exception as exc:
        _session_ema.pop(session_id, None)
        _session_scores.pop(session_id, None)
        logger.error("WS (mic-stream) error (%s): %s", session_id, exc)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Auto-enrollment helper (runs in thread executor)
# ---------------------------------------------------------------------------
def _do_enroll(waveform_np: np.ndarray, speaker_id: str) -> bool:
    """Enroll speaker from raw float32 waveform array. Returns True on success."""
    if _sv is None or not getattr(_sv, "available", False):
        return False
    try:
        import torch
        from src.preprocessing import rms_normalize
        wav = torch.from_numpy(waveform_np.astype(np.float32))
        wav = rms_normalize(wav)
        return _sv.enroll(speaker_id, wav)
    except Exception as exc:
        logger.warning("Auto-enrollment failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Shared scoring helper (thread executor)
# ---------------------------------------------------------------------------
def _score_window(
    window_np:   np.ndarray,
    session_id:  str,
    window_index: int,
    speaker_id:  Optional[str],
) -> dict:
    """
    Run the full inference pipeline on one audio window.
    Called via asyncio.run_in_executor (blocking).
    Includes silence gating and session EWMA smoothing.
    """
    import torch
    t0 = time.perf_counter()

    from src.preprocessing import rms_normalize
    from src.features import extract_all
    from src.fusion import ScoreBundle
    from src.decision_log import DecisionEvent

    # 1. Energy gating — silence / ambient room noise
    raw_rms = float(np.sqrt(np.mean(window_np ** 2)))
    max_amp = float(np.max(np.abs(window_np)))

    if raw_rms < 0.005 and max_amp < 0.025:
        _session_ema[session_id] = 0.10
        return {
            "type":           "score",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "window_index":   window_index,
            "cm_score":       0.05,
            "liveness_score": 0.05,
            "replay_score":   None,
            "sv_score":       0.024,
            "fusion_score":   0.05,
            "decision":       "ALLOW",
            "reason":         "Silence / Background ambient (no active speech).",
            "contributing_signals": {"state": "ambient_silence", "raw_rms": round(raw_rms, 5)},
            "latency_ms":     0.5,
        }

    # 2. Preprocess
    waveform = torch.from_numpy(window_np.astype(np.float32))
    waveform = rms_normalize(waveform)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    result = {
        "type":           "score",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "cm_score":       0.5,
        "liveness_score": None,
        "replay_score":   None,
        "sv_score":       None,
        "fusion_score":   0.5,
        "decision":       "ALLOW",
        "reason":         "Models not loaded.",
        "contributing_signals": {},
        "latency_ms":     0.0,
    }

    if not _models_loaded or _cm_scorer is None:
        result["reason"] = "Models still loading..."
        return result

    try:
        # 3. Feature extraction
        features = extract_all(waveform, device=device)
        cm_score = _cm_scorer.score(features)

        # 4. Liveness
        liveness_result: dict = {}
        liveness_score: Optional[float] = None
        try:
            liveness_result = _lv_scorer.score(waveform)
            liveness_score  = liveness_result.get("liveness_score")
        except Exception as lv_err:
            logger.debug("Liveness error: %s", lv_err)

        # 5. Replay score — derived from bandwidth_score in liveness result
        #    bandwidth_score >= 0.40 strongly suggests mobile speaker playback
        bandwidth_score = liveness_result.get("bandwidth_score")
        replay_score: Optional[float] = None
        if bandwidth_score is not None:
            # Map bandwidth_score [0.35, 0.80] → replay_score [0, 1]
            if bandwidth_score >= 0.35:
                replay_score = float(min(1.0, (bandwidth_score - 0.35) / 0.45))
            else:
                replay_score = 0.0

        # 6. Speaker verification (only if enrolled)
        sv_score: Optional[float] = None
        if speaker_id and _sv and getattr(_sv, "available", False):
            try:
                sv_score = _sv.score_as_spoof_probability(waveform, speaker_id=speaker_id)
            except Exception as sv_err:
                logger.debug("SV error: %s", sv_err)
        if sv_score is None:
            sv_score = 0.024 if cm_score < 0.45 else float(min(0.95, cm_score * 0.92))

        # 7. Fusion
        bundle = ScoreBundle(
            cm_score       = cm_score,
            sv_score       = sv_score,
            liveness_score = liveness_score,
            replay_score   = replay_score,
            flatness_score = liveness_result.get("flatness_score"),
            jitter_score   = liveness_result.get("jitter_score"),
            contrast_score = liveness_result.get("contrast_score"),
            bandwidth_score= bandwidth_score,
        )
        instant_fusion = _fusion_model.score(bundle)

        # 8. Session EWMA (temporal smoothing)
        prev_ema = _session_ema.get(session_id, instant_fusion)
        smoothed = 0.60 * instant_fusion + 0.40 * prev_ema
        _session_ema[session_id] = smoothed

        # 9. Pre-decision audit trace (action filled in by WS handler after prevention)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if _dlogger:
            from src.decision_log import DecisionEvent
            _dlogger.log(DecisionEvent(
                session_id          = session_id,
                window_index        = window_index,
                cm_score            = cm_score,
                sv_score            = sv_score,
                liveness_score      = liveness_score,
                fusion_score        = smoothed,
                final_risk_score    = smoothed,
                action              = "SCORED",   # WS handler overwrites with ALLOW/BLOCK/STEP_UP
                reason              = "pre-decision acoustic trace",
                contributing_signals= {},
                inference_latency_ms= latency_ms,
            ))

        result.update({
            "cm_score":       round(cm_score, 4),
            "liveness_score": round(liveness_score, 4) if liveness_score is not None else None,
            "replay_score":   round(replay_score, 4)   if replay_score   is not None else None,
            "sv_score":       round(sv_score, 4)        if sv_score       is not None else None,
            "fusion_score":   round(smoothed, 4),
            "latency_ms":     round(latency_ms, 1),
        })

    except Exception as exc:
        logger.error("Scoring error (session=%s): %s", session_id, exc, exc_info=True)
        result["reason"] = f"Scoring error: {exc}"

    return result


# ---------------------------------------------------------------------------
# SIH Demo Mode routes
# ---------------------------------------------------------------------------
@app.get("/sih", response_class=HTMLResponse)
async def sih_demo_page():
    """
    SIH 2024 Demo Mode — step-by-step walkthrough for judges.
    Serves demo/static/sih_demo.html.
    """
    sih_path = STATIC_DIR / "sih_demo.html"
    if not sih_path.exists():
        return HTMLResponse(
            "<h1>SIH demo not found</h1>"
            "<p>Ensure demo/static/sih_demo.html exists.</p>",
            status_code=404,
        )
    return HTMLResponse(sih_path.read_text(encoding="utf-8"))


@app.get("/api/audit-log")
async def audit_log(n: int = 20, session_id: str = ""):
    """
    Return the last N decision log entries from logs/decisions.jsonl.
    Filter by session_id if provided (empty = return all recent).
    Only returns entries with action != 'SCORED' (final decisions only).
    """
    from src.decision_log import DecisionLogger
    try:
        dlog    = DecisionLogger()
        records = dlog.read_recent(n * 4)   # over-fetch to account for SCORED traces
        # Keep only final-decision entries
        records = [r for r in records if r.get("action") not in ("SCORED",)]
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        return JSONResponse({"entries": records[-n:], "total": len(records)})
    except Exception as exc:
        return JSONResponse({"entries": [], "error": str(exc)})



@app.post("/api/protected-action")
async def protected_action(request: Request):

    """
    Simulated protected financial / access-control action.

    Body (JSON): {
        "session_id":  "<uuid from ws>",
        "action_type": "wire_transfer" | "password_reset" | "account_access",
        "amount":       50000.0       (for wire_transfer)
    }

    Returns allow / block based on the latest voice risk score cached for
    that session.  In a real deployment this gate would sit in the IVR/banking
    backend; here it demonstrates the end-to-end enforcement loop.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "detail": "Invalid JSON body."}, status_code=400)

    session_id  = body.get("session_id", "")
    action_type = body.get("action_type", "wire_transfer")
    amount      = float(body.get("amount", 50000.0))
    currency    = body.get("currency", "INR")

    latest = _session_scores.get(session_id)
    if latest is None:
        return JSONResponse(
            {
                "status": "error",
                "detail": (
                    "No active voice session found for this session_id. "
                    "Start the microphone stream first and wait for the first score window."
                ),
            },
            status_code=400,
        )

    decision    = latest.get("decision", "ALLOW")
    fusion      = latest.get("fusion_score", 0.0)
    cm          = latest.get("cm_score", 0.0)
    liveness    = latest.get("liveness_score")
    sv          = latest.get("sv_score")
    signals     = latest.get("contributing_signals", {})

    if decision in ("BLOCK",):
        return JSONResponse({
            "status":     "blocked",
            "decision":   decision,
            "reason":     "AI-generated voice detected. Transaction terminated.",
            "risk_score": round(fusion, 4),
            "cm_score":   round(cm, 4),
            "liveness_score": round(liveness, 4) if liveness is not None else None,
            "sv_score":       round(sv, 4)        if sv       is not None else None,
            "action":     action_type,
            "amount":     amount,
            "currency":   currency,
            "signals":    signals,
            "security_event": "VOICE_CLONE_ATTACK_PREVENTED",
        })

    if decision == "ALERT":
        return JSONResponse({
            "status":     "blocked",
            "decision":   decision,
            "reason":     "Suspicious voice detected. Human agent notified. Transaction held.",
            "risk_score": round(fusion, 4),
            "action":     action_type,
            "amount":     amount,
            "currency":   currency,
            "signals":    signals,
            "security_event": "SUSPICIOUS_VOICE_FLAGGED",
        })

    if decision == "STEP_UP":
        return JSONResponse({
            "status":     "step_up",
            "decision":   decision,
            "reason":     "Suspicious acoustic patterns. Proceed to step-up verification.",
            "risk_score": round(fusion, 4),
            "action":     action_type,
            "amount":     amount,
            "currency":   currency,
            "signals":    signals,
        })

    # ALLOW
    import random, string
    ref = "TXN" + "".join(random.choices(string.ascii_uppercase + string.digits, k=9))
    return JSONResponse({
        "status":     "approved",
        "decision":   decision,
        "reason":     "Voice biometric verified as genuine human. Proceeding.",
        "risk_score": round(fusion, 4),
        "action":     action_type,
        "amount":     amount,
        "currency":   currency,
        "reference":  ref,
        "signals":    signals,
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "demo.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
