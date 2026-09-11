"""
src/mic_streamer.py
Microphone streaming client and file-playback test harness.

Two modes:
  Live mic mode:
    Captures audio from the default microphone via sounddevice, resamples
    to 16 kHz mono if needed, and streams 16-bit PCM over a WebSocket.

  File playback mode (--file):
    Reads a WAV/FLAC file and streams it at real-time speed to simulate
    a microphone — useful for automated testing without a physical mic.

Usage:
  # Live mic → local demo server
  python -m src.mic_streamer

  # File playback (AI-cloned voice test)
  python -m src.mic_streamer --file samples/spoof_tts.wav --url ws://localhost:8000/ws/mic-stream

  # File playback (genuine human test)
  python -m src.mic_streamer --file samples/genuine.wav --expect ALLOW

Dependencies:
  sounddevice   pip install sounddevice
  websockets    pip install websockets
  soundfile     pip install soundfile   (file-playback only)
  numpy         pip install numpy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("mic_streamer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR      = 16_000          # server expects 16 kHz mono
CHUNK_SAMPLES  = TARGET_SR // 4  # 250 ms per send chunk (4 Hz send rate)
DTYPE_SEND     = np.int16        # 16-bit PCM for wire format


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _to_pcm16(audio_f32: np.ndarray) -> bytes:
    """Convert float32 mono array → 16-bit little-endian PCM bytes."""
    clipped = np.clip(audio_f32, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _resample_np(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resample via numpy. Fine for test harness (not real-time)."""
    if src_sr == dst_sr:
        return audio
    try:
        import torchaudio, torch
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        resampled = torchaudio.functional.resample(wav, src_sr, dst_sr)
        return resampled.squeeze(0).numpy()
    except ImportError:
        pass
    # Fallback: scipy
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        pass
    # Last resort: numpy linear interp
    duration = len(audio) / src_sr
    out_len  = int(duration * dst_sr)
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, out_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# File-playback streamer
# ---------------------------------------------------------------------------
async def stream_file(
    filepath: str,
    ws_url:   str,
    expect:   Optional[str] = None,
) -> None:
    """
    Stream a WAV/FLAC file to the WebSocket server at real-time speed.

    Args:
        filepath: Path to the audio file.
        ws_url:   WebSocket URL (e.g. ws://localhost:8000/ws/mic-stream).
        expect:   If provided, assert final decision equals this string (ALLOW/BLOCK/...).
    """
    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile not installed. Run: pip install soundfile")
        sys.exit(1)

    try:
        import websockets
    except ImportError:
        logger.error("websockets not installed. Run: pip install websockets")
        sys.exit(1)

    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    audio    = data.mean(axis=1)  # mix to mono
    audio    = _resample_np(audio, sr, TARGET_SR)

    logger.info(
        "Streaming %s → %s (duration=%.1f s, sr=%d)",
        filepath, ws_url, len(audio) / TARGET_SR, TARGET_SR,
    )

    final_decisions: list[str] = []
    last_reason: str = ""

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        # Receive loop in background
        async def _recv():
            try:
                async for msg in ws:
                    frame = json.loads(msg)
                    ftype = frame.get("type", "")
                    if ftype == "score":
                        dec = frame.get("decision", "?")
                        final_decisions.append(dec)
                        logger.info(
                            "[win=%d] fusion=%.3f cm=%.3f lv=%s sv=%s replay=%s → %s (%.0f ms)",
                            frame.get("window_index", -1),
                            frame.get("fusion_score", 0),
                            frame.get("cm_score", 0),
                            f"{frame['liveness_score']:.3f}" if frame.get("liveness_score") is not None else "n/a",
                            f"{frame['sv_score']:.3f}"       if frame.get("sv_score")       is not None else "n/a",
                            f"{frame['replay_score']:.3f}"   if frame.get("replay_score")   is not None else "n/a",
                            dec,
                            frame.get("latency_ms", 0),
                        )
                    elif ftype == "blocked":
                        logger.warning("🚫 BLOCKED — %s (score=%.4f)", frame.get("reason"), frame.get("risk_score", 0))
                        final_decisions.append("BLOCK")
                    elif ftype == "step_up":
                        ch = frame.get("challenge", {})
                        logger.warning("⚠️  STEP_UP — challenge: %s %s", ch.get("phrase"), ch.get("digits"))
                        final_decisions.append("STEP_UP")
                    elif ftype == "alert":
                        logger.warning("🔔 ALERT — %s", frame.get("reason"))
                        final_decisions.append("ALERT")
                    elif ftype in ("enrolling", "info"):
                        logger.info("[%s] %s", ftype, frame.get("message"))
                    elif ftype == "error":
                        logger.error("[server error] %s", frame.get("message") or frame.get("detail"))
                    # pong / other frames are silently ignored
            except websockets.exceptions.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(_recv())

        # Send audio at real-time speed
        pos = 0
        chunk_size = CHUNK_SAMPLES
        while pos < len(audio):
            chunk = audio[pos : pos + chunk_size]
            pos  += chunk_size
            await ws.send(_to_pcm16(chunk))
            # Sleep to maintain real-time rate
            await asyncio.sleep(chunk_size / TARGET_SR)

        # Wait a bit for the server to process remaining buffer
        await asyncio.sleep(3.0)
        recv_task.cancel()

    # ── Assertion ─────────────────────────────────────────────────────────
    if expect:
        majority = _majority(final_decisions)
        if majority == expect.upper():
            logger.info("✅ PASS — majority decision=%s (expected %s)", majority, expect)
        else:
            logger.error("❌ FAIL — majority decision=%s (expected %s)", majority, expect)
            sys.exit(2)
    else:
        logger.info("Stream complete. Decisions observed: %s", final_decisions)


def _majority(decisions: list[str]) -> str:
    if not decisions:
        return "NONE"
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Live-mic streamer
# ---------------------------------------------------------------------------
async def stream_mic(ws_url: str) -> None:
    """
    Capture audio from the default microphone and stream to the WebSocket.
    Press Ctrl+C to stop.
    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    try:
        import websockets
    except ImportError:
        logger.error("websockets not installed. Run: pip install websockets")
        sys.exit(1)

    logger.info("Starting live mic capture → %s", ws_url)
    logger.info("Press Ctrl+C to stop.")

    mic_sr       = int(sd.query_devices(kind="input")["default_samplerate"])
    chunk_frames = int(mic_sr * 0.25)   # 250 ms chunks
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def _mic_callback(indata, frames, t, status):
        if status:
            logger.debug("Sounddevice status: %s", status)
        # indata shape: [frames, channels] — mix to mono float32
        mono = indata.mean(axis=1).astype(np.float32)
        asyncio.get_event_loop().call_soon_threadsafe(queue.put_nowait, mono.copy())

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
        # Receive messages in background
        async def _recv():
            try:
                async for msg in ws:
                    frame = json.loads(msg)
                    ftype = frame.get("type", "")
                    if ftype == "score":
                        logger.info(
                            "[win=%d] fusion=%.3f → %s (%.0f ms)",
                            frame.get("window_index", -1),
                            frame.get("fusion_score", 0),
                            frame.get("decision", "?"),
                            frame.get("latency_ms", 0),
                        )
                    elif ftype == "blocked":
                        logger.warning("🚫 BLOCKED — %s", frame.get("reason"))
                    elif ftype == "step_up":
                        ch = frame.get("challenge", {})
                        logger.warning("⚠️  STEP_UP — Please say: %s %s", ch.get("phrase", ""), ch.get("digits", ""))
                    elif ftype in ("enrolling", "info"):
                        logger.info("[%s] %s", ftype, frame.get("message"))
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server.")

        recv_task = asyncio.create_task(_recv())

        with sd.InputStream(
            samplerate = mic_sr,
            channels   = 1,
            dtype      = "float32",
            blocksize  = chunk_frames,
            callback   = _mic_callback,
        ):
            try:
                while True:
                    audio_chunk = await queue.get()
                    # Resample if mic SR ≠ 16 kHz
                    if mic_sr != TARGET_SR:
                        audio_chunk = _resample_np(audio_chunk, mic_sr, TARGET_SR)
                    await ws.send(_to_pcm16(audio_chunk))
            except asyncio.CancelledError:
                pass
            except KeyboardInterrupt:
                logger.info("Stopping mic capture.")

        recv_task.cancel()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Microphone streaming client for voice clone detection demo.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--url",
        default="ws://localhost:8000/ws/mic-stream",
        help="WebSocket server URL (default: ws://localhost:8000/ws/mic-stream)",
    )
    p.add_argument(
        "--file",
        default=None,
        help="Audio file to stream instead of live mic (WAV/FLAC/OGG).",
    )
    p.add_argument(
        "--expect",
        default=None,
        choices=["ALLOW", "BLOCK", "STEP_UP", "ALERT"],
        help="Assert that the majority decision equals this value (for CI tests).",
    )
    return p


def main():
    args = _build_parser().parse_args()
    if args.file:
        if not Path(args.file).exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        asyncio.run(stream_file(args.file, args.url, expect=args.expect))
    else:
        try:
            asyncio.run(stream_mic(args.url))
        except KeyboardInterrupt:
            logger.info("Mic streaming stopped.")


if __name__ == "__main__":
    main()
