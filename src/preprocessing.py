"""
src/preprocessing.py
Layer 1 — Audio preprocessing pipeline.

Responsibilities:
  - Resample to 16 kHz mono
  - Voice Activity Detection (Silero VAD; energy-threshold fallback)
  - RMS normalisation
  - Codec-artifact awareness (documented; not actively reversed)

Architecture reference: Layer 1 in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR = 16_000          # All downstream models expect 16 kHz mono
VAD_THRESHOLD = 0.5         # Silero VAD speech probability threshold
FRAME_DURATION_MS = 30      # VAD frame size in ms (Silero supports 30/60/100)
CHUNK_SAMPLES = int(TARGET_SR * FRAME_DURATION_MS / 1000)  # 480 samples @ 16 kHz


# ---------------------------------------------------------------------------
# Lazy imports — torch is optional until installed
# ---------------------------------------------------------------------------
def _get_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for preprocessing. "
            "Install: python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
        )


def _get_torchaudio():
    try:
        import torchaudio
        return torchaudio
    except ImportError:
        raise ImportError(
            "torchaudio is required. "
            "Install: python -m pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
        )


# ---------------------------------------------------------------------------
# Silero VAD loader (cached)
# ---------------------------------------------------------------------------
_silero_vad_model = None
_silero_vad_utils = None


def _load_silero_vad():
    """
    Load Silero VAD from torch.hub (cached after first call).
    Falls back to energy-threshold VAD if torch.hub is unavailable.
    """
    global _silero_vad_model, _silero_vad_utils
    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils

    torch = _get_torch()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        _silero_vad_model = model
        _silero_vad_utils = utils
        logger.info("Silero VAD loaded from torch.hub")
        return model, utils
    except Exception as exc:
        logger.warning(
            "Could not load Silero VAD (%s). "
            "Falling back to energy-threshold VAD.",
            exc,
        )
        return None, None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> Tuple["torch.Tensor", int]:
    """
    Load an audio file and return (waveform [C, T], sample_rate).

    Tries torchaudio first; falls back to soundfile (more robust on torchaudio
    nightly builds that require torchcodec for WAV decoding).
    """
    torch = _get_torch()
    path = str(path)

    # Primary: torchaudio
    try:
        torchaudio = _get_torchaudio()
        waveform, sr = torchaudio.load(path)
        return waveform, sr
    except Exception as e_ta:
        logger.debug("torchaudio.load failed (%s); trying soundfile fallback.", e_ta)

    # Fallback: soundfile + numpy → torch
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile returns [T, C]; convert to [C, T]
        waveform = torch.from_numpy(data.T)
        return waveform, sr
    except Exception as e_sf:
        logger.debug("soundfile fallback failed (%s); trying librosa.", e_sf)

    # Last resort: librosa
    try:
        import librosa
        import numpy as np
        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        waveform = torch.from_numpy(data.astype(np.float32))
        return waveform, sr
    except Exception as e_lr:
        raise RuntimeError(
            f"Could not load audio file '{path}'. "
            f"torchaudio: {e_ta}; soundfile: {e_sf}; librosa: {e_lr}"
        )


def to_mono_16k(waveform: "torch.Tensor", sr: int) -> "torch.Tensor":
    """
    Convert to mono 16 kHz. Returns 1-D tensor of shape [T].
    """
    torch = _get_torch()
    torchaudio = _get_torchaudio()

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform

    # Resample
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    return waveform.squeeze(0)   # [T]


def rms_normalize(waveform: "torch.Tensor", target_rms: float = 0.05) -> "torch.Tensor":
    """
    RMS-normalise waveform to a fixed target level.
    Protects against silence / near-silence clips.
    """
    torch = _get_torch()
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-8:
        logger.debug("Near-silent clip detected; skipping RMS normalisation.")
        return waveform
    return waveform * (target_rms / rms)


def apply_vad(
    waveform: "torch.Tensor",
    sr: int = TARGET_SR,
    threshold: float = VAD_THRESHOLD,
    min_speech_ms: int = 250,
) -> "torch.Tensor":
    """
    Apply Voice Activity Detection; return only voiced frames concatenated.

    Args:
        waveform: 1-D tensor [T] at sr Hz.
        sr: sample rate (must be TARGET_SR for Silero).
        threshold: Silero speech-probability cutoff.
        min_speech_ms: discard speech segments shorter than this.

    Returns:
        Voiced-frames tensor [T'] (may be shorter than input).
        Falls back to energy-threshold VAD if Silero unavailable.
    """
    torch = _get_torch()
    assert sr == TARGET_SR, f"VAD requires {TARGET_SR} Hz; got {sr} Hz"

    model, utils = _load_silero_vad()

    if model is None:
        # ── Energy-threshold fallback ──────────────────────────────────────
        logger.debug("Using energy-threshold VAD fallback.")
        return _energy_vad(waveform, sr, min_speech_ms)

    # ── Silero VAD ─────────────────────────────────────────────────────────
    try:
        get_speech_timestamps, _, _, _, collect_chunks = utils
        speech_timestamps = get_speech_timestamps(
            waveform,
            model,
            threshold=threshold,
            sampling_rate=sr,
            min_speech_duration_ms=min_speech_ms,
        )
        if not speech_timestamps:
            logger.debug("VAD: no speech detected; returning original waveform.")
            return waveform

        voiced = collect_chunks(speech_timestamps, waveform)
        logger.debug(
            "VAD: kept %.1f%% of audio (%d → %d samples).",
            100 * voiced.numel() / waveform.numel(),
            waveform.numel(),
            voiced.numel(),
        )
        return voiced
    except Exception as exc:
        logger.warning("Silero VAD failed (%s); falling back to energy VAD.", exc)
        return _energy_vad(waveform, sr, min_speech_ms)


def _energy_vad(
    waveform: "torch.Tensor",
    sr: int,
    min_speech_ms: int = 250,
    frame_ms: int = 30,
    energy_threshold: float = 0.002,
) -> "torch.Tensor":
    """
    Simple energy-threshold VAD fallback.
    Keeps frames whose RMS energy > energy_threshold.
    """
    torch = _get_torch()
    frame_len = int(sr * frame_ms / 1000)
    min_frames = max(1, int(min_speech_ms / frame_ms))

    frames = waveform.unfold(0, frame_len, frame_len)          # [N_frames, frame_len]
    energies = frames.pow(2).mean(dim=1).sqrt()                # [N_frames]
    voiced_mask = energies > energy_threshold                  # [N_frames]

    # Keep runs of voiced frames ≥ min_frames long
    keep_indices = []
    run_start = None
    for i, is_voiced in enumerate(voiced_mask.tolist()):
        if is_voiced and run_start is None:
            run_start = i
        elif not is_voiced and run_start is not None:
            if i - run_start >= min_frames:
                keep_indices.extend(range(run_start, i))
            run_start = None
    if run_start is not None and len(voiced_mask) - run_start >= min_frames:
        keep_indices.extend(range(run_start, len(voiced_mask)))

    if not keep_indices:
        return waveform

    voiced_frames = frames[keep_indices].reshape(-1)
    return voiced_frames


def preprocess_file(
    path: str | Path,
    apply_vad_flag: bool = True,
) -> Tuple["torch.Tensor", int]:
    """
    Full preprocessing pipeline for a single audio file.

    Returns:
        (waveform_16k_mono_normalised [T], TARGET_SR)
    """
    waveform, sr = load_audio(path)
    waveform = to_mono_16k(waveform, sr)

    if apply_vad_flag:
        waveform = apply_vad(waveform, TARGET_SR)

    waveform = rms_normalize(waveform)
    return waveform, TARGET_SR


def preprocess_chunk(
    chunk: np.ndarray,
    sr: int,
    apply_vad_flag: bool = False,   # VAD on raw mic chunks can be noisy; caller decides
) -> "torch.Tensor":
    """
    Preprocess a raw numpy audio chunk (e.g. from microphone or WebSocket).

    Args:
        chunk: float32 numpy array [T], normalised to [-1, 1].
        sr: original sample rate of the chunk.

    Returns:
        Preprocessed waveform tensor [T'] at TARGET_SR.
    """
    torch = _get_torch()
    waveform = torch.from_numpy(chunk.astype(np.float32))

    if sr != TARGET_SR:
        torchaudio = _get_torchaudio()
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    if apply_vad_flag:
        waveform = apply_vad(waveform, TARGET_SR)

    waveform = rms_normalize(waveform)
    return waveform
