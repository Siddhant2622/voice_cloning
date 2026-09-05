"""
src/features.py
Layer 3 — Feature extraction pipeline.

Extracts complementary, independently informative representations:
  (a) Log-mel spectrogram — spectral/cepstral; sensitive to vocoder artifacts
  (b) WavLM-base SSL embedding — higher-level phonetic/prosodic anomalies;
      generalises better to unseen synthesis methods than hand-crafted features

Architecture reference: Layer 3 in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEL_SAMPLE_RATE = 16_000
MEL_N_FFT       = 400      # 25 ms @ 16 kHz
MEL_HOP_LENGTH  = 160      # 10 ms @ 16 kHz
MEL_N_MELS      = 80
MEL_FMIN        = 20.0
MEL_FMAX        = 8_000.0

SSL_MODEL_ID    = "microsoft/wavlm-base"   # ~94 MB; use -base for CPU/low-VRAM compat
SSL_MAX_SECONDS = 30                       # clip to avoid OOM on long files


# ---------------------------------------------------------------------------
# Lazy module accessors
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _torchaudio():
    import torchaudio
    return torchaudio


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------
_mel_transform = None


def _get_mel_transform():
    """Always returns a CPU mel transform (MelSpectrogram window must match waveform device)."""
    global _mel_transform
    if _mel_transform is None:
        torchaudio = _torchaudio()
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=MEL_SAMPLE_RATE,
            n_fft=MEL_N_FFT,
            hop_length=MEL_HOP_LENGTH,
            n_mels=MEL_N_MELS,
            f_min=MEL_FMIN,
            f_max=MEL_FMAX,
            power=2.0,
        )  # stays on CPU — window tensor stays on CPU
    return _mel_transform


def extract_logmel(
    waveform: "torch.Tensor",
    device: str = "cpu",   # device arg kept for API compat; log-mel always runs on CPU
) -> "torch.Tensor":
    """
    Compute log-mel spectrogram.

    Always computed on CPU — the MelSpectrogram STFT window tensor must reside
    on the same device as the waveform, and keeping both on CPU avoids the
    CUDA/CPU mismatch error on first use before the transform is warmed up.
    Feature is lightweight so CPU is fast enough.

    Args:
        waveform: 1-D float32 tensor [T] at 16 kHz.
        device: ignored (kept for backward compatibility).

    Returns:
        Log-mel tensor [n_mels, T_frames] (float32, on CPU).
    """
    torch = _torch()
    mel_transform = _get_mel_transform()
    # Always use CPU for STFT to avoid window/tensor device mismatch
    waveform_cpu = waveform.cpu()

    with torch.no_grad():
        mel = mel_transform(waveform_cpu)    # [n_mels, T_frames]
        log_mel = torch.log(mel + 1e-9)      # log-compress

    return log_mel  # already on CPU


# ---------------------------------------------------------------------------
# WavLM self-supervised speech embedding
# ---------------------------------------------------------------------------
_ssl_model   = None
_ssl_processor = None
_ssl_device  = None


def _load_ssl_model(device: Optional[str] = None) -> Tuple[object, object, str]:
    """
    Load WavLM-base from HuggingFace (cached after first call).
    The backbone is frozen; we use it purely as a feature extractor.
    """
    global _ssl_model, _ssl_processor, _ssl_device
    if _ssl_model is not None:
        return _ssl_model, _ssl_processor, _ssl_device

    torch = _torch()
    from transformers import WavLMModel, AutoFeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading SSL backbone: %s → %s", SSL_MODEL_ID, device)
    t0 = time.perf_counter()

    _ssl_processor = AutoFeatureExtractor.from_pretrained(SSL_MODEL_ID)
    _ssl_model = WavLMModel.from_pretrained(SSL_MODEL_ID)
    _ssl_model = _ssl_model.to(device)
    _ssl_model.eval()

    # Freeze all backbone parameters
    for param in _ssl_model.parameters():
        param.requires_grad = False

    _ssl_device = device
    logger.info("SSL backbone loaded in %.1f s on %s", time.perf_counter() - t0, device)
    return _ssl_model, _ssl_processor, _ssl_device


def extract_ssl_embedding(
    waveform: "torch.Tensor",
    device: Optional[str] = None,
    layer: int = -1,
) -> "torch.Tensor":
    """
    Extract a mean-pooled WavLM hidden-state embedding.

    Args:
        waveform: 1-D float32 tensor [T] at 16 kHz.
        device: override device selection.
        layer: which transformer layer to pool (-1 = last).

    Returns:
        Embedding tensor [D=768] (float32, on CPU).
    """
    torch = _torch()
    model, processor, dev = _load_ssl_model(device)

    # Clip to SSL_MAX_SECONDS to avoid OOM
    max_samples = SSL_MAX_SECONDS * MEL_SAMPLE_RATE
    if waveform.shape[0] > max_samples:
        logger.debug("Clipping waveform to %d s for SSL embedding.", SSL_MAX_SECONDS)
        waveform = waveform[:max_samples]

    # Preprocess with HuggingFace feature extractor
    inputs = processor(
        waveform.numpy(),
        sampling_rate=MEL_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(dev)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]     # [1, T_frames, 768]
        embedding = hidden.mean(dim=1).squeeze(0) # [768]

    return embedding.cpu().float()


# ---------------------------------------------------------------------------
# Aggregate feature dict
# ---------------------------------------------------------------------------
def extract_all(
    waveform: "torch.Tensor",
    device: Optional[str] = None,
) -> Dict[str, "torch.Tensor"]:
    """
    Run the full feature extraction pipeline on a preprocessed waveform.

    Returns a dict with keys:
      - 'logmel'        : [n_mels, T_frames]
      - 'ssl_embedding' : [768]
    """
    torch = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    logmel = extract_logmel(waveform, device=device)
    t1 = time.perf_counter()
    ssl_emb = extract_ssl_embedding(waveform, device=device)
    t2 = time.perf_counter()

    logger.debug(
        "Feature extraction: logmel=%.1f ms, ssl=%.1f ms",
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
    )

    return {
        "logmel":        logmel,
        "ssl_embedding": ssl_emb,
    }
