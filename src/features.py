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

SSL_MODEL_ID       = "microsoft/wavlm-base"        # WavLM backbone (~94 MB)
WAV2VEC2_MODEL_ID  = "facebook/wav2vec2-base-960h" # Wav2Vec2 backbone (~95 MB)
SSL_MAX_SECONDS    = 30                            # clip to avoid OOM on long files


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
    import numpy as np
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
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
# WavLM
_ssl_model     = None
_ssl_processor = None
_ssl_device    = None

# Wav2Vec2 (second SSL branch for dual-SSL ensemble)
_w2v2_model     = None
_w2v2_processor = None
_w2v2_device    = None


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

    try:
        _ssl_processor = AutoFeatureExtractor.from_pretrained(SSL_MODEL_ID, local_files_only=True)
        _ssl_model = WavLMModel.from_pretrained(SSL_MODEL_ID, local_files_only=True)
    except Exception:
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

    import numpy as np
    if hasattr(waveform, "detach"):
        wav_np = waveform.detach().cpu().numpy()
    elif hasattr(waveform, "numpy"):
        wav_np = waveform.numpy()
    else:
        wav_np = np.asarray(waveform, dtype=np.float32)

    # Preprocess with HuggingFace feature extractor
    inputs = processor(
        wav_np,
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
# Wav2Vec2 self-supervised speech embedding (second SSL branch)
# ---------------------------------------------------------------------------
def _load_wav2vec2_model(device: Optional[str] = None) -> Tuple[object, object, str]:
    """
    Load Wav2Vec2-base-960h from HuggingFace (cached after first call).
    Frozen as feature extractor (no fine-tuning).
    """
    global _w2v2_model, _w2v2_processor, _w2v2_device
    if _w2v2_model is not None:
        return _w2v2_model, _w2v2_processor, _w2v2_device

    torch = _torch()
    from transformers import Wav2Vec2Model, AutoFeatureExtractor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading Wav2Vec2 SSL backbone: %s -> %s", WAV2VEC2_MODEL_ID, device)
    t0 = time.perf_counter()

    try:
        _w2v2_processor = AutoFeatureExtractor.from_pretrained(WAV2VEC2_MODEL_ID, local_files_only=True)
        _w2v2_model     = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_ID, local_files_only=True)
    except Exception:
        _w2v2_processor = AutoFeatureExtractor.from_pretrained(WAV2VEC2_MODEL_ID)
        _w2v2_model     = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_ID)
    _w2v2_model     = _w2v2_model.to(device)
    _w2v2_model.eval()

    for param in _w2v2_model.parameters():
        param.requires_grad = False

    _w2v2_device = device
    logger.info("Wav2Vec2 backbone loaded in %.1f s on %s", time.perf_counter() - t0, device)
    return _w2v2_model, _w2v2_processor, _w2v2_device


def extract_wav2vec2_embedding(
    waveform: "torch.Tensor",
    device:   Optional[str] = None,
    layer:    int = -1,
) -> "torch.Tensor":
    """
    Extract a mean-pooled Wav2Vec2 hidden-state embedding.

    This forms the second SSL branch in the DETECT-2B-style dual-SSL ensemble.
    WavLM and Wav2Vec2 capture different artifacts:
      - WavLM excels at masked speech prediction (prosodic continuity)
      - Wav2Vec2 excels at phoneme boundaries (spectral transitions)
    Disagreement between them is a strong synthetic speech indicator.

    Args:
        waveform: 1-D float32 tensor [T] at 16 kHz.
        device:   override device.
        layer:    which transformer layer to pool (-1 = last).

    Returns:
        Embedding tensor [D=768] (float32, on CPU).
    """
    torch = _torch()
    try:
        model, processor, dev = _load_wav2vec2_model(device)
    except Exception as exc:
        logger.warning("Wav2Vec2 unavailable (%s); returning zeros.", exc)
        return torch.zeros(768)

    max_samples = SSL_MAX_SECONDS * MEL_SAMPLE_RATE
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    import numpy as np
    if hasattr(waveform, "detach"):
        wav_np = waveform.detach().cpu().numpy()
    else:
        wav_np = np.asarray(waveform, dtype=np.float32)

    inputs = processor(
        wav_np,
        sampling_rate=MEL_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(dev)

    with torch.no_grad():
        outputs  = model(input_values, output_hidden_states=True)
        hidden   = outputs.hidden_states[layer]      # [1, T_frames, 768]
        embedding = hidden.mean(dim=1).squeeze(0)    # [768]

    return embedding.cpu().float()


# ---------------------------------------------------------------------------
# Aggregate feature dict
# ---------------------------------------------------------------------------
def extract_all(
    waveform: "torch.Tensor",
    device:   Optional[str] = None,
    extract_wav2vec2: bool = True,
) -> Dict[str, "torch.Tensor"]:
    """
    Run the full feature extraction pipeline on a preprocessed waveform.

    Returns a dict with keys:
      - 'logmel'            : [n_mels, T_frames]
      - 'ssl_embedding'     : [768]  (WavLM)
      - 'wav2vec2_embedding': [768]  (Wav2Vec2, zero if extract_wav2vec2=False)
    """
    torch = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    logmel = extract_logmel(waveform, device=device)
    t1 = time.perf_counter()
    ssl_emb = extract_ssl_embedding(waveform, device=device)
    t2 = time.perf_counter()

    w2v2_emb = torch.zeros(768)
    if extract_wav2vec2:
        try:
            w2v2_emb = extract_wav2vec2_embedding(waveform, device=device)
        except Exception as exc:
            logger.warning("Wav2Vec2 extraction failed (%s); using zeros.", exc)
    t3 = time.perf_counter()

    logger.debug(
        "Feature extraction: logmel=%.1f ms, wavlm=%.1f ms, wav2vec2=%.1f ms",
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
        (t3 - t2) * 1000,
    )

    return {
        "logmel":             logmel,
        "ssl_embedding":      ssl_emb,
        "wav2vec2_embedding": w2v2_emb,
    }
