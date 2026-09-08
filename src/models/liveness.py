"""
src/models/liveness.py
Layer 4c — Passive liveness heuristic.

Computes spectral and prosodic features that synthetic speech tends to get
"too clean" or "too smooth":

  1. Spectral flatness: neural vocoders often produce unnaturally flat spectra
     in fricative regions. High flatness → suspicious.
  2. F0 jitter: human F0 has natural micro-variation; synthetic F0 is smoother.
  3. Spectral contrast: measures peak-to-valley ratios per sub-band;
     vocoder outputs tend to be more uniform.

All three are combined via a linear scoring rule calibrated on a small
heuristic basis. In Phase 3, this score feeds into the fusion layer.

Architecture reference: Layer 4c (passive liveness) in architecture.md
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        raise ImportError(
            "librosa is required for liveness analysis. "
            "Install: python -m pip install librosa"
        )


# ---------------------------------------------------------------------------
# Individual liveness signals
# ---------------------------------------------------------------------------
def spectral_flatness_score(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
) -> float:
    """
    Mean spectral flatness across frames, mapped to a liveness score.

    Spectral flatness (Wiener entropy) ∈ [0, 1]:
      - 0 = tonal / periodic (typical of synthetic neural vocoder output)
      - 1 = noise-like / flat

    Human speech has intermediate, variable flatness. Neural TTS often has
    unusually low flatness (too tonal) outside fricatives.

    Returns:
        Suspicious flatness score ∈ [0, 1] (1 = suspicious = possibly synthetic).
        Higher when flatness is unusually LOW (unexpectedly tonal).
    """
    librosa = _librosa()
    S = np.abs(librosa.stft(waveform_np, n_fft=n_fft, hop_length=hop_length))
    flatness = librosa.feature.spectral_flatness(S=S)  # [1, T_frames]
    mean_flatness = float(flatness.mean())

    # Heuristic: genuine speech flatness is ~0.02–0.12 (depends on content)
    # Very low flatness (< 0.01) → suspicious
    # Very high flatness (> 0.2) → also suspicious (replay / codec artifact)
    low_thresh, high_thresh = 0.01, 0.20
    if mean_flatness < low_thresh:
        score = 1.0 - (mean_flatness / low_thresh)    # rises toward 1 as flatness → 0
    elif mean_flatness > high_thresh:
        score = min(1.0, (mean_flatness - high_thresh) / (1.0 - high_thresh))
    else:
        score = 0.0   # in the typical human range — no suspicion from flatness

    logger.debug("Spectral flatness: %.4f → liveness suspicion score: %.3f", mean_flatness, score)
    return float(score)


def f0_jitter_score(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    fmin: float = 60.0,
    fmax: float = 500.0,
) -> float:
    """
    F0 (fundamental frequency) jitter as a liveness signal.

    Natural human speech has ~0.5–2% cycle-to-cycle jitter.
    Neural TTS often produces unnaturally smooth F0 → near-zero jitter.
    Very high jitter → noise / click artifacts.

    Returns:
        Suspicious jitter score ∈ [0, 1] (1 = suspicious).
        Scores high when jitter is abnormally low (too smooth = synthetic clue).
    """
    librosa = _librosa()
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform_np,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 10:
        logger.debug("Not enough voiced frames for jitter analysis; returning neutral.")
        return 0.3   # neutral / uncertain

    # Relative standard deviation of F0 as a jitter proxy
    f0_rsd = float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-8))

    # Typical human F0 RSD: 0.05 – 0.25 (5–25% variation)
    # < 0.02 → suspiciously smooth
    # > 0.40 → suspiciously erratic (possible replay distortion)
    low_thresh, high_thresh = 0.02, 0.40
    if f0_rsd < low_thresh:
        score = 1.0 - (f0_rsd / low_thresh)
    elif f0_rsd > high_thresh:
        score = min(1.0, (f0_rsd - high_thresh) / (1.0 - high_thresh))
    else:
        score = 0.0

    logger.debug("F0 jitter (RSD=%.4f) → liveness suspicion score: %.3f", f0_rsd, score)
    return float(score)


def spectral_contrast_score(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_bands: int = 6,
) -> float:
    """
    Spectral contrast as a liveness signal.

    Spectral contrast measures peak-to-valley ratios per frequency sub-band.
    Neural TTS vocoders (especially WaveGlow, HiFi-GAN) tend to produce
    unusually uniform contrast across sub-bands.

    Returns:
        Suspicious contrast score ∈ [0, 1].
        Scores high when contrast variance across bands is unnaturally low.
    """
    librosa = _librosa()
    S = np.abs(librosa.stft(waveform_np, n_fft=n_fft, hop_length=hop_length))
    contrast = librosa.feature.spectral_contrast(
        S=S, sr=sr, n_bands=n_bands
    )  # [n_bands+1, T_frames]
    # Standard deviation of mean contrast per band (low = unnaturally uniform)
    band_means = contrast.mean(axis=1)
    contrast_std = float(np.std(band_means))

    # Heuristic: human speech has contrast_std > 5 dB across sub-bands
    suspicious_threshold = 5.0
    if contrast_std < suspicious_threshold:
        score = 1.0 - (contrast_std / suspicious_threshold)
    else:
        score = 0.0

    logger.debug(
        "Spectral contrast std=%.2f dB → liveness suspicion score: %.3f",
        contrast_std,
        score,
    )
    return float(score)


# ---------------------------------------------------------------------------
# Aggregate liveness scorer
# ---------------------------------------------------------------------------
class LivenessScorer:
    """
    Combines spectral flatness, F0 jitter, and spectral contrast into one
    aggregate liveness suspicion score.

    Score ∈ [0, 1]:
      0 = high confidence bona fide (genuine human)
      1 = high suspicion of synthetic origin
    """

    def __init__(
        self,
        flatness_weight: float = 0.35,
        jitter_weight:   float = 0.40,
        contrast_weight: float = 0.25,
    ):
        self.flatness_weight = flatness_weight
        self.jitter_weight   = jitter_weight
        self.contrast_weight = contrast_weight

    def score(
        self,
        waveform: "torch.Tensor",
        sr: int = 16_000,
    ) -> dict:
        """
        Args:
            waveform: 1-D float32 tensor at 16 kHz.

        Returns:
            dict with keys:
              - 'liveness_score': float ∈ [0, 1]  (0=genuine, 1=suspicious)
              - 'flatness_score': float
              - 'jitter_score': float
              - 'contrast_score': float
        """
        if hasattr(waveform, "detach"):
            waveform_np = waveform.detach().cpu().numpy().astype(np.float32)
        elif hasattr(waveform, "numpy"):
            waveform_np = waveform.numpy().astype(np.float32)
        else:
            waveform_np = np.asarray(waveform, dtype=np.float32)

        flatness = spectral_flatness_score(waveform_np, sr=sr)

        try:
            jitter = f0_jitter_score(waveform_np, sr=sr)
        except Exception as exc:
            logger.warning("F0 jitter analysis failed (%s); using neutral score.", exc)
            jitter = 0.3

        try:
            contrast = spectral_contrast_score(waveform_np, sr=sr)
        except Exception as exc:
            logger.warning("Spectral contrast analysis failed (%s); using neutral.", exc)
            contrast = 0.0

        aggregate = (
            self.flatness_weight * flatness
            + self.jitter_weight * jitter
            + self.contrast_weight * contrast
        )

        return {
            "liveness_score": float(aggregate),
            "flatness_score": float(flatness),
            "jitter_score":   float(jitter),
            "contrast_score": float(contrast),
        }
