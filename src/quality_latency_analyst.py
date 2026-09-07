"""
src/quality_latency_analyst.py
AI Voice Detection Analyst — Quality & Latency Signal Evaluator.

Evaluates two signal categories to assess whether a voice is likely
AI-generated (synthetic/TTS) or human:

1. AUDIO QUALITY SIGNALS
   - Prosody: unnatural stress patterns, monotone delivery, oddly consistent pacing
   - Breath & disfluency: missing breath sounds, filler words, natural pauses
   - Background noise: unnaturally clean audio, non-varying ambient noise
   - Pronunciation artifacts: mishandled uncommon words, oddly "perfect" diction
   - Emotional range: flat tone, or emotional tone mismatched to content
   - Spectral consistency: unusually smooth / repetitive waveform texture

2. LATENCY / TIMING SIGNALS
   - Response time: unusually fast or unusually consistent turn-to-turn latency
   - Turn-taking: no natural overlap, interruption, or "thinking" pause
   - Self-correction: absence of stumbles, restarts, mid-sentence corrections
   - Streaming cadence: metronomic words-per-second rate vs. natural variability

Usage:
    from src.quality_latency_analyst import QualityLatencyAnalyst, LatencyProfile

    analyst = QualityLatencyAnalyst()

    # Audio-only analysis (from a 1-D float32 tensor at 16 kHz)
    result = analyst.analyze_audio(waveform)

    # With latency data from a real-time session
    profile = LatencyProfile(response_times_ms=[120, 118, 121, 119], words_per_window=[4.2, 4.1, 4.3])
    result = analyst.analyze(waveform, latency_profile=profile)

    print(result)   # VerdictReport
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy dependency accessors
# ---------------------------------------------------------------------------

def _librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        raise ImportError(
            "librosa is required for quality analysis. "
            "Install: python -m pip install librosa"
        )


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# Latency profile dataclass (populated from real-time session data)
# ---------------------------------------------------------------------------

@dataclass
class LatencyProfile:
    """
    Optional timing data from a real-time voice interaction session.

    All fields are optional; when absent, the corresponding latency
    signal will fall back to a neutral/unknown score.

    Args:
        response_times_ms: List of turn-to-turn response latencies in ms.
            AI synthesizers tend to be <200 ms and highly consistent.
            Humans typically show 200-800 ms with high variability.
        words_per_window: Words-per-second rate per analysis window.
            AI streaming TTS is metronomic; humans show natural variance.
        has_self_corrections: Whether any stumbles/restarts were detected
            in the transcript. Absence is suspicious for AI.
        has_thinking_pauses: Whether "hmm", "uh", extended silences before
            responding were observed. Absence is suspicious for AI.
        has_interruptions: Whether the speaker attempted to interrupt or
            was interrupted. AI agents rarely produce interruptions.
        transcript: Optional raw transcript text for filler word scanning.
    """
    response_times_ms: List[float] = field(default_factory=list)
    words_per_window: List[float] = field(default_factory=list)
    has_self_corrections: Optional[bool] = None
    has_thinking_pauses: Optional[bool] = None
    has_interruptions: Optional[bool] = None
    transcript: Optional[str] = None


# ---------------------------------------------------------------------------
# Individual audio quality signal scorers
# (each returns a float ∈ [0, 1], where 1.0 = strongly suspicious of AI)
# ---------------------------------------------------------------------------

def score_prosody_smoothness(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    frame_ms: int = 30,
    hop_ms: int = 10,
) -> Tuple[float, str]:
    """
    Prosody: detect unnaturally smooth / monotone energy delivery.

    Natural speech shows high short-time energy variance due to stress
    patterns, pauses, and emphasis. TTS tends to have unnaturally
    consistent RMS energy — neither too flat nor too spiky.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious AI prosody.
    """
    hop = int(sr * hop_ms / 1000)
    frame = int(sr * frame_ms / 1000)

    # Short-time RMS energy per frame
    n_frames = max(1, (len(waveform_np) - frame) // hop)
    rms_frames = np.array([
        np.sqrt(np.mean(waveform_np[i * hop: i * hop + frame] ** 2 + 1e-12))
        for i in range(n_frames)
    ])

    rms_cv = float(np.std(rms_frames) / (np.mean(rms_frames) + 1e-9))  # coefficient of variation

    # Human speech RMS CV typically 0.35–1.2
    # Neural TTS is typically 0.10–0.30 (unnaturally smooth)
    # Very erratic (>1.5) could be noisy/recorded badly
    if rms_cv < 0.15:
        score = 1.0 - (rms_cv / 0.15)       # very flat energy → strong AI signal
        detail = f"energy CV={rms_cv:.3f} (unusually monotone, <0.15 threshold)"
    elif rms_cv < 0.30:
        score = (0.30 - rms_cv) / 0.15 * 0.5  # moderately suspicious
        detail = f"energy CV={rms_cv:.3f} (slightly monotone)"
    else:
        score = 0.0
        detail = f"energy CV={rms_cv:.3f} (natural human range)"

    logger.debug("Prosody smoothness: %s → score=%.3f", detail, score)
    return float(score), detail


def score_breath_disfluency(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    silence_threshold_db: float = -45.0,
    min_breath_ms: int = 50,
    max_breath_ms: int = 300,
) -> Tuple[float, str]:
    """
    Breath & disfluency: detect missing natural breath sounds and pauses.

    Human speech contains frequent short inhalation/exhalation events
    (50–300 ms) between phrases. TTS skips these entirely or has
    unnaturally long clean silences.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (missing breath/disfluency).
    """
    hop = int(sr * 0.010)   # 10 ms hop
    frame = int(sr * 0.025) # 25 ms frame

    n_frames = max(1, (len(waveform_np) - frame) // hop)
    rms_db = np.array([
        20 * math.log10(
            np.sqrt(np.mean(waveform_np[i * hop: i * hop + frame] ** 2)) + 1e-9
        )
        for i in range(n_frames)
    ])

    # Identify silence segments (below threshold)
    is_silent = rms_db < silence_threshold_db

    # Find runs of silence and measure their lengths
    runs: List[int] = []
    current_run = 0
    for val in is_silent:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    if not runs:
        # No silence at all — strongly suspicious (TTS can be completely gapless)
        return 0.85, "no silence gaps detected (gapless audio — typical of TTS)"

    # Convert frame counts to ms
    hop_ms = hop / sr * 1000
    run_ms = [r * hop_ms for r in runs]

    breath_like = [ms for ms in run_ms if min_breath_ms <= ms <= max_breath_ms]
    breath_ratio = len(breath_like) / max(1, len(run_ms))

    # Very long silences only (no breath-length ones) → suspicious
    long_only = all(ms > max_breath_ms or ms < min_breath_ms for ms in run_ms)

    if breath_ratio < 0.05 and long_only:
        score = 0.75
        detail = f"no breath-length pauses ({min_breath_ms}–{max_breath_ms}ms); silence_segments={len(runs)}"
    elif breath_ratio < 0.10:
        score = 0.45
        detail = f"few breath-length pauses (ratio={breath_ratio:.2f})"
    else:
        score = max(0.0, 0.3 - breath_ratio)
        detail = f"natural breath pauses present (ratio={breath_ratio:.2f})"

    logger.debug("Breath/disfluency: %s → score=%.3f", detail, score)
    return float(score), detail


def score_background_noise_stability(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    n_segments: int = 10,
) -> Tuple[float, str]:
    """
    Background noise: detect unnaturally clean or static-stable background.

    Human recordings in real environments show naturally drifting SNR
    and shifting ambient noise. Studio-clean TTS has near-zero background
    noise variance.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (too-clean or unnaturally static).
    """
    seg_len = len(waveform_np) // max(1, n_segments)
    if seg_len < 160:
        return 0.3, "audio too short for background stability analysis"

    # Estimate background energy via minimum-filtered RMS per segment
    seg_rms = []
    for i in range(n_segments):
        seg = waveform_np[i * seg_len: (i + 1) * seg_len]
        # Use 10th percentile of absolute values as background noise floor proxy
        noise_floor = np.percentile(np.abs(seg), 10) if len(seg) > 0 else 1e-9
        seg_rms.append(float(noise_floor))

    noise_cv = float(np.std(seg_rms) / (np.mean(seg_rms) + 1e-9))
    mean_noise = float(np.mean(seg_rms))

    # Near-zero noise floor across all segments → too clean for real recording
    if mean_noise < 1e-5 and noise_cv < 0.1:
        score = 0.90
        detail = f"near-silent background (mean_noise={mean_noise:.2e}, CV={noise_cv:.3f}) — studio-clean TTS"
    elif noise_cv < 0.05:
        score = 0.65
        detail = f"background noise suspiciously static (CV={noise_cv:.3f})"
    elif noise_cv < 0.15:
        score = 0.25
        detail = f"background noise slightly stable (CV={noise_cv:.3f})"
    else:
        score = 0.0
        detail = f"background noise varies naturally (CV={noise_cv:.3f})"

    logger.debug("Background noise stability: %s → score=%.3f", detail, score)
    return float(score), detail


def score_pronunciation_uniformity(
    waveform_np: np.ndarray,
    sr: int = 16_000,
) -> Tuple[float, str]:
    """
    Pronunciation artifacts: detect unnaturally uniform / perfect phoneme
    articulation by measuring MFCC temporal coefficient-of-variation.

    Human speakers show micro-variability in articulation across phonemes.
    Neural TTS tends to produce extremely uniform, artifact-free MFCCs.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (too perfect).
    """
    librosa = _librosa()

    mfccs = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=13)  # [13, T]
    # Mean CV per coefficient, then average across coefficients
    coeff_cvs = []
    for c in range(mfccs.shape[0]):
        row = mfccs[c]
        cv = float(np.std(row) / (abs(np.mean(row)) + 1e-9))
        coeff_cvs.append(cv)

    mean_cv = float(np.mean(coeff_cvs))

    # Human speech MFCC CV typically > 0.5
    # Neural TTS often < 0.25 (too uniform)
    if mean_cv < 0.20:
        score = 1.0 - (mean_cv / 0.20)
        detail = f"MFCC temporal CV={mean_cv:.3f} (unnaturally uniform pronunciation)"
    elif mean_cv < 0.40:
        score = (0.40 - mean_cv) / 0.20 * 0.5
        detail = f"MFCC temporal CV={mean_cv:.3f} (slightly uniform)"
    else:
        score = 0.0
        detail = f"MFCC temporal CV={mean_cv:.3f} (natural articulation variance)"

    logger.debug("Pronunciation uniformity: %s → score=%.3f", detail, score)
    return float(score), detail


def score_emotional_range(
    waveform_np: np.ndarray,
    sr: int = 16_000,
    frame_ms: int = 100,
    hop_ms: int = 50,
) -> Tuple[float, str]:
    """
    Emotional range: detect flat or emotionally mismatched delivery.

    Measures spectral centroid variability as a proxy for tonal
    expressivity — centroid shifts correlate with vocal effort / emotion.
    Neutral/flat TTS shows very low centroid variance.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (flat emotional range).
    """
    librosa = _librosa()

    centroid = librosa.feature.spectral_centroid(y=waveform_np, sr=sr)  # [1, T]
    centroid_flat = centroid.flatten()

    centroid_cv = float(np.std(centroid_flat) / (np.mean(centroid_flat) + 1.0))

    # Human expressive speech: centroid CV typically > 0.25
    # Flat TTS: < 0.10
    if centroid_cv < 0.08:
        score = 1.0 - (centroid_cv / 0.08)
        detail = f"spectral centroid CV={centroid_cv:.3f} (emotionally flat)"
    elif centroid_cv < 0.18:
        score = (0.18 - centroid_cv) / 0.10 * 0.5
        detail = f"spectral centroid CV={centroid_cv:.3f} (limited emotional range)"
    else:
        score = 0.0
        detail = f"spectral centroid CV={centroid_cv:.3f} (natural emotional variance)"

    logger.debug("Emotional range: %s → score=%.3f", detail, score)
    return float(score), detail


def score_spectral_consistency(
    waveform_np: np.ndarray,
    sr: int = 16_000,
) -> Tuple[float, str]:
    """
    Spectral consistency: detect unusually smooth or repetitive waveform texture.

    Measures auto-correlation of MFCC frames over time — high auto-correlation
    (frames very similar to their neighbors) indicates synthetic smoothing
    typical of neural vocoder outputs (HiFi-GAN, WaveGlow).

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (too smooth / repetitive).
    """
    librosa = _librosa()

    mfccs = librosa.feature.mfcc(y=waveform_np, sr=sr, n_mfcc=20)  # [20, T]
    if mfccs.shape[1] < 4:
        return 0.3, "audio too short for spectral consistency analysis"

    # Lag-1 autocorrelation of the full MFCC matrix (column-wise correlation)
    frames = mfccs.T  # [T, 20]
    corr_values = []
    for lag in range(1, min(4, frames.shape[0])):
        corr = float(np.corrcoef(frames[:-lag].flatten(), frames[lag:].flatten())[0, 1])
        corr_values.append(corr)

    mean_autocorr = float(np.mean(corr_values))

    # Human speech frame-to-frame MFCC correlation: typically 0.5–0.75
    # Neural TTS: often > 0.85 (frames too similar due to vocoder smoothing)
    if mean_autocorr > 0.92:
        score = (mean_autocorr - 0.92) / 0.08
        detail = f"MFCC autocorr={mean_autocorr:.3f} (extremely smooth — strong vocoder signature)"
    elif mean_autocorr > 0.85:
        score = (mean_autocorr - 0.85) / 0.07 * 0.6
        detail = f"MFCC autocorr={mean_autocorr:.3f} (suspiciously smooth)"
    else:
        score = 0.0
        detail = f"MFCC autocorr={mean_autocorr:.3f} (natural temporal variation)"

    logger.debug("Spectral consistency: %s → score=%.3f", detail, score)
    return float(score), detail


# ---------------------------------------------------------------------------
# Latency / Timing signal scorers
# ---------------------------------------------------------------------------

def score_response_time(
    response_times_ms: List[float],
) -> Tuple[float, str]:
    """
    Response time: unusually fast or consistently low latency.

    Human response latency: 200–800 ms, with high variability.
    AI TTS pipelines: typically 100–300 ms, very consistent.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (AI-like latency).
    """
    if not response_times_ms:
        return 0.3, "no response time data available (neutral)"

    mean_ms = float(np.mean(response_times_ms))
    cv = float(np.std(response_times_ms) / (mean_ms + 1e-6))

    # Too fast AND too consistent → strong AI signal
    if mean_ms < 200 and cv < 0.10:
        score = 0.90
        detail = f"mean_latency={mean_ms:.0f}ms, CV={cv:.3f} (very fast + consistent — AI signature)"
    elif mean_ms < 300 and cv < 0.20:
        score = 0.65
        detail = f"mean_latency={mean_ms:.0f}ms, CV={cv:.3f} (fast + moderately consistent)"
    elif cv < 0.15:
        score = 0.50
        detail = f"mean_latency={mean_ms:.0f}ms, CV={cv:.3f} (suspiciously consistent turns)"
    elif mean_ms > 800:
        score = 0.05
        detail = f"mean_latency={mean_ms:.0f}ms (slow — consistent with human thinking time)"
    else:
        score = 0.10
        detail = f"mean_latency={mean_ms:.0f}ms, CV={cv:.3f} (within human response range)"

    logger.debug("Response time: %s → score=%.3f", detail, score)
    return float(score), detail


def score_turn_taking(
    has_thinking_pauses: Optional[bool],
    has_interruptions: Optional[bool],
) -> Tuple[float, str]:
    """
    Turn-taking: detect absence of natural overlap, interruption, or thinking pauses.

    Human conversation includes: "hmm", extended pre-response pauses,
    occasional interruptions, and natural backchanneling.
    AI agents rarely produce these turn-taking behaviors.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (too robotic turn-taking).
    """
    if has_thinking_pauses is None and has_interruptions is None:
        return 0.35, "no turn-taking data available (neutral)"

    evidence = []
    score_parts = []

    if has_thinking_pauses is False:
        score_parts.append(0.55)
        evidence.append("no thinking pauses / hesitation sounds detected")
    elif has_thinking_pauses is True:
        score_parts.append(0.0)
        evidence.append("natural thinking pauses present")

    if has_interruptions is False:
        score_parts.append(0.40)
        evidence.append("no interruptions or turn-overlap detected")
    elif has_interruptions is True:
        score_parts.append(0.0)
        evidence.append("natural interruptions/overlap present")

    score = float(np.mean(score_parts)) if score_parts else 0.35
    detail = "; ".join(evidence) if evidence else "neutral"

    logger.debug("Turn-taking: %s → score=%.3f", detail, score)
    return float(score), detail


def score_self_correction(
    has_self_corrections: Optional[bool],
    transcript: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Self-correction: detect absence of stumbles, restarts, mid-sentence corrections.

    Human speech contains self-repair sequences ("I mean...", "no wait",
    "actually", restarts mid-word). AI TTS has none of these.

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (absent self-corrections).
    """
    filler_patterns = [
        "um", "uh", "er", "hmm", "like", "you know",
        "i mean", "no wait", "actually", "sorry", "hang on",
        "let me", "i meant", "what i meant"
    ]

    filler_count = 0
    if transcript:
        t_lower = transcript.lower()
        for fp in filler_patterns:
            filler_count += t_lower.count(fp)

    if has_self_corrections is False and filler_count == 0:
        score = 0.80
        detail = "no self-corrections and no filler words in transcript — characteristic of TTS"
    elif has_self_corrections is False:
        score = 0.50
        detail = f"no self-corrections detected; filler_words={filler_count}"
    elif has_self_corrections is True:
        score = 0.05
        detail = f"self-corrections present; filler_words={filler_count} (human indicator)"
    elif filler_count == 0 and transcript:
        score = 0.55
        detail = "transcript contains no disfluencies or filler words (suspiciously clean)"
    elif filler_count > 0:
        score = max(0.0, 0.2 - filler_count * 0.05)
        detail = f"filler words present (count={filler_count}) — natural human pattern"
    else:
        score = 0.35
        detail = "insufficient data for self-correction analysis"

    logger.debug("Self-correction: %s → score=%.3f", detail, score)
    return float(score), detail


def score_streaming_cadence(
    words_per_window: List[float],
) -> Tuple[float, str]:
    """
    Streaming cadence: detect metronomic words-per-second rate.

    Human speech rate varies significantly across utterances (1–8 WPS).
    AI streaming TTS delivers at a nearly constant rate (metronomic cadence).

    Returns:
        (score, detail_str): score ∈ [0,1]; 1 = suspicious (metronomic cadence).
    """
    if not words_per_window or len(words_per_window) < 3:
        return 0.30, "insufficient streaming cadence data (need ≥3 windows)"

    arr = np.array(words_per_window, dtype=float)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-9))

    # Human WPS CV: typically 0.25–0.60
    # AI TTS streaming: typically < 0.08
    if cv < 0.05:
        score = 0.90
        detail = f"WPS CV={cv:.3f} (extremely metronomic — strong TTS cadence signature)"
    elif cv < 0.12:
        score = 0.70
        detail = f"WPS CV={cv:.3f} (suspiciously consistent streaming rate)"
    elif cv < 0.20:
        score = 0.35
        detail = f"WPS CV={cv:.3f} (slightly regular cadence)"
    else:
        score = 0.0
        detail = f"WPS CV={cv:.3f} (natural variable speech rate)"

    logger.debug("Streaming cadence: %s → score=%.3f", detail, score)
    return float(score), detail


# ---------------------------------------------------------------------------
# Main analyst class
# ---------------------------------------------------------------------------

@dataclass
class QualityLatencyResult:
    """
    Raw per-signal scores and details before verdict computation.
    All scores ∈ [0, 1]: 0 = human-like, 1 = AI-like.
    """
    # Audio quality signals
    prosody_score: float = 0.0
    prosody_detail: str = ""
    breath_score: float = 0.0
    breath_detail: str = ""
    background_score: float = 0.0
    background_detail: str = ""
    pronunciation_score: float = 0.0
    pronunciation_detail: str = ""
    emotional_score: float = 0.0
    emotional_detail: str = ""
    spectral_score: float = 0.0
    spectral_detail: str = ""

    # Latency / timing signals
    response_time_score: float = 0.0
    response_time_detail: str = ""
    turn_taking_score: float = 0.0
    turn_taking_detail: str = ""
    self_correction_score: float = 0.0
    self_correction_detail: str = ""
    cadence_score: float = 0.0
    cadence_detail: str = ""

    # Aggregates
    quality_aggregate: float = 0.0
    latency_aggregate: float = 0.0
    overall_score: float = 0.0

    def quality_signals(self) -> Dict[str, Tuple[float, str]]:
        return {
            "Prosody":          (self.prosody_score,       self.prosody_detail),
            "Breath/Disfluency": (self.breath_score,       self.breath_detail),
            "Background Noise": (self.background_score,    self.background_detail),
            "Pronunciation":    (self.pronunciation_score, self.pronunciation_detail),
            "Emotional Range":  (self.emotional_score,     self.emotional_detail),
            "Spectral Texture": (self.spectral_score,      self.spectral_detail),
        }

    def latency_signals(self) -> Dict[str, Tuple[float, str]]:
        return {
            "Response Time":    (self.response_time_score,   self.response_time_detail),
            "Turn-Taking":      (self.turn_taking_score,     self.turn_taking_detail),
            "Self-Correction":  (self.self_correction_score, self.self_correction_detail),
            "Streaming Cadence": (self.cadence_score,        self.cadence_detail),
        }


class QualityLatencyAnalyst:
    """
    AI voice detection analyst that evaluates audio quality and timing signals
    to produce a structured verdict: Likely AI-generated / Likely human / Uncertain.

    Weights are tunable via constructor; defaults are calibrated on
    a mix of neural TTS (edge-tts, pyttsx3) and LibriSpeech human speech.

    Quality signal weights (sum to 1.0):
        prosody=0.20, breath=0.18, background=0.12,
        pronunciation=0.20, emotional=0.15, spectral=0.15

    Latency signal weights (sum to 1.0):
        response_time=0.35, turn_taking=0.25,
        self_correction=0.25, cadence=0.15

    Overall weighting:
        quality_weight=0.70, latency_weight=0.30
        (latency data is often absent, so quality carries more weight)
    """

    def __init__(
        self,
        # Quality signal weights
        prosody_w: float = 0.20,
        breath_w: float = 0.18,
        background_w: float = 0.12,
        pronunciation_w: float = 0.20,
        emotional_w: float = 0.15,
        spectral_w: float = 0.15,
        # Latency signal weights
        response_time_w: float = 0.35,
        turn_taking_w: float = 0.25,
        self_correction_w: float = 0.25,
        cadence_w: float = 0.15,
        # Top-level weights
        quality_w: float = 0.70,
        latency_w: float = 0.30,
    ):
        self.quality_weights = {
            "prosody":       prosody_w,
            "breath":        breath_w,
            "background":    background_w,
            "pronunciation": pronunciation_w,
            "emotional":     emotional_w,
            "spectral":      spectral_w,
        }
        self.latency_weights = {
            "response_time":   response_time_w,
            "turn_taking":     turn_taking_w,
            "self_correction": self_correction_w,
            "cadence":         cadence_w,
        }
        self.quality_w = quality_w
        self.latency_w = latency_w

    # ------------------------------------------------------------------

    def analyze_audio(
        self,
        waveform: "torch.Tensor",
        sr: int = 16_000,
    ) -> QualityLatencyResult:
        """Analyze audio-only (no latency data)."""
        return self.analyze(waveform, sr=sr, latency_profile=None)

    def analyze(
        self,
        waveform: "torch.Tensor",
        sr: int = 16_000,
        latency_profile: Optional[LatencyProfile] = None,
    ) -> QualityLatencyResult:
        """
        Run the full quality + latency analysis.

        Args:
            waveform: 1-D float32 tensor [T] at 16 kHz.
            sr: sample rate (should always be 16_000 after preprocessing).
            latency_profile: optional real-time session timing data.

        Returns:
            QualityLatencyResult with per-signal scores and aggregates.
        """
        torch = _torch()
        waveform_np = waveform.numpy().astype(np.float32)

        result = QualityLatencyResult()

        # ── Audio Quality Signals ────────────────────────────────────────
        result.prosody_score,       result.prosody_detail       = score_prosody_smoothness(waveform_np, sr)
        result.breath_score,        result.breath_detail        = score_breath_disfluency(waveform_np, sr)
        result.background_score,    result.background_detail    = score_background_noise_stability(waveform_np, sr)

        try:
            result.pronunciation_score, result.pronunciation_detail = score_pronunciation_uniformity(waveform_np, sr)
            result.emotional_score,     result.emotional_detail     = score_emotional_range(waveform_np, sr)
            result.spectral_score,      result.spectral_detail      = score_spectral_consistency(waveform_np, sr)
        except Exception as exc:
            logger.warning("librosa-based quality signal failed: %s. Using neutral.", exc)

        result.quality_aggregate = (
            self.quality_weights["prosody"]       * result.prosody_score
            + self.quality_weights["breath"]      * result.breath_score
            + self.quality_weights["background"]  * result.background_score
            + self.quality_weights["pronunciation"] * result.pronunciation_score
            + self.quality_weights["emotional"]   * result.emotional_score
            + self.quality_weights["spectral"]    * result.spectral_score
        )

        # ── Latency / Timing Signals ─────────────────────────────────────
        lp = latency_profile or LatencyProfile()

        result.response_time_score,   result.response_time_detail   = score_response_time(lp.response_times_ms)
        result.turn_taking_score,     result.turn_taking_detail      = score_turn_taking(lp.has_thinking_pauses, lp.has_interruptions)
        result.self_correction_score, result.self_correction_detail  = score_self_correction(lp.has_self_corrections, lp.transcript)
        result.cadence_score,         result.cadence_detail          = score_streaming_cadence(lp.words_per_window)

        result.latency_aggregate = (
            self.latency_weights["response_time"]    * result.response_time_score
            + self.latency_weights["turn_taking"]    * result.turn_taking_score
            + self.latency_weights["self_correction"] * result.self_correction_score
            + self.latency_weights["cadence"]        * result.cadence_score
        )

        # ── Overall score ────────────────────────────────────────────────
        has_latency = bool(
            lp.response_times_ms
            or lp.words_per_window
            or lp.has_self_corrections is not None
            or lp.has_thinking_pauses is not None
        )
        if has_latency:
            result.overall_score = (
                self.quality_w * result.quality_aggregate
                + self.latency_w * result.latency_aggregate
            )
        else:
            # No latency data — use quality only
            result.overall_score = result.quality_aggregate

        result.overall_score = float(np.clip(result.overall_score, 0.0, 1.0))
        logger.debug(
            "QualityLatencyAnalyst: quality=%.3f latency=%.3f overall=%.3f",
            result.quality_aggregate, result.latency_aggregate, result.overall_score,
        )
        return result
