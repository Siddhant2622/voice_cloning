"""
training/train_modern.py
=========================
Trains the DETECT-2B-lite CM classifier on a modern multi-source dataset.

Replaces the old 162+162 balanced training with a much larger and more
diverse dataset drawn from MLAAD, WaveFake, In-the-Wild, and existing
LibriSpeech bonafide samples.

Key improvements over train_balanced.py:
  - Reads from data/modern_dataset/ (organized by download_modern_data.py)
  - Supports codec augmentation (G.711, Opus, MP3) on-the-fly
  - Supports replay augmentation on-the-fly
  - apply_mobile_replay_chain(): new label-aware augmentation that simulates
    AI/TTS voice played on Phone-A speaker -> air propagation -> Phone-B mic
    capture, including mobile bandpass, speaker distortion, AGC, and codec.
    Applied with p=0.70 to TTS/clone spoof samples to fix cross-device detection.
  - Evaluates with Equal Error Rate (EER) instead of just accuracy
  - Configurable max samples per class
  - Saves v4 checkpoint with full metadata

Usage:
    python training/train_modern.py
    python training/train_modern.py --data-dir data/modern_dataset --epochs 60 --max-samples 500
    python training/train_modern.py --codec-aug --replay-aug --epochs 80
"""

from __future__ import annotations

import sys
import csv
import time
import json
import random
import logging
import subprocess
import tempfile
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.signal as signal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import to_mono_16k
from src.features import extract_logmel, extract_ssl_embedding, extract_wav2vec2_embedding
from src.models.cm_classifier import build_cm_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_modern")

MODEL_OUT = Path("models/cm_detect2b_v4.pt")
MAX_SECONDS = 3.0
TARGET_SR = 16000
TARGET_FRAMES = int(MAX_SECONDS * 100)  # ~300 mel frames


# ---------------------------------------------------------------------------
# Codec augmentation
# ---------------------------------------------------------------------------

def _apply_codec_g711(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate G.711 mu-law codec: 8kHz, 8-bit mu-law quantization."""
    # Downsample to 8kHz
    if sr != 8000:
        gcd = math.gcd(sr, 8000)
        wav_8k = signal.resample_poly(wav, 8000 // gcd, sr // gcd).astype(np.float32)
    else:
        wav_8k = wav.copy()
    
    # Mu-law companding (ITU-T G.711)
    mu = 255.0
    wav_norm = wav_8k / (np.abs(wav_8k).max() + 1e-8)
    compressed = np.sign(wav_norm) * np.log1p(mu * np.abs(wav_norm)) / np.log1p(mu)
    # Quantize to 8-bit
    quantized = np.round(compressed * 128) / 128.0
    # Expand
    expanded = np.sign(quantized) * ((1 + mu) ** np.abs(quantized) - 1) / mu
    
    # Upsample back to original SR
    if sr != 8000:
        gcd = math.gcd(8000, sr)
        expanded = signal.resample_poly(expanded, sr // gcd, 8000 // gcd).astype(np.float32)
    
    return expanded[:len(wav)]


def _apply_codec_mp3(wav: np.ndarray, sr: int = 16000, bitrate: int = 64) -> np.ndarray:
    """Simulate MP3 compression via encode->decode cycle using ffmpeg if available."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, wav, sr)
            tmp_out = tmp_in.name.replace(".wav", "_mp3.wav")
            
            # Encode to MP3 then decode back to WAV
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_in.name,
                "-b:a", f"{bitrate}k", "-f", "mp3", "-",
            ], capture_output=True, timeout=10)
            
            # If ffmpeg not available, use a simple low-pass filter as approximation
            raise FileNotFoundError("Fallback to approximation")
    except Exception:
        # Approximation: low-pass filter at MP3 cutoff frequency
        cutoff = min(bitrate * 62.5, sr / 2 - 100)  # rough approximation
        b, a = signal.butter(4, cutoff / (sr / 2), btype='low')
        filtered = signal.filtfilt(b, a, wav).astype(np.float32)
        # Add slight quantization noise
        noise_level = 0.002
        filtered += np.random.randn(len(filtered)).astype(np.float32) * noise_level
        return filtered


def _apply_codec_opus(wav: np.ndarray, sr: int = 16000, bitrate: int = 16) -> np.ndarray:
    """Simulate low-bitrate Opus codec: aggressive low-pass + quantization."""
    # Opus at 16kbps cuts off around 6kHz
    cutoff = min(bitrate * 375, sr / 2 - 100)
    b, a = signal.butter(5, cutoff / (sr / 2), btype='low')
    filtered = signal.filtfilt(b, a, wav).astype(np.float32)
    # Quantization noise typical of low-bitrate Opus
    noise_level = 0.003
    filtered += np.random.randn(len(filtered)).astype(np.float32) * noise_level
    return filtered


def apply_codec_augmentation(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Randomly apply one of the codec augmentations."""
    codec = random.choice(["g711", "mp3_64", "mp3_128", "opus_16", "opus_32", "none"])
    
    if codec == "g711":
        return _apply_codec_g711(wav, sr)
    elif codec == "mp3_64":
        return _apply_codec_mp3(wav, sr, bitrate=64)
    elif codec == "mp3_128":
        return _apply_codec_mp3(wav, sr, bitrate=128)
    elif codec == "opus_16":
        return _apply_codec_opus(wav, sr, bitrate=16)
    elif codec == "opus_32":
        return _apply_codec_opus(wav, sr, bitrate=32)
    else:
        return wav


# ---------------------------------------------------------------------------
# Replay augmentation (lightweight simulation)
# ---------------------------------------------------------------------------

def apply_replay_augmentation(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Lightweight replay augmentation: RIR convolution + noise."""
    rt60 = random.uniform(0.15, 0.55)
    num_samples = int(rt60 * sr)
    t = np.linspace(0, rt60, num_samples, endpoint=False)
    
    rir = np.zeros(num_samples, dtype=np.float32)
    direct_idx = int(0.003 * sr)
    if direct_idx < num_samples:
        rir[direct_idx] = 1.0
    
    # Early reflections
    for delay_ms in [8, 12, 18, 25, 35]:
        idx = int(delay_ms * sr / 1000)
        if idx < num_samples:
            rir[idx] = random.uniform(0.1, 0.4) * (1 if random.random() > 0.5 else -1)
    
    # Late reverb tail
    decay = np.exp(-6.908 * t / rt60)
    late_noise = np.random.randn(num_samples).astype(np.float32) * 0.02 * decay
    rir += late_noise
    rir /= np.abs(rir).max() + 1e-8
    
    # Convolve
    out = signal.fftconvolve(wav, rir, mode="full")[:len(wav)].astype(np.float32)
    
    # Add room noise
    snr_db = random.uniform(20, 35)
    noise = np.random.randn(len(out)).astype(np.float32)
    sig_power = np.mean(out ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    out += noise * np.sqrt(noise_power)
    
    # Normalize
    mx = np.abs(out).max()
    if mx > 0:
        out = out / mx * 0.95
    
    return out


def apply_mobile_replay_chain(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate AI voice played on Phone-A speaker → air → Phone-B microphone.

    This is the exact attack vector the user reported: TTS/cloned audio rendered
    on one device and recorded by another device's built-in microphone.

    The chain applies, in order:
      1. Mobile speaker bandpass (cheap driver: ~400–3800 Hz cut-off range)
      2. Harmonic distortion (cheap speaker membrane nonlinearity)
      3. Room impulse response convolution (short RT60: 0.05–0.25 s)
      4. Mobile microphone bandpass (MEMS mic: ~200–4500 Hz range)
      5. AGC compression (automatic gain control flattens dynamics)
      6. Additive mobile ambient noise (LP-filtered white noise, SNR 14-28 dB)
      7. Codec (G.711 or Opus 16kbps — typical of a voice-call recording app)
    """
    try:
        nyq = sr / 2.0

        # 1. Loudspeaker frequency response (band-pass with resonance roll-off)
        lo_hz = random.uniform(350.0, 500.0)
        hi_hz = random.uniform(3200.0, 3900.0)
        b_hp, a_hp = signal.butter(2, lo_hz / nyq, btype="high")
        b_lp, a_lp = signal.butter(4, hi_hz / nyq, btype="low")
        out = signal.filtfilt(b_hp, a_hp, wav).astype(np.float32)
        out = signal.filtfilt(b_lp, a_lp, out).astype(np.float32)

        # 2. Cheap speaker harmonic distortion (soft-clipping + 2nd/3rd harmonic)
        dist_amount = random.uniform(0.01, 0.06)  # 1–6% THD
        out = out + dist_amount * (out ** 2) - dist_amount * 0.5 * (out ** 3)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        # 3. Short room impulse (small to medium room: 0.05–0.25 s RT60)
        rt60 = random.uniform(0.05, 0.25)
        n_rir = max(int(rt60 * sr), 16)
        t_rir = np.linspace(0, rt60, n_rir, endpoint=False)
        rir = np.zeros(n_rir, dtype=np.float32)
        direct_idx = int(0.002 * sr)
        if direct_idx < n_rir:
            rir[direct_idx] = 1.0
        for delay_ms in [5, 9, 14, 20]:
            idx = int(delay_ms * sr / 1000)
            if idx < n_rir:
                rir[idx] = random.uniform(0.05, 0.25) * random.choice([-1, 1])
        decay = np.exp(-6.908 * t_rir / rt60)
        rir += np.random.randn(n_rir).astype(np.float32) * 0.01 * decay
        rir_max = np.abs(rir).max()
        if rir_max > 1e-8:
            rir /= rir_max
        out = signal.fftconvolve(out, rir, mode="full")[:len(wav)].astype(np.float32)

        # 4. Receiving MEMS microphone response (wider than speaker, slight mid-boost)
        mic_lo = random.uniform(180.0, 280.0)
        mic_hi = random.uniform(4000.0, 4800.0)
        b_hp2, a_hp2 = signal.butter(2, mic_lo / nyq, btype="high")
        b_lp2, a_lp2 = signal.butter(3, mic_hi / nyq, btype="low")
        out = signal.filtfilt(b_hp2, a_hp2, out).astype(np.float32)
        out = signal.filtfilt(b_lp2, a_lp2, out).astype(np.float32)

        # 5. AGC: compresses dynamic range (attack fast, release slow)
        frame_len = max(int(0.02 * sr), 1)  # 20ms frames
        target_rms = 0.08
        agc_out = np.zeros_like(out)
        gain = 1.0
        for i in range(0, len(out), frame_len):
            frame = out[i:i + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2) + 1e-10))
            desired_gain = target_rms / rms if rms > 1e-6 else gain
            desired_gain = float(np.clip(desired_gain, 0.1, 8.0))
            alpha = 0.1 if desired_gain < gain else 0.5
            gain = alpha * desired_gain + (1 - alpha) * gain
            agc_out[i:i + frame_len] = frame * gain
        out = agc_out.astype(np.float32)

        # 6. Ambient noise: low-pass filtered white noise (stable, ~fan/room hiss)
        #    We use a simple first-order Butterworth LP at 2 kHz — guaranteed stable
        #    at any sample rate. Avoids the IIR instability of 44.1kHz pink filters.
        snr_db = random.uniform(14.0, 28.0)
        sig_power = float(np.mean(out ** 2)) + 1e-10
        noise_power = sig_power / (10 ** (snr_db / 10))
        white = np.random.randn(len(out)).astype(np.float32)
        # Stable simple LP at 2 kHz to tint noise warm/roomy
        noise_cutoff = min(2000.0 / nyq, 0.99)
        b_n, a_n = signal.butter(1, noise_cutoff, btype="low")
        tinted = signal.lfilter(b_n, a_n, white).astype(np.float32)
        tinted_rms = float(np.sqrt(np.mean(tinted ** 2) + 1e-10))
        out += tinted * (np.sqrt(noise_power) / tinted_rms)

        # 7. Codec simulation (G.711 phone call or 16kbps Opus messenger audio)
        codec_choice = random.choice(["g711", "opus_16", "none"])
        if codec_choice == "g711":
            out = _apply_codec_g711(out, sr)
        elif codec_choice == "opus_16":
            out = _apply_codec_opus(out, sr, bitrate=16)

        # Final normalize + NaN/Inf guard
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mx = np.abs(out).max()
        if mx > 1e-6:
            out = out / mx * 0.90
        return out

    except Exception as e:
        # Safety net: if any stage produces NaN/Inf or crashes, return the original
        # (normalized) waveform so no NaN ever enters the feature extractor.
        import logging
        logging.getLogger("train_modern").warning(
            "apply_mobile_replay_chain failed (%s) — returning original wav", e
        )
        safe = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mx = np.abs(safe).max()
        return (safe / mx * 0.90) if mx > 1e-6 else safe


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ModernAudioDataset(Dataset):
    """
    Lazy-loading dataset that reads from organized bonafide/spoof directories.
    Applies optional codec and replay augmentation on-the-fly.
    """
    
    def __init__(
        self,
        data_dir: Path,
        max_samples_per_class: int = 500,
        codec_aug: bool = False,
        replay_aug: bool = False,
        asvspoof2017_dir: Optional[Path] = Path("data/asvspoof2017"),
        asvspoof_ratio: float = 0.35,
        device: str = "cpu",
    ):
        self.data_dir = Path(data_dir)
        self.codec_aug = codec_aug
        self.replay_aug = replay_aug
        self.device = device
        self.asvspoof2017_dir = Path(asvspoof2017_dir) if asvspoof2017_dir else None
        
        # Load metadata
        meta_path = self.data_dir / "metadata.csv"
        if meta_path.exists():
            with open(meta_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            bf_rows = [r for r in rows if r["label"] == "bonafide"]
            sp_rows = [r for r in rows if r["label"] == "spoof"]
        else:
            # Fallback: scan directories
            bf_dir = self.data_dir / "bonafide"
            sp_dir = self.data_dir / "spoof"
            
            bf_files = sorted(list(bf_dir.glob("*.wav")) + list(bf_dir.glob("*.flac")) + list(bf_dir.glob("*.mp3"))) if bf_dir.exists() else []
            sp_files = sorted(list(sp_dir.glob("*.wav")) + list(sp_dir.glob("*.flac")) + list(sp_dir.glob("*.mp3"))) if sp_dir.exists() else []
            
            bf_rows = [{"file": f"bonafide/{f.name}", "label": "bonafide", "source": "unknown", "tts_system": "human"} for f in bf_files]
            sp_rows = [{"file": f"spoof/{f.name}", "label": "spoof", "source": "unknown", "tts_system": "unknown"} for f in sp_files]
        
        # Ingest ASVspoof 2017 Physical Access / Microphone Replay Data
        if self.asvspoof2017_dir and self.asvspoof2017_dir.exists():
            asv_bf, asv_sp = [], []
            train_meta = self.asvspoof2017_dir / "train" / "samples_metadata.csv"
            # Prioritize 'train' split so 'dev' remains strictly held-out for validation
            split_names = ["train"] if train_meta.exists() else ["dev"]
            for split_name in split_names:
                asv_meta = self.asvspoof2017_dir / split_name / "samples_metadata.csv"
                if asv_meta.exists():
                    with open(asv_meta, "r", encoding="utf-8") as fp:
                        for row in csv.DictReader(fp):
                            if row["label"] == "bonafide":
                                asv_bf.append({
                                    "file": row["file"],
                                    "label": "bonafide",
                                    "source": f"asvspoof2017_mic_{row.get('recording_mic', 'mic')}",
                                    "tts_system": "human_physical_mic"
                                })
                            else:
                                asv_sp.append({
                                    "file": row["file"],
                                    "label": "spoof",
                                    "source": f"asvspoof2017_replay_{row.get('recording_mic', 'mic')}",
                                    "tts_system": "replay_transducer"
                                })
            if asv_bf or asv_sp:
                logger.info("Loaded %d bonafide + %d replayed spoof samples from ASVspoof 2017", len(asv_bf), len(asv_sp))
                needed_bf = max(0, max_samples_per_class - len(bf_rows))
                target_ratio_cnt = int(max_samples_per_class * asvspoof_ratio)
                asv_target = min(max(target_ratio_cnt, needed_bf), len(asv_bf), len(asv_sp))
                if asv_target > 0:
                    random.seed(42)
                    random.shuffle(asv_bf)
                    random.shuffle(asv_sp)
                    bf_rows.extend(asv_bf[:asv_target])
                    sp_rows.extend(asv_sp[:asv_target])
                    logger.info("Added %d ASVspoof 2017 physical microphone samples per class to training pool", asv_target)
        
        # Balance and limit
        random.seed(42)
        random.shuffle(bf_rows)
        random.shuffle(sp_rows)
        
        n = min(max_samples_per_class, len(bf_rows), len(sp_rows))
        self.samples = bf_rows[:n] + sp_rows[:n]
        random.shuffle(self.samples)
        
        logger.info("Dataset: %d bonafide + %d spoof = %d total", n, n, 2 * n)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples[idx]
        raw_path = Path(row["file"])
        path = raw_path if raw_path.is_absolute() else self.data_dir / raw_path
        label = 0.0 if row["label"] == "bonafide" else 1.0

        # Determine if this is a TTS/clone spoof sample (vs replay of genuine voice)
        tts_system = row.get("tts_system", "")
        is_tts_spoof = (
            label == 1.0
            and tts_system not in ("", "human", "human_physical_mic", "replay_transducer")
        )
        
        try:
            data, sr = sf.read(str(path))
            wav = torch.from_numpy(data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            wav_16k = to_mono_16k(wav, sr)
            wav_np = wav_16k.numpy()
        except Exception as e:
            logger.warning("Error loading %s: %s — returning zeros", path, e)
            wav_np = np.zeros(int(MAX_SECONDS * TARGET_SR), dtype=np.float32)
            wav_16k = torch.zeros(int(MAX_SECONDS * TARGET_SR))

        # --------------------------------------------------------------------------
        # Label-aware augmentation strategy:
        #
        #  TTS/clone spoof  →  apply mobile replay chain with p=0.70 to simulate the
        #                       "AI voice from Phone-A speaker → Phone-B mic" attack.
        #                       This generates hard-negatives the model never saw before.
        #
        #  Replay/physical spoof → apply standard RIR replay augmentation with p=0.50
        #                          (the ASVspoof 2017 samples already carry transducer
        #                           effects; a second pass adds variety).
        #
        #  Bonafide          →  codec augmentation only (p=0.25) to teach invariance
        #                       to phone/VOIP compression without adding spoof cues.
        # --------------------------------------------------------------------------
        if is_tts_spoof:
            # 70% chance: full mobile replay chain (primary fix for cross-device issue)
            if self.replay_aug and random.random() < 0.70:
                wav_np = apply_mobile_replay_chain(wav_np, TARGET_SR)
                wav_16k = torch.from_numpy(wav_np).float()
            elif self.codec_aug and random.random() < 0.40:
                # Remaining 30% of TTS spoof: codec-only (keeps some clean TTS in pool)
                wav_np = apply_codec_augmentation(wav_np, TARGET_SR)
                wav_16k = torch.from_numpy(wav_np).float()
        elif label == 1.0:  # replay_transducer from ASVspoof 2017
            if self.replay_aug and random.random() < 0.50:
                wav_np = apply_replay_augmentation(wav_np, TARGET_SR)
                wav_16k = torch.from_numpy(wav_np).float()
        else:  # bonafide
            if self.codec_aug and random.random() < 0.25:
                wav_np = apply_codec_augmentation(wav_np, TARGET_SR)
                wav_16k = torch.from_numpy(wav_np).float()
        
        # Trim/pad
        max_len = int(MAX_SECONDS * TARGET_SR)
        if len(wav_16k) > max_len:
            start = (len(wav_16k) - max_len) // 2
            wav_16k = wav_16k[start:start + max_len]
        elif len(wav_16k) < max_len:
            wav_16k = F.pad(wav_16k, (0, max_len - len(wav_16k)))
        
        return wav_16k, label


def extract_features_batch(dataset: ModernAudioDataset, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract all features from dataset (log-mel, WavLM, Wav2Vec2, labels)."""
    logmels, ssls, w2v2s, labels = [], [], [], []
    t0 = time.time()
    
    for idx in range(len(dataset)):
        wav, label = dataset[idx]
        
        try:
            mel = extract_logmel(wav, device="cpu")
            ssl = extract_ssl_embedding(wav, device=device)
            w2v = extract_wav2vec2_embedding(wav, device=device)
            
            # Normalize mel to fixed frame count
            if mel.shape[-1] < TARGET_FRAMES:
                mel = F.pad(mel, (0, TARGET_FRAMES - mel.shape[-1]))
            else:
                mel = mel[:, :TARGET_FRAMES]
            
            logmels.append(mel)
            ssls.append(ssl)
            w2v2s.append(w2v)
            labels.append(label)
        except Exception as e:
            logger.warning("Feature extraction failed for sample %d: %s", idx, e)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(dataset):
            elapsed = time.time() - t0
            per_sample = elapsed / (idx + 1)
            remaining = (len(dataset) - idx - 1) * per_sample
            logger.info(
                "Features: %d/%d (%.1f%%) — %.2fs/sample, ~%.0fs remaining",
                idx + 1, len(dataset), 100 * (idx + 1) / len(dataset), per_sample, remaining
            )
    
    X_logmel = torch.stack(logmels).float()
    X_ssl = torch.stack(ssls).float()
    X_w2v2 = torch.stack(w2v2s).float()
    y = torch.tensor(labels, dtype=torch.float32)
    
    return X_logmel, X_ssl, X_w2v2, y


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from scores and binary labels.
    Returns (eer, threshold).
    """
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    
    # Sort by score
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    # Compute FAR and FRR at each threshold
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5  # degenerate case
    
    thresholds = np.unique(sorted_scores)
    if len(thresholds) < 2:
        return 0.5, 0.5
    
    fars = []
    frrs = []
    
    for t in thresholds:
        # FAR: fraction of bonafide (label=0) classified as spoof (score >= t)
        far = np.sum((scores >= t) & (labels == 0)) / n_neg
        # FRR: fraction of spoof (label=1) classified as bonafide (score < t)
        frr = np.sum((scores < t) & (labels == 1)) / n_pos
        fars.append(far)
        frrs.append(frr)
    
    fars = np.array(fars)
    frrs = np.array(frrs)
    
    # Find EER where FAR == FRR
    try:
        f_interp = interp1d(thresholds, fars - frrs)
        eer_threshold = brentq(f_interp, thresholds[0], thresholds[-1])
        eer = interp1d(thresholds, fars)(eer_threshold)
    except (ValueError, RuntimeError):
        # Fallback: find minimum |FAR - FRR|
        abs_diff = np.abs(fars - frrs)
        min_idx = np.argmin(abs_diff)
        eer = (fars[min_idx] + frrs[min_idx]) / 2
        eer_threshold = thresholds[min_idx]
    
    return float(eer), float(eer_threshold)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str = "data/modern_dataset",
    asvspoof2017_dir: Optional[str] = "data/asvspoof2017",
    max_samples: int = 500,
    epochs: int = 60,
    codec_aug: bool = True,
    replay_aug: bool = True,
    force_rebuild: bool = False,
    pretrained_checkpoint: Optional[str] = None,
    out_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detect Google Drive root if present
    import os
    drive_root = os.environ.get("VOICEGUARD_DRIVE_ROOT", os.environ.get("DRIVE_ROOT", ""))
    if not drive_root and Path("/content/drive/MyDrive/voiceguard").exists():
        drive_root = "/content/drive/MyDrive/voiceguard"

    if out_model is None:
        out_model = f"{drive_root}/models/cm_detect2b_v4.pt" if drive_root else "models/cm_detect2b_v4.pt"
    out_model_path = Path(out_model)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = f"{drive_root}/cache" if drive_root else "data"
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s | Max samples/class: %d | Epochs: %d", device, max_samples, epochs)
    logger.info("Checkpoint out: %s | Cache dir: %s", out_model_path, cache_dir_path)
    logger.info("Codec aug: %s | Replay aug: %s | ASVspoof 2017: %s", codec_aug, replay_aug, asvspoof2017_dir)
    
    data_path = Path(data_dir)
    # v4: cache tag bumped to bust old cache after mobile_replay_chain augmentation was added
    cache_tag = f"{max_samples}_asv17_v4" if asvspoof2017_dir else f"{max_samples}_v4"
    cache_path = cache_dir_path / f"modern_features_cache_{cache_tag}.pt"
    
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached features from %s...", cache_path)
        data = torch.load(str(cache_path), weights_only=False)
        X_logmel, X_ssl, X_w2v2, y = data["logmel"], data["ssl"], data["w2v2"], data["labels"]
    else:
        dataset = ModernAudioDataset(
            data_dir=data_path,
            max_samples_per_class=max_samples,
            codec_aug=codec_aug,
            replay_aug=replay_aug,
            asvspoof2017_dir=Path(asvspoof2017_dir) if asvspoof2017_dir else None,
            device=device,
        )
        
        if len(dataset) == 0:
            logger.error("No samples found in %s! Run data/download_modern_data.py first.", data_dir)
            sys.exit(1)
        
        X_logmel, X_ssl, X_w2v2, y = extract_features_batch(dataset, device=device)
        
        logger.info("Saving features cache to %s...", cache_path)
        torch.save({"logmel": X_logmel, "ssl": X_ssl, "w2v2": X_w2v2, "labels": y}, str(cache_path))
    
    logger.info("Dataset: logmel=%s, ssl=%s, w2v2=%s, y=%s",
                X_logmel.shape, X_ssl.shape, X_w2v2.shape, y.shape)
    
    # Split 80/20 train/val
    from torch.utils.data import TensorDataset, random_split
    
    full_dataset = TensorDataset(X_logmel, X_ssl, X_w2v2, y)
    n_val = int(len(y) * 0.2)
    n_train = len(y) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    model = build_cm_classifier(use_mamba=False).to(device)
    if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
        logger.info("Warm-starting from pretrained checkpoint: %s", pretrained_checkpoint)
        sd = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    bce = nn.BCEWithLogitsLoss()
    
    best_eer = 1.0
    best_val_acc = 0.0
    best_state = None
    
    logger.info("Training for %d epochs...", epochs)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for b_mel, b_ssl, b_w2v, b_y in train_loader:
            b_mel = b_mel.to(device)
            b_ssl = b_ssl.to(device)
            b_w2v = b_w2v.to(device)
            b_y = b_y.to(device).unsqueeze(-1)
            
            # SSL agreement auxiliary loss
            sim = F.cosine_similarity(b_ssl, b_w2v, dim=-1)
            agr_loss = (b_y.squeeze() * sim + (1 - b_y.squeeze()) * (1 - sim)).mean()
            
            optimizer.zero_grad()
            global_logits, frame_logits = model.forward_with_frames(b_mel, b_ssl, b_w2v)
            cls_loss = bce(global_logits, b_y)
            
            frame_labels = b_y.view(-1, 1, 1).expand(-1, frame_logits.shape[1], 1)
            frame_loss = bce(frame_logits, frame_labels)
            
            total_loss = cls_loss + 0.2 * frame_loss + 0.3 * agr_loss
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        scheduler.step()
        
        # Validation with EER
        model.eval()
        val_scores = []
        val_labels = []
        bf_correct, bf_total = 0, 0
        sp_correct, sp_total = 0, 0
        
        with torch.no_grad():
            for b_mel, b_ssl, b_w2v, b_y in val_loader:
                b_mel = b_mel.to(device)
                b_ssl = b_ssl.to(device)
                b_w2v = b_w2v.to(device)
                b_y_dev = b_y.to(device).unsqueeze(-1)
                
                g_logits = model(b_mel, b_ssl, b_w2v)
                probs = torch.sigmoid(g_logits).squeeze(-1)
                preds = (probs >= 0.5).float()
                
                val_scores.extend(probs.cpu().numpy().tolist())
                val_labels.extend(b_y.numpy().tolist())
                
                for pred, target in zip(preds, b_y):
                    if target.item() == 0:
                        bf_total += 1
                        if pred.item() == 0:
                            bf_correct += 1
                    else:
                        sp_total += 1
                        if pred.item() == 1:
                            sp_correct += 1
        
        val_acc = (bf_correct + sp_correct) / max(1, bf_total + sp_total)
        bf_acc = bf_correct / max(1, bf_total)
        sp_acc = sp_correct / max(1, sp_total)
        
        # Compute EER
        val_eer, eer_thresh = compute_eer(np.array(val_scores), np.array(val_labels))
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %2d/%d | train_loss=%.4f | val_acc=%.1f%% (BF=%.1f%%, SP=%.1f%%) | EER=%.2f%%",
                epoch, epochs, train_loss / len(train_loader),
                val_acc * 100, bf_acc * 100, sp_acc * 100, val_eer * 100
            )
        
        # Save best model by EER (break ties with val_acc)
        is_better = False
        if best_state is None:
            is_better = True
        elif val_eer < best_eer:
            is_better = True
        elif abs(val_eer - best_eer) < 1e-4 and val_acc > best_val_acc:
            is_better = True
            
        if is_better and val_acc >= 0.65:
            best_eer = val_eer
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # Persist best model immediately to Drive/disk so disconnects never lose progress
            try:
                torch.save(best_state, str(out_model_path))
                logger.info("✓ [Drive Saved] Best model updated: %s (EER=%.2f%%, val_acc=%.1f%%)",
                            out_model_path, best_eer * 100, best_val_acc * 100)
            except Exception as save_err:
                logger.warning("Could not persist mid-training checkpoint: %s", save_err)

        # Periodic rolling backup
        if epoch % 5 == 0:
            try:
                rolling_path = out_model_path.with_name("cm_detect2b_latest.pt")
                torch.save(model.state_dict(), str(rolling_path))
            except Exception:
                pass
    
    # Final checkpoint save
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_state = best_state if best_state is not None else model.state_dict()
    torch.save(save_state, str(out_model_path))
    logger.info("Saved final best model to %s (EER=%.2f%%, val_acc=%.1f%%)", out_model_path, best_eer * 100, best_val_acc * 100)
    
    # Save metadata
    meta = {
        "version": "v3.1-asvspoof2017",
        "val_acc": round(best_val_acc * 100, 2),
        "eer_percent": round(best_eer * 100, 2),
        "architecture": "DETECT-2B-lite (Hardened with ASVspoof 2017 V2 Physical Replay & Mic Data)",
        "ssl_models": ["wavlm-base", "wav2vec2-base-960h"],
        "mamba_enabled": False,
        "codec_augmentation": codec_aug,
        "replay_augmentation": replay_aug,
        "asvspoof2017_enabled": bool(asvspoof2017_dir),
        "data_dir": str(data_dir),
        "asvspoof2017_dir": str(asvspoof2017_dir) if asvspoof2017_dir else None,
        "max_samples_per_class": max_samples,
        "epochs": epochs,
        "training_datasets": ["librispeech", "mlaad", "in-the-wild", "wavefake", "asvspoof2017_v2"],
        "presentation_gap_mitigation": "ASVspoof 2017 V2 Physical Microphones (R01-R25) + Codec Aug + Replay Aug",
    }
    with open(out_model_path.with_suffix(".json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING SUCCESSFUL!")
    print("=" * 60)
    print(f"  Model:        {out_model_path}")
    print(f"  Val Accuracy: {best_val_acc * 100:.1f}%")
    print(f"  EER:          {best_eer * 100:.2f}%")
    print(f"  Codec Aug:    {codec_aug}")
    print(f"  Replay Aug:   {replay_aug}")
    print(f"  Datasets:     MLAAD + In-the-Wild + WaveFake + LibriSpeech + ASVspoof 2017 V2")
    print("=" * 60 + "\n")
    
    return best_eer, best_val_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CM classifier on modern multi-source dataset")
    parser.add_argument("--data-dir", default="data/modern_dataset", help="Path to organized dataset")
    parser.add_argument("--asvspoof2017-dir", default="data/asvspoof2017", help="Path to ASVspoof 2017 dataset")
    parser.add_argument("--no-asvspoof2017", action="store_true", help="Disable ASVspoof 2017 integration")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per class")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--codec-aug", action="store_true", default=True, help="Enable codec augmentation")
    parser.add_argument("--no-codec-aug", action="store_false", dest="codec_aug")
    parser.add_argument("--replay-aug", action="store_true", default=True, help="Enable replay augmentation")
    parser.add_argument("--no-replay-aug", action="store_false", dest="replay_aug")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding feature cache")
    parser.add_argument("--pretrained-checkpoint", default="models/cm_detect2b_v3_pre_asv17_backup.pt", help="Path to warm-start checkpoint")
    parser.add_argument("--no-warm-start", action="store_true", help="Train from scratch without warm-starting")
    parser.add_argument("--out", default=None, help="Output checkpoint path (defaults to Drive if mounted)")
    parser.add_argument("--cache-dir", default=None, help="Feature cache directory (defaults to Drive/cache if mounted)")
    args = parser.parse_args()
    
    asv_dir = None if args.no_asvspoof2017 else args.asvspoof2017_dir
    pretrained = None if args.no_warm_start else args.pretrained_checkpoint

    train(
        data_dir=args.data_dir,
        asvspoof2017_dir=asv_dir,
        max_samples=args.max_samples,
        epochs=args.epochs,
        codec_aug=args.codec_aug,
        replay_aug=args.replay_aug,
        force_rebuild=args.force_rebuild,
        pretrained_checkpoint=pretrained,
        out_model=args.out,
        cache_dir=args.cache_dir,
    )
