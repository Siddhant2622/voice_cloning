"""
src/models/speaker_verify.py
Layer 4b — Spoofing-aware speaker verification (SASV).

Wraps SpeechBrain's ECAPA-TDNN speaker embedding model.
Computes cosine similarity between a test utterance and an enrolled reference.
Only activates when an enrolled voiceprint is available.

IMPORTANT: Enrolled embeddings are treated as biometric data.
  - In production: encrypt at rest, never log in plaintext.
  - In this prototype: stub encryption is noted but not fully implemented.
  - Use the ENCRYPT_STUB comments to locate where real encryption should go.

Architecture reference: Layer 4b (SASV) in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM   = 192   # ECAPA-TDNN output dimension


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# SpeechBrain ECAPA-TDNN wrapper
# ---------------------------------------------------------------------------
_sb_model  = None
_sb_device = None


def _load_ecapa_model(device: Optional[str] = None) -> tuple:
    global _sb_model, _sb_device
    if _sb_model is not None:
        return _sb_model, _sb_device

    torch = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from speechbrain.pretrained import EncoderClassifier
        logger.info("Loading ECAPA-TDNN speaker model → %s", device)
        _sb_model = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL_ID,
            run_opts={"device": device},
        )
        _sb_device = device
        logger.info("ECAPA-TDNN loaded on %s", device)
    except ImportError:
        logger.warning(
            "SpeechBrain not installed. Speaker verification pillar disabled. "
            "Install: python -m pip install speechbrain"
        )
        _sb_model = None
        _sb_device = device
    except Exception as exc:
        logger.warning("Failed to load ECAPA-TDNN (%s). Pillar disabled.", exc)
        _sb_model = None
        _sb_device = device

    return _sb_model, _sb_device


def _embed(waveform: "torch.Tensor", device: str) -> Optional["torch.Tensor"]:
    """Extract ECAPA-TDNN speaker embedding from a waveform."""
    model, dev = _load_ecapa_model(device)
    if model is None:
        return None

    torch = _torch()
    wav = waveform.unsqueeze(0).to(dev)    # [1, T]
    lens = torch.tensor([1.0]).to(dev)

    with torch.no_grad():
        embedding = model.encode_batch(wav, lens)   # [1, 1, 192]
    return embedding.squeeze().cpu().float()         # [192]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class SpeakerVerifier:
    """
    Manages speaker enrollment and verification.

    Usage (CLI):
        sv = SpeakerVerifier()
        sv.enroll("alice", reference_waveform)
        score = sv.score(test_waveform, speaker_id="alice")

    ENCRYPT_STUB: In production, enrolled embeddings should be:
      1. AES-256-GCM encrypted before persistence.
      2. Stored in a secure key-value store (e.g. encrypted object storage).
      3. Retrieved only for the duration of the call, then zeroed from memory.
    """

    def __init__(self, device: Optional[str] = None):
        torch = _torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # In-memory enrollment store  (ENCRYPT_STUB: replace with encrypted KV store)
        self._enrolled: Dict[str, "torch.Tensor"] = {}

    @property
    def available(self) -> bool:
        """True if the ECAPA-TDNN model loaded successfully."""
        model, _ = _load_ecapa_model(self.device)
        return model is not None

    def enroll(self, speaker_id: str, reference_waveform: "torch.Tensor") -> bool:
        """
        Compute and store a speaker embedding for a reference utterance.

        Args:
            speaker_id: Caller ID / account ID string.
            reference_waveform: 1-D float32 tensor at 16 kHz.

        Returns:
            True if enrollment succeeded, False if model unavailable.
        """
        embedding = _embed(reference_waveform, self.device)
        if embedding is None:
            return False
        # ENCRYPT_STUB: encrypt embedding before storing
        self._enrolled[speaker_id] = embedding
        logger.info("Enrolled speaker: %s (embedding dim=%d)", speaker_id, embedding.shape[0])
        return True

    def score(
        self,
        test_waveform: "torch.Tensor",
        speaker_id: Optional[str] = None,
        reference_embedding: Optional["torch.Tensor"] = None,
    ) -> Optional[float]:
        """
        Compute cosine similarity between test utterance and enrolled reference.

        Args:
            test_waveform: 1-D float32 tensor at 16 kHz.
            speaker_id: Look up enrolled embedding by ID.
            reference_embedding: Or pass embedding directly.

        Returns:
            Cosine similarity ∈ [-1, 1] (higher = more similar).
            None if model unavailable or no reference provided.
        """
        torch = _torch()

        ref_emb = reference_embedding
        if ref_emb is None and speaker_id is not None:
            ref_emb = self._enrolled.get(speaker_id)

        if ref_emb is None:
            logger.debug("No enrolled reference for speaker '%s'; SASV pillar skipped.", speaker_id)
            return None

        test_emb = _embed(test_waveform, self.device)
        if test_emb is None:
            return None

        # L2-normalise then dot product = cosine similarity
        ref_norm  = torch.nn.functional.normalize(ref_emb.unsqueeze(0), dim=-1)
        test_norm = torch.nn.functional.normalize(test_emb.unsqueeze(0), dim=-1)
        similarity = float((ref_norm * test_norm).sum())
        logger.debug("Speaker similarity (ECAPA): %.4f", similarity)
        return similarity

    def score_as_spoof_probability(
        self,
        test_waveform: "torch.Tensor",
        speaker_id: Optional[str] = None,
        reference_embedding: Optional["torch.Tensor"] = None,
        accept_threshold: float = 0.25,
    ) -> Optional[float]:
        """
        Convert cosine similarity into a spoof-contribution probability.
        Low similarity (poor speaker match) → high spoof contribution.

        Returns a value in [0, 1] suitable for fusion, or None if unavailable.
        """
        sim = self.score(test_waveform, speaker_id, reference_embedding)
        if sim is None:
            return None
        # Sigmoid-style inversion: low similarity → high spoof probability
        # Threshold ~0.25 corresponds to typical EER operating point for ECAPA
        import math
        spoof_prob = 1.0 / (1.0 + math.exp(5.0 * (sim - accept_threshold)))
        return float(spoof_prob)
