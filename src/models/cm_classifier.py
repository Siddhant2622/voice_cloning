"""
src/models/cm_classifier.py
============================
Upgraded Countermeasure (CM) classifier — Resemble AI DETECT-2B inspired.

Architecture (v2):
  Input: log-mel [n_mels=80, T_frames]  +  WavLM SSL [768]  +  Wav2Vec2 SSL [768]

  ┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
  │  Log-Mel Branch     │    │  WavLM Branch        │    │  Wav2Vec2 Branch (NEW│
  │  Conv1D + ASP pool  │    │  Linear proj         │    │  Linear proj         │
  │  → 256-dim          │    │  → 256-dim           │    │  → 256-dim           │
  └─────────┬───────────┘    └──────────┬───────────┘    └──────────┬───────────┘
            └─────────────────┬─────────┘─────────────────┬─────────┘
                              │     Concat [768-dim]        │
                              ▼                             │
                      Mamba-SSM Encoder (NEW)               │
                      4 S4-lite blocks                      │
                      Temporal frame modeling               │
                              │                             │
                   ┌──────────┴──────────┐                  │
                   ▼                     ▼                  │
             Global Head           Frame Head (NEW)         │
             (BCE loss)            (per-frame BCE)          │
                   │                     │                  │
            Global Score         Frame Scores [T]           │
            (existing API)        (new output)              │

Key improvements inspired by Resemble AI DETECT-2B:
  1. Dual SSL ensemble (WavLM + Wav2Vec2) — each captures different
     artifacts; disagreement between them is a synthetic speech signal
  2. Mamba-SSM temporal encoder — frame-level sequential modeling,
     catching temporal inconsistencies in prosody and pitch contour
  3. Frame-level output head — pinpoints exactly which frames are synthetic
     (critical for detecting spliced/partial AI injection attacks)
  4. SSL agreement auxiliary loss in training — detects cross-SSL inconsistency

0 = definitely bona fide (genuine human speech)
1 = definitely synthetic / spoofed
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy torch import
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch

def _nn():
    import torch.nn as nn
    return nn


# ---------------------------------------------------------------------------
# Attentive Statistics Pooling
# ---------------------------------------------------------------------------
class AttentiveStatisticsPooling:
    """
    Attentive statistics pooling:
      Input:  [batch, T, C]
      Output: [batch, 2*C]
    """

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _ASP(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(in_dim, in_dim // 2),
                    nn.Tanh(),
                    nn.Linear(in_dim // 2, 1),
                )

            def forward(self, x):
                # x: [B, T, C]
                import torch
                attn = self.attention(x)           # [B, T, 1]
                attn = torch.softmax(attn, dim=1)  # normalise over time
                mean = (attn * x).sum(dim=1)       # [B, C]
                var  = (attn * x.pow(2)).sum(dim=1) - mean.pow(2)
                std  = (var.clamp(min=1e-8)).sqrt()
                return torch.cat([mean, std], dim=-1)  # [B, 2C]

        return _ASP(*args, **kwargs)


# ---------------------------------------------------------------------------
# CM Classifier network (v2 — Resemble AI DETECT-2B inspired)
# ---------------------------------------------------------------------------
def build_cm_classifier(
    logmel_n_mels: int = 80,
    wavlm_dim:     int = 768,
    wav2vec2_dim:  int = 768,
    hidden:        int = 256,
    use_mamba:     bool = True,
    mamba_layers:  int = 4,
):
    """
    Factory function — returns an nn.Module.

    Args:
        logmel_n_mels:  Number of mel bands in log-mel spectrogram input.
        wavlm_dim:      WavLM SSL embedding dimension (768 for wavlm-base).
        wav2vec2_dim:   Wav2Vec2 SSL embedding dimension (768 for base).
        hidden:         Hidden layer size for fusion MLP.
        use_mamba:      Whether to use Mamba-SSM temporal encoder.
                        Auto-disabled if import fails (pure-PyTorch fallback).
        mamba_layers:   Number of S4-lite blocks in the Mamba encoder.
    """
    import torch
    import torch.nn as nn

    # Try to import Mamba encoder
    _use_mamba = use_mamba
    MambaEncoder = None
    if _use_mamba:
        try:
            from src.models.mamba_ssm import MambaEncoder as _ME
            MambaEncoder = _ME
        except Exception as e:
            logger.warning("Mamba SSM unavailable (%s) — using mean pooling.", e)
            _use_mamba = False

    class _LogMelBranch(nn.Module):
        """1-D temporal CNN + attentive statistics pooling over mel frames."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(logmel_n_mels, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
            )
            self.asp = AttentiveStatisticsPooling(128)
            self.out_dim = 256   # 2 × 128

        def forward(self, logmel):
            # logmel: [B, n_mels, T]
            x = self.conv(logmel)      # [B, 128, T]
            x = x.permute(0, 2, 1)    # [B, T, 128]
            x = self.asp(x)            # [B, 256]
            return x

    class _SSLBranch(nn.Module):
        """Projection of a single SSL embedding (WavLM or Wav2Vec2)."""

        def __init__(self, in_dim: int, out_dim: int = hidden):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            )
            self.out_dim = out_dim

        def forward(self, ssl_emb):
            # ssl_emb: [B, in_dim]
            return self.proj(ssl_emb)

    class CMClassifier(nn.Module):
        """
        Full v2 countermeasure classifier.
        Fuses log-mel + WavLM + Wav2Vec2 → Mamba-SSM → global + frame-level scores.
        """

        def __init__(self):
            super().__init__()
            self.mel_branch     = _LogMelBranch()
            self.wavlm_branch   = _SSLBranch(wavlm_dim,    hidden)
            self.wav2vec2_branch= _SSLBranch(wav2vec2_dim, hidden)

            # Feature fusion input dimension
            # mel=256, wavlm=hidden, wav2vec2=hidden
            fusion_in = self.mel_branch.out_dim + hidden + hidden  # 256 + 256 + 256 = 768

            # Mamba-SSM temporal encoder (applied to joint frame-level features)
            self._use_mamba = _use_mamba
            if _use_mamba and MambaEncoder is not None:
                self.mamba = MambaEncoder(d_model=fusion_in, n_layers=mamba_layers)
                logger.info("CMClassifier: Mamba-SSM encoder enabled (%d layers, d=%d)",
                           mamba_layers, fusion_in)
            else:
                self.mamba = None

            # Global classification head
            self.global_head = nn.Sequential(
                nn.Linear(fusion_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

            # Frame-level classification head (NEW)
            # Receives per-frame features [B, T, fusion_in] → [B, T, 1]
            self.frame_head = nn.Sequential(
                nn.Linear(fusion_in, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def _encode(
            self,
            logmel:       "torch.Tensor",  # [B, n_mels, T]
            wavlm_emb:    "torch.Tensor",  # [B, wavlm_dim]
            wav2vec2_emb: "torch.Tensor",  # [B, wav2vec2_dim]
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Encode inputs into:
              - global_feat: [B, fusion_in]  (mean-pooled, or Mamba-pooled)
              - frame_feats: [B, T, fusion_in]  (per-frame for frame head)
            """
            # Branch encodings
            mel_feat      = self.mel_branch(logmel)          # [B, 256]
            wavlm_feat    = self.wavlm_branch(wavlm_emb)     # [B, hidden]
            wav2vec2_feat = self.wav2vec2_branch(wav2vec2_emb)  # [B, hidden]

            # Global fusion (concatenate pooled features)
            global_feat = torch.cat([mel_feat, wavlm_feat, wav2vec2_feat], dim=-1)  # [B, 768]

            if self._use_mamba and self.mamba is not None:
                # Expand global feat across T frames for Mamba temporal encoding
                T = logmel.shape[2]  # number of mel time-frames
                # Project mel conv features back to frame-level
                mel_frames = self.mel_branch.conv(logmel).permute(0, 2, 1)  # [B, T, 128]
                # Tile global SSL features across frames
                wavlm_tiled    = wavlm_feat.unsqueeze(1).expand(-1, T, -1)       # [B, T, hidden]
                wav2vec2_tiled = wav2vec2_feat.unsqueeze(1).expand(-1, T, -1)    # [B, T, hidden]
                # Pad mel_frames to match fusion_in (256 → 256 already, need 768 total)
                mel_pad = torch.zeros(mel_frames.shape[0], T, hidden, device=mel_frames.device)
                mel_frames_padded = torch.cat([mel_frames, mel_pad], dim=-1)  # [B, T, 128+hidden]
                # Concat all streams
                frame_feats = torch.cat([
                    mel_frames_padded,       # [B, T, 128+hidden]
                    wavlm_tiled,             # [B, T, hidden]
                    wav2vec2_tiled,          # [B, T, hidden]
                ], dim=-1)  # aim: [B, T, 768] — ensure dims match fusion_in
                # Adjust to fusion_in size via linear if needed
                if frame_feats.shape[-1] != global_feat.shape[-1]:
                    # Fast fix: use only wavlm+wav2vec2+mel_frames[:128]
                    frame_feats = torch.cat([
                        mel_frames,       # [B, T, 128]
                        wavlm_tiled,      # [B, T, 256]
                        wav2vec2_tiled,   # [B, T, 256]
                    ], dim=-1)           # [B, T, 640] — close enough; Mamba handles dim
                    # Pad/trim to fusion_in
                    if frame_feats.shape[-1] < global_feat.shape[-1]:
                        pad = torch.zeros(frame_feats.shape[0], T,
                                         global_feat.shape[-1] - frame_feats.shape[-1],
                                         device=frame_feats.device)
                        frame_feats = torch.cat([frame_feats, pad], dim=-1)
                    else:
                        frame_feats = frame_feats[:, :, :global_feat.shape[-1]]

                # Mamba temporal encoding
                frame_feats = self.mamba(frame_feats)   # [B, T, fusion_in]
                global_feat = frame_feats.mean(dim=1)   # [B, fusion_in] — Mamba-pooled
            else:
                # Without Mamba, use simple tiling for frame features
                T = logmel.shape[2]
                mel_frames = self.mel_branch.conv(logmel).permute(0, 2, 1)  # [B, T, 128]
                wavlm_tiled    = wavlm_feat.unsqueeze(1).expand(-1, T, -1)
                wav2vec2_tiled = wav2vec2_feat.unsqueeze(1).expand(-1, T, -1)
                # Pad mel to match
                pad_size = global_feat.shape[-1] - 128 - hidden - hidden
                if pad_size > 0:
                    mel_pad = torch.zeros(mel_frames.shape[0], T, pad_size, device=mel_frames.device)
                    frame_feats = torch.cat([mel_frames, mel_pad, wavlm_tiled, wav2vec2_tiled], dim=-1)
                else:
                    frame_feats = torch.cat([mel_frames, wavlm_tiled, wav2vec2_tiled], dim=-1)
                frame_feats = frame_feats[:, :, :global_feat.shape[-1]]

            return global_feat, frame_feats

        def forward(
            self,
            logmel:        "torch.Tensor",
            ssl_embedding: "torch.Tensor",
            wav2vec2_embedding: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """
            Forward pass — global spoof logit.

            Args:
                logmel:             [B, n_mels, T_frames]
                ssl_embedding:      [B, wavlm_dim]  (WavLM embedding)
                wav2vec2_embedding: [B, wav2vec2_dim] (optional; zeros if not provided)

            Returns:
                global_logit: [B, 1]
            """
            if wav2vec2_embedding is None:
                wav2vec2_embedding = torch.zeros(
                    logmel.shape[0], wav2vec2_dim, device=logmel.device)

            global_feat, _ = self._encode(logmel, ssl_embedding, wav2vec2_embedding)
            return self.global_head(global_feat)

        def forward_with_frames(
            self,
            logmel:             "torch.Tensor",
            ssl_embedding:      "torch.Tensor",
            wav2vec2_embedding: "torch.Tensor | None" = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Forward pass — returns both global logit and per-frame logits.

            Returns:
                global_logit: [B, 1]
                frame_logits: [B, T, 1]
            """
            if wav2vec2_embedding is None:
                wav2vec2_embedding = torch.zeros(
                    logmel.shape[0], wav2vec2_dim, device=logmel.device)

            global_feat, frame_feats = self._encode(logmel, ssl_embedding, wav2vec2_embedding)
            global_logit = self.global_head(global_feat)    # [B, 1]
            frame_logits = self.frame_head(frame_feats)      # [B, T, 1]
            return global_logit, frame_logits

        def predict_proba(
            self,
            logmel:             "torch.Tensor",
            ssl_embedding:      "torch.Tensor",
            wav2vec2_embedding: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """Returns sigmoid global probability [0, 1]."""
            logit = self.forward(logmel, ssl_embedding, wav2vec2_embedding)
            return torch.sigmoid(logit)

        def predict_with_frames(
            self,
            logmel:             "torch.Tensor",
            ssl_embedding:      "torch.Tensor",
            wav2vec2_embedding: "torch.Tensor | None" = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Returns (global_prob [B,1], frame_probs [B,T,1]) in [0,1]."""
            global_logit, frame_logits = self.forward_with_frames(
                logmel, ssl_embedding, wav2vec2_embedding)
            return torch.sigmoid(global_logit), torch.sigmoid(frame_logits)

    return CMClassifier()


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------
class CMScorerWrapper:
    """
    Stateless inference wrapper used by detect.py and the streaming server.

    Backward-compatible: accepts features dict with optional 'wav2vec2_embedding'.

    Usage:
        scorer = CMScorerWrapper(checkpoint_path="models/cm.pt")
        result = scorer.score(features_dict)         # float in [0, 1]
        result = scorer.score_with_frames(features)  # (global, frame_list)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device:          Optional[str]         = None,
        use_mamba:       bool                  = True,
    ):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_cm_classifier(use_mamba=use_mamba).to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            logger.info("CM v2 checkpoint loaded: %s", checkpoint_path)
        else:
            if checkpoint_path:
                logger.warning(
                    "CM checkpoint not found at %s; using random weights. "
                    "Run training/train_cm.py to train the model.",
                    checkpoint_path,
                )

    def score(self, features: Dict[str, "torch.Tensor"]) -> float:
        """
        Score a feature dict → global spoof probability float in [0.0, 1.0].
        Backward-compatible: wav2vec2_embedding is optional.
        """
        import torch
        logmel     = features["logmel"].unsqueeze(0).to(self.device)       # [1, 80, T]
        wavlm_emb  = features["ssl_embedding"].unsqueeze(0).to(self.device)  # [1, 768]
        wav2v_emb  = features.get("wav2vec2_embedding")
        if wav2v_emb is not None:
            wav2v_emb = wav2v_emb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model.predict_proba(logmel, wavlm_emb, wav2v_emb)
        return float(prob.squeeze().cpu())

    def score_with_frames(
        self, features: Dict[str, "torch.Tensor"]
    ) -> Tuple[float, list]:
        """
        Returns (global_score: float, frame_scores: list[float]).
        frame_scores contains per-frame spoof probability (one per mel time-frame).
        """
        import torch
        logmel     = features["logmel"].unsqueeze(0).to(self.device)
        wavlm_emb  = features["ssl_embedding"].unsqueeze(0).to(self.device)
        wav2v_emb  = features.get("wav2vec2_embedding")
        if wav2v_emb is not None:
            wav2v_emb = wav2v_emb.unsqueeze(0).to(self.device)

        with torch.no_grad():
            global_prob, frame_probs = self.model.predict_with_frames(
                logmel, wavlm_emb, wav2v_emb)

        return (
            float(global_prob.squeeze().cpu()),
            frame_probs.squeeze(-1).squeeze(0).cpu().tolist(),
        )
