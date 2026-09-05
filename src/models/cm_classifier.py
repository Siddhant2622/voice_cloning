"""
src/models/cm_classifier.py
Layer 4a — Countermeasure (CM) classifier: bona fide vs. synthetic scoring.

Architecture:
  Input: log-mel spectrogram [n_mels=80, T_frames] + WavLM SSL embedding [768]
  ├── Log-mel branch: attentive statistics pooling → frame-level features
  └── SSL branch: direct MLP
  → Fusion MLP → sigmoid spoof score ∈ [0, 1]

  0 = definitely bona fide (genuine human speech)
  1 = definitely synthetic / spoofed

Architecture reference: Layer 4a, AASIST-lite variant using SSL front-end
(as described in voice-clone-detection-architecture.md)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

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
    For each frame position, compute an attention weight via a small MLP,
    then output weighted mean + weighted std as the pooled representation.

    Input:  [batch, T, C]
    Output: [batch, 2*C]
    """

    def __new__(cls, *args, **kwargs):
        torch = _torch()
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
                attn = self.attention(x)          # [B, T, 1]
                attn = torch.softmax(attn, dim=1) # normalise over time
                mean = (attn * x).sum(dim=1)      # [B, C]
                var  = (attn * x.pow(2)).sum(dim=1) - mean.pow(2)
                std  = (var.clamp(min=1e-8)).sqrt()
                return torch.cat([mean, std], dim=-1)  # [B, 2C]

        return _ASP(*args, **kwargs)


# ---------------------------------------------------------------------------
# CM Classifier network
# ---------------------------------------------------------------------------
def build_cm_classifier(logmel_n_mels: int = 80, ssl_dim: int = 768, hidden: int = 256):
    """
    Factory function (avoids torch import at module load time).
    Returns an nn.Module.
    """
    torch = _torch()
    import torch.nn as nn

    class _LogMelBranch(nn.Module):
        """1-D temporal CNN + attentive statistics pooling over mel frames."""

        def __init__(self):
            super().__init__()
            # Lightweight 1-D convolutions along the time axis
            self.conv = nn.Sequential(
                nn.Conv1d(logmel_n_mels, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
            )
            # Attentive statistics pooling
            self.asp = AttentiveStatisticsPooling(128)
            self.out_dim = 256   # 2 * 128

        def forward(self, logmel):
            # logmel: [B, n_mels, T]
            x = self.conv(logmel)      # [B, 128, T]
            x = x.permute(0, 2, 1)    # [B, T, 128]
            x = self.asp(x)            # [B, 256]
            return x

    class _SSLBranch(nn.Module):
        """Simple projection of the SSL embedding."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(ssl_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            self.out_dim = hidden

        def forward(self, ssl_emb):
            # ssl_emb: [B, ssl_dim]
            return self.proj(ssl_emb)

    class CMClassifier(nn.Module):
        """
        Full countermeasure classifier.
        Fuses log-mel branch + SSL branch → spoof probability.
        """

        def __init__(self):
            super().__init__()
            self.mel_branch = _LogMelBranch()
            self.ssl_branch = _SSLBranch()
            fusion_in = self.mel_branch.out_dim + self.ssl_branch.out_dim

            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(
            self,
            logmel: "torch.Tensor",
            ssl_embedding: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Args:
                logmel:        [B, n_mels, T_frames]
                ssl_embedding: [B, ssl_dim]

            Returns:
                spoof_logit: [B, 1]  (apply sigmoid for probability)
            """
            mel_feat = self.mel_branch(logmel)
            ssl_feat = self.ssl_branch(ssl_embedding)
            fused = torch.cat([mel_feat, ssl_feat], dim=-1)
            return self.fusion_mlp(fused)

        def predict_proba(
            self,
            logmel: "torch.Tensor",
            ssl_embedding: "torch.Tensor",
        ) -> "torch.Tensor":
            """Convenience: returns sigmoid probability in [0, 1]."""
            logit = self.forward(logmel, ssl_embedding)
            return torch.sigmoid(logit)

    return CMClassifier()


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------
class CMScorerWrapper:
    """
    Stateless inference wrapper used by detect.py and the streaming server.

    Usage:
        scorer = CMScorerWrapper(checkpoint_path="models/cm.pt")
        score = scorer.score(features_dict)  # returns float in [0, 1]
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ):
        torch = _torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_cm_classifier().to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info("CM checkpoint loaded: %s", checkpoint_path)
        else:
            if checkpoint_path:
                logger.warning(
                    "CM checkpoint not found at %s; using random weights. "
                    "Run training/train_cm.py to train the model.",
                    checkpoint_path,
                )
            else:
                logger.warning(
                    "No CM checkpoint provided; using random weights. "
                    "Run training/train_cm.py to train the model."
                )

    def score(self, features: Dict[str, "torch.Tensor"]) -> float:
        """
        Score a feature dict (as returned by src/features.extract_all).

        Returns:
            spoof_probability: float in [0.0, 1.0]
            (0 = bona fide, 1 = synthetic)
        """
        torch = _torch()
        logmel    = features["logmel"].unsqueeze(0).to(self.device)       # [1, 80, T]
        ssl_emb   = features["ssl_embedding"].unsqueeze(0).to(self.device) # [1, 768]

        with torch.no_grad():
            prob = self.model.predict_proba(logmel, ssl_emb)
        return float(prob.squeeze().cpu())
