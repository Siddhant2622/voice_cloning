"""
src/fusion.py
Layer 5 — Calibrated score fusion.

Combines independent CM, speaker-verification, and liveness signals
into one calibrated risk probability via logistic regression.

Design decisions:
  - Logistic regression is interpretable, fast, and doesn't overfit
    on small validation sets (unlike a deep fusion head at this scale).
  - Missing signals (e.g. no enrollment → no SV score) are imputed
    with a neutral 0.5 value so the fusion layer always receives a
    complete feature vector.
  - Calibration is performed via Platt scaling (logistic regression
    on held-out scores), aligning with ISO/IEC 30107-3:2023 guidance.

Architecture reference: Layer 5 in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

NEUTRAL_SCORE = 0.5    # used when a signal is unavailable


# ---------------------------------------------------------------------------
# Score bundle dataclass
# ---------------------------------------------------------------------------
@dataclass
class ScoreBundle:
    """
    All signals from the detection ensemble, before fusion.
    Any field can be None if the corresponding pillar was unavailable.
    """
    cm_score:          float                  # CM classifier spoof prob [0,1]
    sv_score:          Optional[float] = None # Speaker verification spoof prob [0,1]
    liveness_score:    Optional[float] = None # Liveness suspicion score [0,1]
    # Sub-scores from liveness
    flatness_score:    Optional[float] = None
    jitter_score:      Optional[float] = None
    contrast_score:    Optional[float] = None
    # Watermark signal (Layer 2)
    watermark_detected: Optional[bool] = None

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to a fixed-length numpy feature vector for the fusion model.
        Missing values are imputed with NEUTRAL_SCORE.
        """
        sv  = self.sv_score        if self.sv_score       is not None else NEUTRAL_SCORE
        lv  = self.liveness_score  if self.liveness_score is not None else NEUTRAL_SCORE
        wm  = 1.0 if self.watermark_detected else 0.0
        return np.array([self.cm_score, sv, lv, wm], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------
class FusionModel:
    """
    Logistic regression over [cm_score, sv_score, liveness_score, watermark].

    Training:
        fm = FusionModel()
        fm.fit(X_train, y_train)   # X: list of ScoreBundle, y: 0/1 labels
        fm.save("models/fusion.pkl")

    Inference:
        fm = FusionModel.load("models/fusion.pkl")
        prob = fm.score(bundle)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # CalibratedClassifierCV wraps LR with Platt scaling calibration
        self._base_lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )
        self.model: Optional[CalibratedClassifierCV] = None
        self._trained = False

    # --- Rule-based fallback (used before training) --------------------------

    @staticmethod
    def _heuristic_score(bundle: ScoreBundle) -> float:
        """
        Weighted average of available signals, used when the LR model
        hasn't been trained yet. Calibrated to avoid false alarms on real human speech
        while strongly penalizing synthetic acoustic artifacts.
        """
        # Base weights: balanced between CM classifier and physical liveness
        cm_w = 0.45
        lv_w = 0.35
        sv_w = 0.20 if bundle.sv_score is not None else 0.0

        if bundle.liveness_score is not None:
            total_w = cm_w + lv_w + sv_w
            score = (
                cm_w * bundle.cm_score
                + sv_w * (bundle.sv_score if bundle.sv_score is not None else 0.0)
                + lv_w * bundle.liveness_score
            ) / total_w

            # Biological liveness protection:
            # If CM score is low-to-moderate (< 0.70) and liveness is clearly human (< 0.25),
            # protect genuine speakers from false positives due to room acoustics.
            if bundle.cm_score < 0.70 and bundle.liveness_score < 0.25:
                human_confidence = 1.0 - (bundle.liveness_score / 0.25)
                score = score * (1.0 - 0.25 * human_confidence)

            # High-confidence attack override:
            # When the deep neural CM classifier detects synthetic speech with high confidence (>= 0.75),
            # passive liveness must not suppress the detection (critical for neural TTS like Gemini Live).
            if bundle.cm_score >= 0.75:
                score = max(score, bundle.cm_score * 0.92)
        elif bundle.sv_score is not None:
            total_w = cm_w + sv_w
            score = (cm_w * bundle.cm_score + sv_w * bundle.sv_score) / total_w
        else:
            score = bundle.cm_score

        # Watermark: hard override — if detected, score -> 0.97
        if bundle.watermark_detected:
            score = max(score, 0.97)

        return float(np.clip(score, 0.0, 1.0))

    # --- Training ------------------------------------------------------------

    def fit(
        self,
        bundles: list[ScoreBundle],
        labels: list[int],           # 0 = bona fide, 1 = spoof
        calibrate: bool = True,
    ) -> "FusionModel":
        X = np.array([b.to_feature_vector() for b in bundles])
        y = np.array(labels, dtype=int)

        X_scaled = self.scaler.fit_transform(X)

        if calibrate:
            self.model = CalibratedClassifierCV(
                self._base_lr, method="sigmoid", cv=5
            )
        else:
            self.model = self._base_lr

        self.model.fit(X_scaled, y)
        self._trained = True
        logger.info(
            "Fusion model trained on %d samples (calibrated=%s).",
            len(y),
            calibrate,
        )
        return self

    # --- Inference -----------------------------------------------------------

    def score(self, bundle: ScoreBundle) -> float:
        """
        Return calibrated spoof probability ∈ [0, 1].
        Falls back to heuristic if model not trained.
        """
        if not self._trained or self.model is None:
            logger.debug("Fusion model not trained; using heuristic fallback.")
            return self._heuristic_score(bundle)

        x = bundle.to_feature_vector().reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        prob = self.model.predict_proba(x_scaled)[0, 1]   # P(spoof)
        return float(prob)

    # --- Persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model, "trained": self._trained}, f)
        logger.info("Fusion model saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FusionModel":
        fm = cls()
        with open(path, "rb") as f:
            state = pickle.load(f)
        fm.scaler   = state["scaler"]
        fm.model    = state["model"]
        fm._trained = state.get("trained", True)
        logger.info("Fusion model loaded: %s", path)
        return fm


# ---------------------------------------------------------------------------
# EER / evaluation utilities
# ---------------------------------------------------------------------------
def compute_eer(genuine_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) from score arrays.

    Args:
        genuine_scores: spoof probability scores for bona fide samples (want low)
        spoof_scores:   spoof probability scores for spoof samples (want high)

    Returns:
        eer: float (e.g. 0.05 = 5% EER)
    """
    from sklearn.metrics import roc_curve

    all_scores = np.concatenate([genuine_scores, spoof_scores])
    all_labels = np.concatenate([
        np.zeros(len(genuine_scores)),
        np.ones(len(spoof_scores))
    ])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr

    # EER is where FPR ≈ FNR
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return eer


def compute_apcer_bpcer(
    genuine_scores: np.ndarray,
    spoof_scores:   np.ndarray,
    threshold:      float,
) -> tuple[float, float]:
    """
    Compute APCER and BPCER at a given decision threshold.
    (ISO/IEC 30107-3:2023 metrics)

    APCER (Attack Presentation Classification Error Rate):
        Proportion of spoof samples classified as bona fide (missed attacks).

    BPCER (Bona fide Presentation Classification Error Rate):
        Proportion of genuine samples classified as spoof (false alarms).

    Args:
        genuine_scores: spoof probabilities for genuine callers
        spoof_scores:   spoof probabilities for attackers
        threshold: decision boundary (score > threshold → classify as spoof)

    Returns:
        (apcer, bpcer): floats
    """
    apcer = float(np.mean(spoof_scores <= threshold))    # attacks that slipped through
    bpcer = float(np.mean(genuine_scores > threshold))   # genuine callers wrongly flagged
    return apcer, bpcer
