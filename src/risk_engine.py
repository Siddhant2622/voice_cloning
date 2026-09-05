"""
src/risk_engine.py
Layers 6 & 7 — Contextual risk engine and decision/prevention policy.

STUB: This module implements the interfaces described in the architecture.
Real integration points are clearly marked with # INTEGRATION_POINT comments.
In production, these would connect to:
  - Telephony metadata enrichment services
  - Transaction/CRM APIs
  - Real-time alerting infrastructure (PagerDuty, Slack webhooks, etc.)

Architecture reference: Layers 6 (contextual risk) and 7 (decision & prevention)
in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------
class Decision(str, Enum):
    ALLOW   = "ALLOW"      # Low risk — proceed normally
    STEP_UP = "STEP_UP"    # Medium risk — request additional verification
    ALERT   = "ALERT"      # High risk — notify human agent in real time
    BLOCK   = "BLOCK"      # Very high confidence attack — terminate session


# ---------------------------------------------------------------------------
# Risk context dataclass
# ---------------------------------------------------------------------------
@dataclass
class RiskContext:
    """
    Non-audio contextual signals that augment the acoustic risk score.
    All fields are optional; missing fields are treated as neutral.

    Architecture reference: Layer 6 — contextual risk signals.
    """
    # Audio-side
    fusion_score:          float = 0.5        # from fusion.py [0, 1]

    # Call metadata                           # INTEGRATION_POINT: Telephony API / caller-ID SIP headers
    caller_id_spoofed:     bool  = False      # True if CLI/ANI appears spoofed
    geolocation_mismatch:  bool  = False      # True if call origin ≠ account's home region
    sim_swap_flagged:      bool  = False      # True if telecom SIM swap detected in last 24 h
    number_reputation_bad: bool  = False      # True if number is on a fraud blocklist

    # Transaction context                     # INTEGRATION_POINT: CRM / transaction API
    transaction_value_usd: float = 0.0        # Dollar value of requested action
    action_type:           str   = "support"  # e.g. "wire_transfer", "password_reset", "support"

    # Historical / behavioural                # INTEGRATION_POINT: account-history service
    unusual_time:          bool  = False      # True if call time is outside normal pattern
    unusual_request:       bool  = False      # True if requested action is abnormal for this caller

    # Session metadata
    session_id:            str   = ""
    caller_id:             str   = ""


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------
@dataclass
class PolicyDecision:
    action:              Decision
    reason:              str
    contributing_signals: Dict[str, Any] = field(default_factory=dict)
    final_risk_score:    float = 0.0


class PolicyEngine:
    """
    Maps a RiskContext → PolicyDecision using configurable thresholds.

    Default thresholds (can be overridden via config/policy.yaml):
      - BLOCK   if adjusted_score ≥ 0.88
      - ALERT   if adjusted_score ≥ 0.70
      - STEP_UP if adjusted_score ≥ 0.45
      - ALLOW   otherwise

    The adjusted score blends the acoustic fusion score with boolean
    context signals via additive penalties.
    """

    DEFAULT_CONFIG = {
        "thresholds": {
            "block":   0.88,
            "alert":   0.70,
            "step_up": 0.45,
        },
        "context_penalties": {
            "caller_id_spoofed":     0.10,
            "geolocation_mismatch":  0.06,
            "sim_swap_flagged":      0.12,
            "number_reputation_bad": 0.08,
            "unusual_time":          0.04,
            "unusual_request":       0.05,
        },
        "transaction_risk_tiers": {
            # Above these USD values, add extra penalty
            1_000:  0.03,
            10_000: 0.07,
            50_000: 0.12,
        },
        "high_risk_actions": {
            "wire_transfer":  0.10,
            "password_reset": 0.05,
            "account_takeover": 0.15,
        },
    }

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                override = yaml.safe_load(f)
            self.config.update(override)
            logger.info("Policy config loaded: %s", config_path)

    def _compute_adjusted_score(self, ctx: RiskContext) -> tuple[float, Dict[str, Any]]:
        """Add context penalties to the acoustic fusion score."""
        score = ctx.fusion_score
        signals: Dict[str, Any] = {"acoustic_fusion_score": round(ctx.fusion_score, 4)}
        penalties = self.config.get("context_penalties", {})

        for flag, penalty in penalties.items():
            if getattr(ctx, flag, False):
                score += penalty
                signals[flag] = f"+{penalty:.2f} penalty"

        # Transaction value tier
        txn_tiers = self.config.get("transaction_risk_tiers", {})
        txn_penalty = 0.0
        for threshold, penalty in sorted(txn_tiers.items()):
            if ctx.transaction_value_usd >= threshold:
                txn_penalty = penalty
        if txn_penalty > 0:
            score += txn_penalty
            signals["transaction_value_penalty"] = f"+{txn_penalty:.2f} (${ctx.transaction_value_usd:,.0f})"

        # Action type risk
        action_risks = self.config.get("high_risk_actions", {})
        action_penalty = action_risks.get(ctx.action_type, 0.0)
        if action_penalty > 0:
            score += action_penalty
            signals["action_type_penalty"] = f"+{action_penalty:.2f} ({ctx.action_type})"

        score = float(min(1.0, score))
        signals["adjusted_score"] = round(score, 4)
        return score, signals

    def decide(self, ctx: RiskContext) -> PolicyDecision:
        """
        Main decision method.

        Returns:
            PolicyDecision with action, human-readable reason, and signals.
        """
        thresholds = self.config.get("thresholds", self.DEFAULT_CONFIG["thresholds"])
        adjusted, signals = self._compute_adjusted_score(ctx)

        if adjusted >= thresholds["block"]:
            action = Decision.BLOCK
            reason = (
                f"High-confidence synthetic speech detected (score={adjusted:.2f}). "
                "Session terminated. Account holder will be notified out-of-band."
            )
        elif adjusted >= thresholds["alert"]:
            action = Decision.ALERT
            reason = (
                f"Likely synthetic speech detected (score={adjusted:.2f}). "
                "Human fraud agent alerted in real time."
            )
        elif adjusted >= thresholds["step_up"]:
            action = Decision.STEP_UP
            reason = (
                f"Suspicious acoustic patterns detected (score={adjusted:.2f}). "
                "Additional verification required."
            )
        else:
            action = Decision.ALLOW
            reason = f"Audio appears genuine (score={adjusted:.2f}). Proceeding normally."

        logger.info(
            "Decision: %s | score=%.3f | session=%s",
            action.value,
            adjusted,
            ctx.session_id,
        )
        return PolicyDecision(
            action=action,
            reason=reason,
            contributing_signals=signals,
            final_risk_score=adjusted,
        )
