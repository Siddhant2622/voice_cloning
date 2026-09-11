"""
src/prevention.py
Prevention Engine — converts a PolicyDecision into a concrete enforcement action.

Action matrix:
  ALLOW   → return PreventionResult(action="ALLOW")
  STEP_UP → return PreventionResult(action="STEP_UP", challenge=<Challenge>)
  ALERT   → log + return PreventionResult(action="ALERT")
  BLOCK   → raise BlockedSessionError (caller must close the session/WebSocket)

Design decisions:
  - BLOCK is implemented as an exception so WS handlers cannot accidentally
    continue processing audio after a block decision.
  - STEP_UP embeds a fully-formed Challenge object so the caller can immediately
    serialise it to the client without a second call.
  - PreventionResult is a plain dataclass (not a Pydantic model) so it can be
    used in both FastAPI and plain async code without import overhead.

Architecture reference: Layer 7 (decision & prevention) in
voice-clone-detection-architecture.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from src.challenge import Challenge, ChallengeGenerator
from src.risk_engine import Decision, PolicyDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BlockedSessionError(Exception):
    """
    Raised by PreventionEngine.apply() when the risk decision is BLOCK.

    Attributes:
        reason:      Human-readable explanation from the PolicyDecision.
        risk_score:  Final adjusted risk score (0–1).
        session_id:  Caller / session identifier (optional).
    """

    def __init__(self, reason: str, risk_score: float, session_id: str = ""):
        super().__init__(reason)
        self.reason     = reason
        self.risk_score = risk_score
        self.session_id = session_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":       "blocked",
            "reason":     self.reason,
            "risk_score": round(self.risk_score, 4),
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# Prevention result
# ---------------------------------------------------------------------------
@dataclass
class PreventionResult:
    """
    Serialisable result of a prevention action.

    Returned for ALLOW / STEP_UP / ALERT.
    BLOCK never returns — it raises BlockedSessionError instead.
    """
    action:      str                           # "ALLOW" | "STEP_UP" | "ALERT"
    reason:      str      = ""
    risk_score:  float    = 0.0
    challenge:   Optional[Challenge] = None   # populated for STEP_UP
    session_id:  str      = ""
    signals:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type":       self.action.lower(),
            "action":     self.action,
            "reason":     self.reason,
            "risk_score": round(self.risk_score, 4),
            "session_id": self.session_id,
            "signals":    self.signals,
        }
        if self.challenge is not None:
            d["challenge"] = {
                "challenge_id": self.challenge.challenge_id,
                "phrase":       self.challenge.phrase,
                "digits":       self.challenge.digits,
                "expires_in_s": self.challenge.expires_in_s,
            }
        return d


# ---------------------------------------------------------------------------
# Prevention engine
# ---------------------------------------------------------------------------
class PreventionEngine:
    """
    Maps a PolicyDecision to a concrete prevention action.

    Usage (in a WebSocket handler)::

        prevention = PreventionEngine()
        try:
            result = prevention.apply(decision, session_id=session_id)
            await ws.send_json(result.to_dict())
        except BlockedSessionError as e:
            await ws.send_json(e.to_dict())
            await ws.close(code=1008, reason=e.reason[:123])
            return          # ← stop processing this session
    """

    def __init__(self, challenge_mode: str = "digits"):
        """
        Args:
            challenge_mode: "digits" or "phrase" — type of STEP_UP challenge.
        """
        self._challenge_mode = challenge_mode
        self._challenge_gen  = ChallengeGenerator()

    def apply(
        self,
        decision:   PolicyDecision,
        session_id: str = "",
    ) -> PreventionResult:
        """
        Apply the policy decision.

        Args:
            decision:   Output of PolicyEngine.decide().
            session_id: For logging and response payload.

        Returns:
            PreventionResult for ALLOW / STEP_UP / ALERT.

        Raises:
            BlockedSessionError for BLOCK — caller must close the session.
        """
        action = decision.action
        score  = decision.final_risk_score
        reason = decision.reason

        if action == Decision.BLOCK:
            logger.warning(
                "BLOCK | session=%s score=%.4f | %s",
                session_id, score, reason,
            )
            raise BlockedSessionError(reason=reason, risk_score=score, session_id=session_id)

        if action == Decision.STEP_UP:
            challenge = self._challenge_gen.generate(mode=self._challenge_mode)
            logger.info(
                "STEP_UP | session=%s score=%.4f challenge=%s",
                session_id, score, challenge.challenge_id,
            )
            return PreventionResult(
                action=action.value,
                reason=reason,
                risk_score=score,
                challenge=challenge,
                session_id=session_id,
                signals=decision.contributing_signals,
            )

        if action == Decision.ALERT:
            logger.warning(
                "ALERT | session=%s score=%.4f | %s",
                session_id, score, reason,
            )
            # INTEGRATION_POINT: trigger PagerDuty / Slack webhook here
            return PreventionResult(
                action=action.value,
                reason=reason,
                risk_score=score,
                session_id=session_id,
                signals=decision.contributing_signals,
            )

        # ALLOW
        logger.debug("ALLOW | session=%s score=%.4f", session_id, score)
        return PreventionResult(
            action=action.value,
            reason=reason,
            risk_score=score,
            session_id=session_id,
            signals=decision.contributing_signals,
        )
