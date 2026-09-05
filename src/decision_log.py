"""
src/decision_log.py
Layer 9 — Structured decision audit log.

Every detection decision is persisted as a JSON-lines record containing:
  - Timestamp and session metadata
  - All individual signal scores (CM, SV, liveness, watermark)
  - Fusion score and final decision
  - Human-readable reason (explainability trail)

This is the audit trail described in Layer 9 of the architecture.
In production this would write to an append-only store (e.g. TimescaleDB,
a SIEM, or an encrypted S3-compatible object store).

PRIVACY NOTE: This log does NOT contain raw audio or voiceprints.
Only numerical scores and metadata are recorded.

Architecture reference: Layer 9 in voice-clone-detection-architecture.md
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs") / "decisions.jsonl"


# ---------------------------------------------------------------------------
# Log record schema
# ---------------------------------------------------------------------------
@dataclass
class DecisionEvent:
    """
    One logged detection decision.
    All score fields are in [0, 1]; None = signal unavailable.
    """
    # Identity
    session_id:           str
    timestamp:            str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    window_index:         int = 0        # which streaming window (0 = first)
    caller_id:            str = ""

    # Individual signal scores
    cm_score:             float = 0.5
    sv_score:             Optional[float] = None
    liveness_score:       Optional[float] = None
    flatness_score:       Optional[float] = None
    jitter_score:         Optional[float] = None
    contrast_score:       Optional[float] = None
    watermark_detected:   Optional[bool]  = None

    # Fusion output
    fusion_score:         float = 0.5
    final_risk_score:     float = 0.5    # post-context-adjustment

    # Decision
    action:               str = "ALLOW"
    reason:               str = ""
    contributing_signals: Dict[str, Any] = field(default_factory=dict)

    # Challenge (if active)
    challenge_issued:     bool = False
    challenge_passed:     Optional[bool] = None
    challenge_delta:      Optional[float] = None

    # Latency
    inference_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict, omitting None values for compactness."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class DecisionLogger:
    """
    Append-only structured decision logger.

    Usage:
        dlog = DecisionLogger()
        dlog.log(event)

    INTEGRATION_POINT: Replace the file-write backend with a real
    append-only store. Candidates:
      - TimescaleDB (hypertable on timestamp column)
      - AWS Kinesis Data Firehose → S3 → Athena
      - Azure Event Hubs → ADX
      - A SIEM (Splunk, Elastic, Datadog Logs)

    DATA GOVERNANCE:
      - Log files must be encrypted at rest (AES-256 minimum).
      - Retention period must comply with applicable regulations
        (GDPR Art. 5(1)(e), CCPA, BIPA as applicable).
      - Deletion requests must cascade to log records containing
        that caller's session data.
    """

    def __init__(self, log_path: Optional[str | Path] = None):
        self.log_path = Path(log_path or DEFAULT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: DecisionEvent) -> None:
        """Append a decision event as a JSON line."""
        record = event.to_dict()
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug(
                "Logged decision: session=%s action=%s score=%.3f",
                event.session_id,
                event.action,
                event.final_risk_score,
            )
        except Exception as exc:
            logger.error("Failed to write decision log: %s", exc)

    def read_recent(self, n: int = 20) -> list[Dict[str, Any]]:
        """Read the last N decision events (for dashboard / debugging)."""
        if not self.log_path.exists():
            return []
        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            logger.error("Failed to read decision log: %s", exc)
        return records[-n:]
