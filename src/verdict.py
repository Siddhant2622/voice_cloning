"""
src/verdict.py
AI Voice Detection Analyst — VerdictReport dataclass and rich renderer.

Converts a QualityLatencyResult (raw per-signal scores) into the
structured analyst verdict format:

    Verdict:      [Likely AI-generated / Likely human / Uncertain]
    Confidence:   [Low / Medium / High]
    Key evidence: 2–4 bullet points citing specific signals observed
    Caveats:      anything that could mislead the verdict

The verdict is computed by:
  1. Classifying the overall_score into verdict tier.
  2. Selecting the top-N most informative signals as key evidence bullets.
  3. Applying caveats based on ambiguous signal patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------
# overall_score ∈ [0, 1]: 0 = definitely human, 1 = definitely AI

LIKELY_AI_THRESHOLD   = 0.62   # above → "Likely AI-generated"
LIKELY_HUMAN_THRESHOLD = 0.35  # below → "Likely human"
# between → "Uncertain"

# Confidence thresholds (distance from the boundary)
HIGH_CONFIDENCE_MARGIN   = 0.18   # e.g. score > 0.80 or < 0.17 → High
MEDIUM_CONFIDENCE_MARGIN = 0.08   # moderate distance → Medium
# otherwise → Low


# ---------------------------------------------------------------------------
# VerdictReport
# ---------------------------------------------------------------------------

@dataclass
class VerdictReport:
    """
    Structured verdict from the AI Voice Detection Analyst.

    Attributes:
        verdict:       "Likely AI-generated" | "Likely human" | "Uncertain"
        confidence:    "High" | "Medium" | "Low"
        key_evidence:  2–4 bullet-point strings citing specific signals
        caveats:       list of strings noting potential misleading factors
        quality_score: aggregated audio-quality AI-likelihood score [0,1]
        latency_score: aggregated latency/timing AI-likelihood score [0,1]
        overall_score: final combined score [0,1] (0=human, 1=AI)
    """
    verdict: str
    confidence: str
    key_evidence: List[str]
    caveats: List[str]
    quality_score: float
    latency_score: float
    overall_score: float

    # Internal breakdown for display
    signal_scores: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Verdict:    {self.verdict}",
            f"Confidence: {self.confidence}",
            "",
            "Key evidence:",
        ]
        for bullet in self.key_evidence:
            lines.append(f"  * {bullet}")
        if self.caveats:
            lines.append("")
            lines.append("Caveats:")
            for c in self.caveats:
                lines.append(f"  [!] {c}")
        lines.append("")
        lines.append(f"  Quality score:  {self.quality_score:.3f}")
        lines.append(f"  Latency score:  {self.latency_score:.3f}")
        lines.append(f"  Overall score:  {self.overall_score:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "verdict":       self.verdict,
            "confidence":    self.confidence,
            "key_evidence":  self.key_evidence,
            "caveats":       self.caveats,
            "quality_score": round(self.quality_score, 4),
            "latency_score": round(self.latency_score, 4),
            "overall_score": round(self.overall_score, 4),
        }


# ---------------------------------------------------------------------------
# Verdict builder
# ---------------------------------------------------------------------------

def build_verdict(result: "QualityLatencyResult") -> VerdictReport:
    """
    Convert a QualityLatencyResult into a VerdictReport.

    Args:
        result: the raw per-signal analysis result.

    Returns:
        VerdictReport with verdict, confidence, key_evidence, caveats.
    """
    score = result.overall_score

    # ── Verdict tier ──────────────────────────────────────────────────────
    if score >= LIKELY_AI_THRESHOLD:
        verdict = "Likely AI-generated"
        distance_from_boundary = score - LIKELY_AI_THRESHOLD
    elif score <= LIKELY_HUMAN_THRESHOLD:
        verdict = "Likely human"
        distance_from_boundary = LIKELY_HUMAN_THRESHOLD - score
    else:
        verdict = "Uncertain"
        # Distance from nearest boundary
        distance_from_boundary = min(
            score - LIKELY_HUMAN_THRESHOLD,
            LIKELY_AI_THRESHOLD - score
        )

    # ── Confidence ────────────────────────────────────────────────────────
    if distance_from_boundary >= HIGH_CONFIDENCE_MARGIN:
        confidence = "High"
    elif distance_from_boundary >= MEDIUM_CONFIDENCE_MARGIN:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Collect all signals sorted by score descending ────────────────────
    all_signals: List[Tuple[str, float, str]] = []

    quality_map = result.quality_signals()
    latency_map = result.latency_signals()

    for name, (sig_score, detail) in quality_map.items():
        all_signals.append((name, sig_score, detail))
    for name, (sig_score, detail) in latency_map.items():
        all_signals.append((name, sig_score, detail))

    # Sort by score descending (most suspicious first)
    all_signals.sort(key=lambda x: x[1], reverse=True)

    # ── Key evidence: top 2–4 most informative signals ────────────────────
    key_evidence: List[str] = []

    for name, sig_score, detail in all_signals[:4]:
        if sig_score >= 0.50:
            direction = "elevated suspicion"
        elif sig_score >= 0.25:
            direction = "mild suspicion"
        else:
            direction = "consistent with human"

        pct = int(sig_score * 100)
        key_evidence.append(
            f"[{name}] score={pct}% — {detail} ({direction})"
        )

    # If top signal is actually human-like (score < 0.3), note that too
    if all_signals and all_signals[0][1] < 0.30:
        key_evidence = [
            f"[{name}] score={int(sig_score*100)}% — {detail} (human-like)"
            for name, sig_score, detail in all_signals[:4]
        ]

    # Ensure at least 2 bullets
    if len(key_evidence) < 2 and len(all_signals) >= 2:
        name, sig_score, detail = all_signals[1]
        key_evidence.append(
            f"[{name}] score={int(sig_score*100)}% — {detail}"
        )

    # ── Caveats ───────────────────────────────────────────────────────────
    caveats: List[str] = []

    # Caveat: scripted/teleprompter human
    if (verdict == "Likely AI-generated"
            and result.prosody_score >= 0.4
            and result.breath_score < 0.3):
        caveats.append(
            "A human reading from a teleprompter or script may show similar "
            "prosody smoothness without breath deficits — verify with additional signals."
        )

    # Caveat: deliberate filler injection
    if verdict in ("Likely AI-generated", "Uncertain") and result.self_correction_score >= 0.5:
        caveats.append(
            "Some conversational AI agents deliberately inject filler words (um, uh) "
            "and thinking pauses to mimic human disfluency — a clean transcript alone "
            "does not confirm TTS."
        )

    # Caveat: codec/phone compression
    if result.background_score >= 0.70:
        caveats.append(
            "Heavy audio compression (VoIP/phone codec, G.711/Opus) removes "
            "background noise and can make genuine human speech appear studio-clean; "
            "consider the recording channel before concluding AI origin."
        )

    # Caveat: no latency data
    if (result.response_time_score == 0.3
            and result.cadence_score == 0.30
            and result.turn_taking_score in (0.35, 0.30)):
        caveats.append(
            "No real-time timing data was provided — latency signals defaulted to "
            "neutral. Supplying session timing data (LatencyProfile) would significantly "
            "improve detection accuracy."
        )

    # Caveat: short audio
    if result.spectral_score == 0.3 or result.pronunciation_score == 0.3:
        caveats.append(
            "Some signals could not be computed due to short audio duration — "
            "longer samples (>5 s) yield more reliable per-signal estimates."
        )

    # ── Build signal_scores dict for UI rendering ─────────────────────────
    signal_scores = {
        "quality": {
            name: {"score": round(s, 4), "detail": d}
            for name, (s, d) in quality_map.items()
        },
        "latency": {
            name: {"score": round(s, 4), "detail": d}
            for name, (s, d) in latency_map.items()
        },
    }

    return VerdictReport(
        verdict=verdict,
        confidence=confidence,
        key_evidence=key_evidence,
        caveats=caveats,
        quality_score=round(result.quality_aggregate, 4),
        latency_score=round(result.latency_aggregate, 4),
        overall_score=round(result.overall_score, 4),
        signal_scores=signal_scores,
    )


# ---------------------------------------------------------------------------
# Rich console renderer
# ---------------------------------------------------------------------------

def render_verdict_rich(report: VerdictReport) -> None:
    """
    Print a beautifully formatted verdict report to the terminal using rich.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.text import Text

    console = Console()

    # Verdict color
    verdict_colors = {
        "Likely AI-generated": "bright_red",
        "Likely human":        "bright_green",
        "Uncertain":           "yellow",
    }
    confidence_colors = {
        "High":   "bright_white",
        "Medium": "yellow",
        "Low":    "dim",
    }
    vc = verdict_colors.get(report.verdict, "white")
    cc = confidence_colors.get(report.confidence, "white")

    # ── Header panel ──────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold {vc}]{report.verdict}[/]   "
        f"[{cc}]Confidence: {report.confidence}[/]\n"
        f"[dim]Quality={report.quality_score:.3f}  "
        f"Latency={report.latency_score:.3f}  "
        f"Overall={report.overall_score:.3f}[/]",
        title="[bold]🔍 AI Voice Analyst — Verdict",
        border_style=vc,
        expand=False,
    ))

    # ── Quality signals table ─────────────────────────────────────────────
    q_table = Table(
        title="Audio Quality Signals",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    q_table.add_column("Signal",  style="bold", width=22)
    q_table.add_column("Score",   justify="right", width=8)
    q_table.add_column("Bar",     width=20)
    q_table.add_column("Detail",  style="dim")

    def _score_bar(s: float, width: int = 18) -> str:
        filled = int(s * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def _score_color(s: float) -> str:
        if s < 0.30:
            return "bright_green"
        if s < 0.55:
            return "yellow"
        if s < 0.75:
            return "bright_yellow"
        return "bright_red"

    for name, (s, d) in report.signal_scores.get("quality", {}).items():
        col = _score_color(s)
        q_table.add_row(
            name,
            Text(f"{s:.3f}", style=col),
            Text(_score_bar(s), style=col),
            d,
        )
    console.print(q_table)

    # ── Latency signals table ─────────────────────────────────────────────
    l_table = Table(
        title="Latency / Timing Signals",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    l_table.add_column("Signal",  style="bold", width=22)
    l_table.add_column("Score",   justify="right", width=8)
    l_table.add_column("Bar",     width=20)
    l_table.add_column("Detail",  style="dim")

    for name, (s, d) in report.signal_scores.get("latency", {}).items():
        col = _score_color(s)
        l_table.add_row(
            name,
            Text(f"{s:.3f}", style=col),
            Text(_score_bar(s), style=col),
            d,
        )
    console.print(l_table)

    # ── Key evidence ──────────────────────────────────────────────────────
    ev_text = "\n".join(f"  • {e}" for e in report.key_evidence)
    console.print(Panel(
        ev_text,
        title="[bold]Key Evidence",
        border_style="cyan",
        expand=False,
    ))

    # ── Caveats ───────────────────────────────────────────────────────────
    if report.caveats:
        cv_text = "\n".join(f"  ⚠  {c}" for c in report.caveats)
        console.print(Panel(
            cv_text,
            title="[bold yellow]Caveats",
            border_style="yellow",
            expand=False,
        ))
