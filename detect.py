"""
detect.py — Phase 1 CLI entry point.

Usage:
    python detect.py <audio_file> [--model <checkpoint>] [--enroll <ref_audio>] [--verbose]

Output (JSON to stdout):
    {
        "file": "audio.wav",
        "spoof_score": 0.87,
        "label": "SYNTHETIC",
        "cm_score": 0.82,
        "liveness_score": 0.71,
        "sv_score": null,
        "fusion_score": 0.87,
        "decision": "ALERT",
        "reason": "...",
        "latency_ms": 142.3
    }

Exit codes:
    0 — ALLOW (bona fide)
    1 — STEP_UP
    2 — ALERT
    3 — BLOCK
    4 — error
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voice Clone Detector — score a single audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_file", help="Path to audio file (WAV, FLAC, MP3, OGG)")
    parser.add_argument(
        "--model", default=None,
        help="Path to CM classifier checkpoint (.pt). Defaults to models/cm.pt if it exists."
    )
    parser.add_argument(
        "--fusion-model", default=None,
        help="Path to fusion model checkpoint (.pkl). Defaults to models/fusion.pkl."
    )
    parser.add_argument(
        "--enroll", default=None,
        help="Path to reference audio for speaker verification enrollment."
    )
    parser.add_argument(
        "--policy", default="config/policy.yaml",
        help="Path to policy config YAML. [default: config/policy.yaml]"
    )
    parser.add_argument(
        "--transaction-value", type=float, default=0.0,
        help="Transaction value in USD for risk-engine context. [default: 0]"
    )
    parser.add_argument(
        "--action-type", default="support",
        help="Action type string (support, wire_transfer, password_reset, …). [default: support]"
    )
    parser.add_argument(
        "--no-liveness", action="store_true",
        help="Skip liveness analysis (faster)."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of rich formatted output."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging."
    )
    return parser.parse_args()


def _score_color(score: float) -> str:
    if score < 0.35:
        return "bright_green"
    if score < 0.65:
        return "yellow"
    if score < 0.85:
        return "bright_yellow"
    return "bright_red"


def _decision_color(decision: str) -> str:
    return {
        "ALLOW": "bright_green",
        "STEP_UP": "yellow",
        "ALERT": "bright_yellow",
        "BLOCK": "bright_red",
    }.get(decision, "white")


EXIT_CODES = {"ALLOW": 0, "STEP_UP": 1, "ALERT": 2, "BLOCK": 3}


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s %(levelname)s %(message)s",
    )

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        console.print(f"[bright_red]Error:[/] File not found: {audio_path}")
        return 4

    # ── Lazy imports (after arg validation so --help is fast) ─────────────
    try:
        import torch
    except ImportError:
        console.print(
            "[bright_red]PyTorch not installed.[/] "
            "Run: python -m pip install --pre torch torchaudio "
            "--index-url https://download.pytorch.org/whl/nightly/cu128"
        )
        return 4

    from src.preprocessing import preprocess_file
    from src.features import extract_all
    from src.models.cm_classifier import CMScorerWrapper
    from src.models.liveness import LivenessScorer
    from src.models.speaker_verify import SpeakerVerifier
    from src.fusion import FusionModel, ScoreBundle
    from src.risk_engine import PolicyEngine, RiskContext
    from src.decision_log import DecisionLogger, DecisionEvent
    import uuid

    t_start = time.perf_counter()

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    if not args.json:
        console.print(f"\n[bold]Analyzing:[/] {audio_path.name}", end="  ")
    waveform, sr = preprocess_file(audio_path)

    # ── Step 2: Feature extraction ─────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = extract_all(waveform, device=device)

    # ── Step 3: CM scoring ─────────────────────────────────────────────────
    cm_ckpt = args.model or (
        "models/cm.pt" if Path("models/cm.pt").exists() else None
    )
    cm_scorer = CMScorerWrapper(checkpoint_path=cm_ckpt, device=device)
    cm_score = cm_scorer.score(features)

    # ── Step 4: Liveness ──────────────────────────────────────────────────
    liveness_result = {}
    liveness_score = None
    if not args.no_liveness:
        try:
            lv = LivenessScorer()
            liveness_result = lv.score(waveform)
            liveness_score = liveness_result.get("liveness_score")
        except Exception as exc:
            logging.warning("Liveness analysis failed: %s", exc)

    # ── Step 5: Speaker verification ──────────────────────────────────────
    sv_score = None
    sv = SpeakerVerifier(device=device)
    if args.enroll and Path(args.enroll).exists():
        ref_wav, _ = preprocess_file(args.enroll)
        sv.enroll("caller", ref_wav)
        sv_score = sv.score_as_spoof_probability(waveform, speaker_id="caller")

    # ── Step 6: Fusion ────────────────────────────────────────────────────
    fusion_ckpt = args.fusion_model or (
        "models/fusion.pkl" if Path("models/fusion.pkl").exists() else None
    )
    fm = FusionModel()
    if fusion_ckpt and Path(fusion_ckpt).exists():
        fm = FusionModel.load(fusion_ckpt)

    bundle = ScoreBundle(
        cm_score=cm_score,
        sv_score=sv_score,
        liveness_score=liveness_score,
        flatness_score=liveness_result.get("flatness_score"),
        jitter_score=liveness_result.get("jitter_score"),
        contrast_score=liveness_result.get("contrast_score"),
    )
    fusion_score = fm.score(bundle)

    # ── Step 7: Risk & decision ───────────────────────────────────────────
    policy = PolicyEngine(config_path=args.policy)
    ctx = RiskContext(
        fusion_score=fusion_score,
        session_id=str(uuid.uuid4()),
        caller_id=audio_path.stem,
        transaction_value_usd=args.transaction_value,
        action_type=args.action_type,
    )
    decision = policy.decide(ctx)

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    # ── Step 8: Structured log ────────────────────────────────────────────
    dlog = DecisionLogger()
    dlog.log(DecisionEvent(
        session_id=ctx.session_id,
        caller_id=ctx.caller_id,
        cm_score=cm_score,
        sv_score=sv_score,
        liveness_score=liveness_score,
        flatness_score=liveness_result.get("flatness_score"),
        jitter_score=liveness_result.get("jitter_score"),
        contrast_score=liveness_result.get("contrast_score"),
        fusion_score=fusion_score,
        final_risk_score=decision.final_risk_score,
        action=decision.action.value,
        reason=decision.reason,
        contributing_signals=decision.contributing_signals,
        inference_latency_ms=latency_ms,
    ))

    # ── Output ────────────────────────────────────────────────────────────
    label = "SYNTHETIC" if fusion_score >= 0.5 else "BONA FIDE"
    result = {
        "file":           str(audio_path),
        "spoof_score":    round(fusion_score, 4),
        "label":          label,
        "cm_score":       round(cm_score, 4),
        "liveness_score": round(liveness_score, 4) if liveness_score is not None else None,
        "sv_score":       round(sv_score, 4) if sv_score is not None else None,
        "fusion_score":   round(fusion_score, 4),
        "decision":       decision.action.value,
        "reason":         decision.reason,
        "latency_ms":     round(latency_ms, 1),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Rich formatted output
        console.print("[green]✓[/]")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Signal", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Notes")

        table.add_row(
            "CM Classifier",
            Text(f"{cm_score:.4f}", style=_score_color(cm_score)),
            "Bona fide vs. synthetic (0=genuine, 1=spoof)",
        )
        if liveness_score is not None:
            table.add_row(
                "Liveness",
                Text(f"{liveness_score:.4f}", style=_score_color(liveness_score)),
                f"Flatness={liveness_result.get('flatness_score',0):.3f}  "
                f"Jitter={liveness_result.get('jitter_score',0):.3f}  "
                f"Contrast={liveness_result.get('contrast_score',0):.3f}",
            )
        if sv_score is not None:
            table.add_row(
                "Speaker Verify",
                Text(f"{sv_score:.4f}", style=_score_color(sv_score)),
                "ECAPA-TDNN cosine → spoof prob",
            )
        table.add_row(
            "Fusion Score",
            Text(f"{fusion_score:.4f}", style=_score_color(fusion_score)),
            "Calibrated aggregate spoof probability",
        )
        table.add_row(
            "Latency",
            f"{latency_ms:.1f} ms",
            f"End-to-end (device: {device})",
        )

        console.print(table)

        dec_color = _decision_color(decision.action.value)
        console.print(Panel(
            f"[bold {dec_color}]{label}[/]  →  Decision: [{dec_color}]{decision.action.value}[/]\n"
            f"[dim]{decision.reason}[/]",
            title="[bold]Verdict",
            border_style=dec_color,
        ))

    return EXIT_CODES.get(decision.action.value, 0)


if __name__ == "__main__":
    sys.exit(main())
