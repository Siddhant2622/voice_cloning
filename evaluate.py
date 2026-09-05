"""
evaluate.py — Phase 3 evaluation harness.

Runs the full detection pipeline on a benchmark subset and reports:
  - EER (Equal Error Rate)
  - APCER / BPCER at a chosen operating threshold
  - Generalization gap (per attack family, if protocol file is available)
  - DET curve (saved as results/det_curve.png)

Usage:
    python evaluate.py --data data/ASVspoof2019_LA --partition eval
    python evaluate.py --data data/samples --partition demo   # demo mode on sample clips
    python evaluate.py --data data/WaveFake

Architecture reference: §7 Evaluation metrics in voice-clone-detection-architecture.md
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_asvspoof2019_la(data_dir: Path, partition: str) -> List[Tuple[Path, int, str]]:
    """
    Load ASVspoof 2019 LA samples from the official protocol file.

    Returns:
        List of (audio_path, label, attack_type)
        label: 0 = bona fide, 1 = spoof
        attack_type: 'genuine' or attack ID (A01–A19)
    """
    # Protocol file location
    proto_dir = data_dir / "ASVspoof2019_LA_cm_protocols"
    proto_file = proto_dir / f"ASVspoof2019.LA.cm.{partition}.trl.txt"

    if not proto_file.exists():
        # Try alternate location
        proto_file = data_dir / f"ASVspoof2019.LA.cm.{partition}.trl.txt"

    if not proto_file.exists():
        logger.warning("Protocol file not found: %s", proto_file)
        return []

    audio_dir = data_dir / f"ASVspoof2019_LA_{partition}" / "flac"
    if not audio_dir.exists():
        audio_dir = data_dir / f"ASVspoof2019_LA_{partition}_audio" / "flac"

    samples = []
    with open(proto_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker, file_id, _, attack_type, label_str = parts[:5]
            label = 0 if label_str == "bonafide" else 1
            audio_path = audio_dir / f"{file_id}.flac"
            if audio_path.exists():
                samples.append((audio_path, label, attack_type))

    logger.info(
        "ASVspoof2019 LA %s: %d samples (%d bona fide, %d spoof)",
        partition,
        len(samples),
        sum(1 for _, l, _ in samples if l == 0),
        sum(1 for _, l, _ in samples if l == 1),
    )
    return samples


def load_generic_dir(data_dir: Path) -> List[Tuple[Path, int, str]]:
    """
    Load from bonafide/ and spoof/ subdirs OR samples_metadata.csv.
    Returns (path, label, attack_type) triples (attack_type='unknown' for non-ASVspoof).
    """
    samples = []
    meta_csv = data_dir / "samples_metadata.csv"
    if meta_csv.exists():
        with open(meta_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = 1 if row["label"] == "spoof" else 0
                source = row.get("source", "unknown")
                ap = data_dir / row["file"]
                if ap.exists():
                    samples.append((ap, label, source))
    else:
        for subdir, label in [("bonafide", 0), ("spoof", 1)]:
            d = data_dir / subdir
            if d.exists():
                for f in sorted(d.glob("*")):
                    if f.suffix.lower() in {".wav", ".flac", ".mp3"}:
                        samples.append((f, label, subdir))
    return samples


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------

def score_dataset(
    samples: List[Tuple[Path, int, str]],
    cm_ckpt: Optional[str],
    fusion_ckpt: Optional[str],
    device: str,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Run full pipeline on all samples.

    Returns:
        (scores, labels, attack_types)
        scores: float array of fusion spoof probabilities
        labels: 0/1 int array
    """
    import torch
    from src.preprocessing import preprocess_file
    from src.features import extract_all
    from src.models.cm_classifier import CMScorerWrapper
    from src.models.liveness import LivenessScorer
    from src.fusion import FusionModel, ScoreBundle

    cm_scorer  = CMScorerWrapper(checkpoint_path=cm_ckpt, device=device)
    lv_scorer  = LivenessScorer()
    fm         = FusionModel()
    if fusion_ckpt and Path(fusion_ckpt).exists():
        fm = FusionModel.load(fusion_ckpt)

    if max_samples:
        samples = samples[:max_samples]

    scores_out, labels_out, attacks_out = [], [], []
    n = len(samples)
    t0 = time.perf_counter()

    for i, (audio_path, label, attack_type) in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            eta_s   = elapsed / (i + 1) * (n - i - 1)
            logger.info("  %d / %d  (ETA: %.0f s)", i + 1, n, eta_s)

        try:
            waveform, _ = preprocess_file(audio_path)
            features    = extract_all(waveform, device=device)
            cm_score    = cm_scorer.score(features)

            liveness_score = None
            try:
                lv_result      = lv_scorer.score(waveform)
                liveness_score = lv_result["liveness_score"]
            except Exception:
                pass

            bundle = ScoreBundle(cm_score=cm_score, liveness_score=liveness_score)
            fusion_score = fm.score(bundle)

            scores_out.append(fusion_score)
            labels_out.append(label)
            attacks_out.append(attack_type)
        except Exception as exc:
            logger.warning("Skipping %s: %s", audio_path.name, exc)

    return np.array(scores_out), np.array(labels_out), attacks_out


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_eer(genuine_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    all_scores = np.concatenate([genuine_scores, spoof_scores])
    all_labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(spoof_scores))])
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)


def compute_apcer_bpcer(
    genuine_scores: np.ndarray, spoof_scores: np.ndarray, threshold: float
) -> Tuple[float, float]:
    apcer = float(np.mean(spoof_scores <= threshold))
    bpcer = float(np.mean(genuine_scores > threshold))
    return apcer, bpcer


def plot_det_curve(
    genuine_scores: np.ndarray,
    spoof_scores: np.ndarray,
    output_path: Path,
    title: str = "DET Curve",
):
    all_scores = np.concatenate([genuine_scores, spoof_scores])
    all_labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(spoof_scores))])
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr

    from scipy.stats import norm as scipy_norm

    def _probit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return scipy_norm.ppf(p)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(_probit(fpr), _probit(fnr), color="#4F8EF7", linewidth=2.5, label="Fusion ensemble")
    ax.plot(_probit(fpr), _probit(fpr), "k--", alpha=0.4, label="Random (EER line)")

    ticks = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    tick_labels = ["1%", "2%", "5%", "10%", "20%", "40%"]
    ax.set_xticks([_probit(t) for t in ticks])
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks([_probit(t) for t in ticks])
    ax.set_yticklabels(tick_labels, fontsize=9)

    eer = compute_eer(genuine_scores, spoof_scores)
    ax.scatter([_probit(eer)], [_probit(eer)], color="red", zorder=5, s=60, label=f"EER = {eer*100:.2f}%")

    ax.set_xlabel("False Alarm Rate (BPCER)", fontsize=12)
    ax.set_ylabel("Miss Rate (APCER)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info("DET curve saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation harness for voice clone detector.")
    parser.add_argument("--data",       default="data/samples",  help="Data directory.")
    parser.add_argument("--partition",  default="demo",          help="eval|dev|train|demo.")
    parser.add_argument("--cm-model",   default="models/cm.pt",  help="CM checkpoint.")
    parser.add_argument("--fusion-model", default="models/fusion.pkl", help="Fusion checkpoint.")
    parser.add_argument("--threshold",  type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--max-samples",type=int, default=None,  help="Limit for quick eval.")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Evaluation device: %s", device)

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load samples ────────────────────────────────────────────────────────
    if args.partition in {"eval", "dev", "train"}:
        samples = load_asvspoof2019_la(data_dir, args.partition)
    else:
        samples = load_generic_dir(data_dir)

    if not samples:
        logger.error("No samples found. Check data directory and partition.")
        sys.exit(1)

    # ── Score ────────────────────────────────────────────────────────────────
    cm_ckpt     = args.cm_model if Path(args.cm_model).exists() else None
    fusion_ckpt = args.fusion_model if Path(args.fusion_model).exists() else None

    logger.info("Scoring %d samples...", len(samples))
    scores, labels, attack_types = score_dataset(
        samples, cm_ckpt, fusion_ckpt, device, max_samples=args.max_samples
    )

    if len(scores) == 0:
        logger.error("No samples scored successfully.")
        sys.exit(1)

    genuine_scores = scores[labels == 0]
    spoof_scores   = scores[labels == 1]

    # ── Aggregate metrics ────────────────────────────────────────────────────
    eer   = compute_eer(genuine_scores, spoof_scores)
    apcer, bpcer = compute_apcer_bpcer(genuine_scores, spoof_scores, args.threshold)

    # ── Per-attack-family EER ────────────────────────────────────────────────
    attack_eers: Dict[str, float] = {}
    unique_attacks = set(t for t, l in zip(attack_types, labels) if l == 1)
    for attack in sorted(unique_attacks):
        mask  = (np.array(attack_types) == attack) & (labels == 1)
        if mask.sum() < 5:
            continue
        a_eer = compute_eer(genuine_scores, scores[mask])
        attack_eers[attack] = a_eer

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {args.data}  [{args.partition}]")
    print("=" * 60)
    print(f"  Samples scored:       {len(scores)}")
    print(f"  Bona fide samples:    {len(genuine_scores)}")
    print(f"  Spoof samples:        {len(spoof_scores)}")
    print()
    print(f"  EER (Equal Error Rate):    {eer*100:.2f}%")
    print(f"  Operating threshold:       {args.threshold:.2f}")
    print(f"  APCER (missed attacks):    {apcer*100:.2f}%")
    print(f"  BPCER (false alarms):      {bpcer*100:.2f}%")
    print()

    if attack_eers:
        print("  Per-attack-family EER:")
        for attack, a_eer in sorted(attack_eers.items(), key=lambda x: x[1]):
            print(f"    {attack:10s}  {a_eer*100:.2f}%")
        known_eer   = float(np.mean(list(attack_eers.values())))
        overall_eer = eer
        print(f"\n  Generalization gap (overall EER vs mean per-attack EER):")
        print(f"    Overall EER:      {overall_eer*100:.2f}%")
        print(f"    Mean per-attack:  {known_eer*100:.2f}%")

    print("=" * 60)

    # ── DET curve ────────────────────────────────────────────────────────────
    det_path = RESULTS_DIR / f"det_curve_{args.partition}.png"
    try:
        plot_det_curve(
            genuine_scores, spoof_scores, det_path,
            title=f"DET Curve — {args.partition} partition"
        )
        print(f"\n  DET curve: {det_path}")
    except Exception as exc:
        logger.warning("DET curve plotting failed: %s", exc)

    # ── Save JSON results ────────────────────────────────────────────────────
    results = {
        "partition":   args.partition,
        "n_total":     int(len(scores)),
        "n_genuine":   int(len(genuine_scores)),
        "n_spoof":     int(len(spoof_scores)),
        "eer":         round(float(eer), 6),
        "threshold":   args.threshold,
        "apcer":       round(float(apcer), 6),
        "bpcer":       round(float(bpcer), 6),
        "attack_eers": {k: round(v, 6) for k, v in attack_eers.items()},
    }
    results_json = RESULTS_DIR / f"eval_{args.partition}.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON: {results_json}")

    # ── Honest limitations note ──────────────────────────────────────────────
    print("""
NOTE ON GENERALIZATION:
  These results reflect performance on the benchmark's known attack families.
  Real-world performance against novel synthesis methods (not in the training set)
  will be materially worse. This is the core unsolved challenge in anti-spoofing.
  See LIMITATIONS.md for a full discussion.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
