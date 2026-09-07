"""
training/eval_eer.py
====================
Compute Equal Error Rate (EER) and min-tDCF on a prepared eval partition.

Usage:
    python training/eval_eer.py --data data/asvspoof2019_prepared/eval \\
                                --checkpoint models/cm.pt \\
                                --out results/benchmark_asvspoof19.json

Output (JSON):
    {
        "dataset":    "asvspoof2019_la_eval",
        "eer":        2.31,
        "min_tDCF":   0.0612,
        "threshold":  0.487,
        "n_bonafide": 7355,
        "n_spoof":    63882,
        "timestamp":  "2026-09-07T15:00:00Z"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("eval_eer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def compute_eer(bonafide_scores: list[float], spoof_scores: list[float]) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) from sorted score lists.

    Args:
        bonafide_scores: CM scores for bonafide utterances (lower = more genuine)
        spoof_scores:    CM scores for spoof utterances (higher = more spoofed)

    Returns:
        (eer, threshold): EER as percentage, decision threshold
    """
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

    all_scores = bonafide_scores + spoof_scores
    all_labels = [0] * len(bonafide_scores) + [1] * len(spoof_scores)

    thresholds = sorted(set(all_scores))
    frrs, fars = [], []

    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in all_scores]
        tp  = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 1)
        fn  = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 1)
        fp  = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 0)
        tn  = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 0)

        frr = fn / max(1, tp + fn)  # False Rejection Rate
        far = fp / max(1, fp + tn)  # False Acceptance Rate
        frrs.append(frr)
        fars.append(far)

    # Find EER at intersection of FRR and FAR curves
    thresholds_arr = np.array(thresholds)
    frrs_arr       = np.array(frrs)
    fars_arr       = np.array(fars)

    diff = frrs_arr - fars_arr
    idx  = np.argmin(np.abs(diff))
    eer  = (frrs_arr[idx] + fars_arr[idx]) / 2.0 * 100.0  # percentage
    thr  = float(thresholds_arr[idx])
    return eer, thr


def compute_min_tdcf(
    bonafide_scores: list[float],
    spoof_scores:    list[float],
    p_spoof:         float = 0.05,
    c_miss:          float = 1.0,
    c_fa:            float = 10.0,
) -> float:
    """
    Compute normalised minimum tandem Detection Cost Function (min-tDCF).
    Uses ASVspoof 2019 default cost parameters.
    """
    import numpy as np

    all_scores = np.array(bonafide_scores + spoof_scores)
    all_labels = np.array([0] * len(bonafide_scores) + [1] * len(spoof_scores))

    thresholds = np.unique(all_scores)
    min_tdcf   = float("inf")

    for thr in thresholds:
        preds   = (all_scores >= thr).astype(int)
        miss_r  = ((preds == 0) & (all_labels == 1)).sum() / max(1, (all_labels == 1).sum())
        fa_r    = ((preds == 1) & (all_labels == 0)).sum() / max(1, (all_labels == 0).sum())
        tdcf    = c_miss * p_spoof * miss_r + c_fa * (1 - p_spoof) * fa_r
        min_tdcf = min(min_tdcf, tdcf)

    # Normalise by default (c_miss * p_spoof + c_fa * (1-p_spoof)) — ASVspoof convention
    norm = c_miss * p_spoof + c_fa * (1 - p_spoof)
    return min_tdcf / norm


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------

def collect_scores(data_dir: Path, checkpoint_path: Path) -> tuple[list, list]:
    """
    Run CM inference on all utterances and return (bonafide_scores, spoof_scores).
    """
    import torch
    import csv
    from src.preprocessing import preprocess_file
    from src.features import extract_all
    from src.models.cm_classifier import CMScorerWrapper

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    scorer    = CMScorerWrapper(checkpoint_path=checkpoint_path, device=device)
    meta_csv  = data_dir / "samples_metadata.csv"

    if not meta_csv.exists():
        logger.error("samples_metadata.csv not found in %s", data_dir)
        sys.exit(1)

    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    bonafide_scores, spoof_scores = [], []
    n_err = 0

    for i, row in enumerate(rows):
        audio_path = Path(row["file"])
        if not audio_path.is_absolute():
            audio_path = data_dir / audio_path
        label = row["label"]

        if (i + 1) % 500 == 0:
            logger.info("  Scored %d / %d  (errors: %d)", i + 1, len(rows), n_err)

        try:
            waveform, _ = preprocess_file(audio_path)
            feats       = extract_all(waveform, device=device)
            score       = scorer.score(feats)

            if label == "bonafide":
                bonafide_scores.append(score)
            else:
                spoof_scores.append(score)
        except Exception as exc:
            n_err += 1
            if n_err <= 5:
                logger.warning("  Error on %s: %s", audio_path.name, exc)

    logger.info("Scored %d bonafide, %d spoof  (%d errors)", len(bonafide_scores), len(spoof_scores), n_err)
    return bonafide_scores, spoof_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EER + min-tDCF for VoiceGuard CM classifier.")
    parser.add_argument("--data",       required=True, help="Eval data dir with samples_metadata.csv.")
    parser.add_argument("--checkpoint", required=True, help="Path to cm.pt checkpoint.")
    parser.add_argument("--out",        default="results/benchmark.json", help="Output JSON path.")
    parser.add_argument("--dataset",    default="asvspoof2019_la_eval",  help="Dataset name tag.")
    args = parser.parse_args()

    data_dir    = Path(args.data)
    ckpt_path   = Path(args.checkpoint)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting scores from: %s", data_dir)
    logger.info("Using checkpoint: %s", ckpt_path)

    t0 = time.time()
    bonafide_scores, spoof_scores = collect_scores(data_dir, ckpt_path)
    elapsed = time.time() - t0

    if not bonafide_scores or not spoof_scores:
        logger.error("Not enough scores to compute EER. Check your data directory.")
        sys.exit(1)

    logger.info("Computing EER...")
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    logger.info("Computing min-tDCF...")
    min_tdcf = compute_min_tdcf(bonafide_scores, spoof_scores)

    results = {
        "dataset":    args.dataset,
        "eer":        round(eer, 4),
        "min_tDCF":   round(min_tdcf, 6),
        "threshold":  round(threshold, 4),
        "n_bonafide": len(bonafide_scores),
        "n_spoof":    len(spoof_scores),
        "inference_time_s": round(elapsed, 1),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n%s", "=" * 50)
    logger.info("  EER:       %.2f%%", eer)
    logger.info("  min-tDCF:  %.4f", min_tdcf)
    logger.info("  Threshold: %.4f", threshold)
    logger.info("  Results:   %s", out_path)
    logger.info("%s", "=" * 50)

    print(f"\n[OK] EER = {eer:.2f}%  |  min-tDCF = {min_tdcf:.4f}  |  Results → {out_path}")


if __name__ == "__main__":
    main()
