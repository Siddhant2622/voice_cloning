"""
training/evaluate_eer.py
=========================
Evaluate the CM classifier using Equal Error Rate (EER) on a held-out
evaluation set of modern voice clones.

Reports:
  - Overall EER
  - Per-TTS-system breakdown
  - Per-source breakdown
  - Score distributions

Usage:
    python training/evaluate_eer.py --checkpoint models/cm_detect2b_v3.pt
    python training/evaluate_eer.py --checkpoint models/cm_detect2b_v3.pt --data-dir data/modern_dataset
    python training/evaluate_eer.py --checkpoint models/cm_detect2b_v3.pt --eval-split 0.2
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import to_mono_16k
from src.features import extract_logmel, extract_ssl_embedding, extract_wav2vec2_embedding
from src.models.cm_classifier import CMScorerWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_eer")

MAX_SECONDS = 3.0
TARGET_SR = 16000


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate. Returns (eer, threshold)."""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5
    
    thresholds = np.unique(scores)
    if len(thresholds) < 2:
        return 0.5, 0.5
    
    fars, frrs = [], []
    for t in thresholds:
        far = np.sum((scores >= t) & (labels == 0)) / n_neg
        frr = np.sum((scores < t) & (labels == 1)) / n_pos
        fars.append(far)
        frrs.append(frr)
    
    fars = np.array(fars)
    frrs = np.array(frrs)
    
    try:
        f_interp = interp1d(thresholds, fars - frrs)
        eer_threshold = brentq(f_interp, thresholds[0], thresholds[-1])
        eer = interp1d(thresholds, fars)(eer_threshold)
    except (ValueError, RuntimeError):
        abs_diff = np.abs(fars - frrs)
        min_idx = np.argmin(abs_diff)
        eer = (fars[min_idx] + frrs[min_idx]) / 2
        eer_threshold = thresholds[min_idx]
    
    return float(eer), float(eer_threshold)


def load_eval_samples(
    data_dir: Path,
    eval_split: float = 0.2,
    max_eval: int = 200,
) -> List[dict]:
    """Load evaluation samples from the modern dataset."""
    meta_path = data_dir / "metadata.csv"
    
    if meta_path.exists():
        with open(meta_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        # Scan directories
        rows = []
        bf_dir = data_dir / "bonafide"
        sp_dir = data_dir / "spoof"
        
        if bf_dir.exists():
            for f in sorted(bf_dir.glob("*")):
                if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    rows.append({"file": f"bonafide/{f.name}", "label": "bonafide",
                                "source": "unknown", "tts_system": "human"})
        if sp_dir.exists():
            for f in sorted(sp_dir.glob("*")):
                if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    rows.append({"file": f"spoof/{f.name}", "label": "spoof",
                                "source": "unknown", "tts_system": "unknown"})
    
    # Take last eval_split fraction as eval set (same seed as training split)
    random.seed(42)
    random.shuffle(rows)
    
    bf_rows = [r for r in rows if r["label"] == "bonafide"]
    sp_rows = [r for r in rows if r["label"] == "spoof"]
    
    n_bf_eval = min(int(len(bf_rows) * eval_split), max_eval // 2)
    n_sp_eval = min(int(len(sp_rows) * eval_split), max_eval // 2)
    
    # Take from the end (training uses the beginning)
    eval_rows = bf_rows[-n_bf_eval:] + sp_rows[-n_sp_eval:]
    random.shuffle(eval_rows)
    
    return eval_rows


def evaluate(
    checkpoint_path: str = "models/cm_detect2b_v3.pt",
    data_dir: str = "data/modern_dataset",
    eval_split: float = 0.2,
    max_eval: int = 200,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    
    # Load model
    scorer = CMScorerWrapper(checkpoint_path=checkpoint_path, device=device)
    logger.info("Loaded checkpoint: %s", checkpoint_path)
    
    # Load eval samples
    data_path = Path(data_dir)
    eval_samples = load_eval_samples(data_path, eval_split, max_eval)
    
    if not eval_samples:
        logger.error("No evaluation samples found!")
        sys.exit(1)
    
    logger.info("Evaluating on %d samples...", len(eval_samples))
    
    all_scores = []
    all_labels = []
    per_source_scores: Dict[str, List[Tuple[float, int]]] = {}
    per_tts_scores: Dict[str, List[Tuple[float, int]]] = {}
    
    t0 = time.time()
    errors = 0
    
    for idx, row in enumerate(eval_samples):
        path = data_path / row["file"]
        label = 0 if row["label"] == "bonafide" else 1
        source = row.get("source", "unknown")
        tts = row.get("tts_system", "unknown")
        
        try:
            data, sr = sf.read(str(path))
            wav = torch.from_numpy(data).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            wav_16k = to_mono_16k(wav, sr)
            
            max_len = int(MAX_SECONDS * TARGET_SR)
            if len(wav_16k) > max_len:
                start = (len(wav_16k) - max_len) // 2
                wav_16k = wav_16k[start:start + max_len]
            elif len(wav_16k) < max_len:
                wav_16k = F.pad(wav_16k, (0, max_len - len(wav_16k)))
            
            mel = extract_logmel(wav_16k, device="cpu")
            ssl = extract_ssl_embedding(wav_16k, device=device)
            w2v = extract_wav2vec2_embedding(wav_16k, device=device)
            
            features = {"logmel": mel, "ssl_embedding": ssl, "wav2vec2_embedding": w2v}
            score = scorer.score(features)
            
            all_scores.append(score)
            all_labels.append(label)
            
            per_source_scores.setdefault(source, []).append((score, label))
            per_tts_scores.setdefault(tts, []).append((score, label))
            
        except Exception as e:
            logger.warning("Error on %s: %s", path.name, e)
            errors += 1
        
        if (idx + 1) % 20 == 0 or (idx + 1) == len(eval_samples):
            elapsed = time.time() - t0
            logger.info("Eval: %d/%d (%.1f%%) — %.2fs/sample",
                       idx + 1, len(eval_samples), 100 * (idx + 1) / len(eval_samples),
                       elapsed / (idx + 1))
    
    if len(all_scores) < 4:
        logger.error("Too few successful evaluations (%d). Aborting.", len(all_scores))
        sys.exit(1)
    
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    
    # Overall EER
    overall_eer, eer_thresh = compute_eer(scores_arr, labels_arr)
    
    # Overall accuracy at 0.5 threshold
    preds = (scores_arr >= 0.5).astype(int)
    accuracy = np.mean(preds == labels_arr)
    
    bf_mask = labels_arr == 0
    sp_mask = labels_arr == 1
    bf_acc = np.mean(preds[bf_mask] == 0) if bf_mask.any() else 0
    sp_acc = np.mean(preds[sp_mask] == 1) if sp_mask.any() else 0
    
    # Per-source breakdown
    source_results = {}
    for source, pairs in sorted(per_source_scores.items()):
        s = np.array([p[0] for p in pairs])
        l = np.array([p[1] for p in pairs])
        if len(np.unique(l)) >= 2:
            src_eer, _ = compute_eer(s, l)
        else:
            src_eer = float("nan")
        src_acc = np.mean((s >= 0.5).astype(int) == l)
        source_results[source] = {
            "n_samples": len(pairs),
            "eer_percent": round(src_eer * 100, 2) if not np.isnan(src_eer) else "N/A",
            "accuracy_percent": round(src_acc * 100, 1),
        }
    
    # Per-TTS breakdown
    tts_results = {}
    for tts, pairs in sorted(per_tts_scores.items()):
        s = np.array([p[0] for p in pairs])
        l = np.array([p[1] for p in pairs])
        avg_score = float(np.mean(s))
        detection_rate = float(np.mean(s >= 0.5)) if all(ll == 1 for _, ll in pairs) else None
        tts_results[tts] = {
            "n_samples": len(pairs),
            "avg_score": round(avg_score, 4),
            "detection_rate": round(detection_rate * 100, 1) if detection_rate is not None else "N/A",
        }
    
    # Report
    report = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "n_evaluated": len(all_scores),
        "n_errors": errors,
        "overall": {
            "eer_percent": round(overall_eer * 100, 2),
            "eer_threshold": round(eer_thresh, 4),
            "accuracy_at_0.5": round(accuracy * 100, 1),
            "bonafide_accuracy": round(bf_acc * 100, 1),
            "spoof_accuracy": round(sp_acc * 100, 1),
        },
        "per_source": source_results,
        "per_tts_system": tts_results,
        "score_stats": {
            "bonafide_mean": round(float(scores_arr[bf_mask].mean()), 4) if bf_mask.any() else "N/A",
            "bonafide_std": round(float(scores_arr[bf_mask].std()), 4) if bf_mask.any() else "N/A",
            "spoof_mean": round(float(scores_arr[sp_mask].mean()), 4) if sp_mask.any() else "N/A",
            "spoof_std": round(float(scores_arr[sp_mask].std()), 4) if sp_mask.any() else "N/A",
        },
    }
    
    # Save report
    report_path = Path(checkpoint_path).with_name("eval_report_v3.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Eval samples:     {len(all_scores)} ({errors} errors)")
    print(f"  Overall EER:      {overall_eer * 100:.2f}%")
    print(f"  EER Threshold:    {eer_thresh:.4f}")
    print(f"  Accuracy @0.5:    {accuracy * 100:.1f}%")
    print(f"  Bonafide Acc:     {bf_acc * 100:.1f}%")
    print(f"  Spoof Acc:        {sp_acc * 100:.1f}%")
    print()
    
    if source_results:
        print("  Per-Source Breakdown:")
        for src, res in sorted(source_results.items()):
            print(f"    {src:30s}  n={res['n_samples']:4d}  acc={res['accuracy_percent']}%  EER={res['eer_percent']}")
    
    if tts_results:
        print("\n  Per-TTS System:")
        for tts, res in sorted(tts_results.items()):
            print(f"    {tts:30s}  n={res['n_samples']:4d}  avg_score={res['avg_score']:.4f}  det_rate={res['detection_rate']}")
    
    print()
    print(f"  Report saved: {report_path}")
    print("=" * 70 + "\n")
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CM classifier with EER")
    parser.add_argument("--checkpoint", default="models/cm_detect2b_v3.pt", help="Model checkpoint")
    parser.add_argument("--data-dir", default="data/modern_dataset", help="Dataset directory")
    parser.add_argument("--eval-split", type=float, default=0.2, help="Fraction to use as eval set")
    parser.add_argument("--max-eval", type=int, default=200, help="Max eval samples")
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        eval_split=args.eval_split,
        max_eval=args.max_eval,
    )
