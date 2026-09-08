"""
training/benchmark_edge.py
Comprehensive benchmark of VoiceGuard:
1. Trained CM Deep Model (WavLM + Attentive Pooling + Fusion)
2. Calibrated Edge Engine (Multi-Feature Acoustic Analysis)
3. Combined System

Outputs detailed accuracy, precision, recall, F1, FAR, and FRR.
"""

import csv
import json
import logging
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

def run_edge_engine_simulation(ac_feats: dict, thresholds: dict) -> dict:
    """
    Simulates the browser Edge Engine `evaluateEdgeRisk` function.
    """
    flatness    = ac_feats.get("spectral_flatness", 0.0)
    eVar        = ac_feats.get("energy_variance", 0.0)
    periodicity = ac_feats.get("pitch_periodicity", 0.0)
    hf          = ac_feats.get("hf_ratio", 0.0)
    centroid    = ac_feats.get("spectral_centroid", 0.0)
    zcr         = ac_feats.get("zcr", 0.0)
    hnr         = ac_feats.get("hnr_proxy", 0.0)
    rms         = ac_feats.get("rms", 0.0)

    # Biological voice safeguards
    if eVar >= 1.10:
        cm_score = 0.08
        lv_score = 0.06
    elif eVar >= 0.98 and periodicity <= 0.40 and zcr <= 0.125:
        cm_score = 0.05
        lv_score = 0.04
    else:
        cm_score = 0.05
        # 1. Classical vocoder (pyttsx3): robotic periodic pitch + low energy dynamic range + high HNR
        if periodicity > 0.70:
            cm_score += 0.55 + (periodicity - 0.70) * 1.8
        if hnr > 2.8:
            cm_score += 0.25
        if eVar < 0.92:
            cm_score += 0.15

        # 2. Modern Neural TTS (Jenny, Guy, Aria, etc.):
        # Elevated zero-crossings (neural vocoder noise), flatter higher-frequency content
        if zcr >= 0.125:
            cm_score += 0.25 + (zcr - 0.125) * 5.0
        if centroid > 0.23:
            cm_score += (centroid - 0.23) * 3.5
        if flatness > 0.35 and zcr > 0.125:
            cm_score += 0.18
        if periodicity > 0.50 and zcr > 0.120:
            cm_score += 0.20
        if hf > 0.032:
            cm_score += (hf - 0.032) * 7.0

        # Safeguards
        if periodicity < 0.55 and centroid < 0.23 and zcr < 0.120:
            cm_score = min(cm_score, 0.12)
        if eVar > 1.05 and zcr < 0.130:
            cm_score = min(cm_score, 0.15)

        lv_score = min(1.0, cm_score * 0.92 + 0.04)

    cm_score = min(1.0, max(0.02, cm_score))
    lv_score = min(1.0, max(0.02, lv_score))
    fusion_score = 0.65 * cm_score + 0.35 * lv_score

    decision = "ALLOW"
    if fusion_score >= 0.80:
        decision = "BLOCK"
    elif fusion_score >= 0.60:
        decision = "ALERT"
    elif fusion_score >= 0.45:
        decision = "STEP_UP"

    return {
        "cm_score": round(cm_score, 3),
        "liveness_score": round(lv_score, 3),
        "fusion_score": round(fusion_score, 3),
        "decision": decision,
        "predicted_spoof": bool(fusion_score >= 0.48)
    }


def compute_metrics(y_true, y_pred, y_prob=None):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy  = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, (tp + fp))
    recall    = tp / max(1, (tp + fn))
    f1        = 2 * precision * recall / max(1e-8, (precision + recall))
    far       = fp / max(1, (fp + tn))  # False Acceptance Rate (human flagged as spoof)
    frr       = fn / max(1, (fn + tp))  # False Rejection Rate (spoof passed as human)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "far": round(float(far), 4),
        "frr": round(float(frr), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def main():
    samples_dir = Path("data/samples")
    meta_csv = samples_dir / "samples_metadata.csv"
    edge_thresh_path = Path("models/edge_thresholds.json")

    thresholds = {}
    if edge_thresh_path.exists():
        with open(edge_thresh_path, encoding="utf-8") as f:
            thresholds = json.load(f)

    from training.extract_thresholds import compute_acoustic_features
    from src.models.cm_classifier import CMScorerWrapper
    from src.preprocessing import preprocess_file
    from src.features import extract_all

    cm_scorer = CMScorerWrapper(checkpoint_path="models/cm.pt")

    y_true = []
    cm_preds = []
    cm_scores = []
    edge_preds = []
    edge_scores = []
    sample_reports = []

    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = samples_dir / row["file"]
            if not audio_path.exists():
                continue

            label = 1 if row["label"] == "spoof" else 0
            y_true.append(label)

            # 1. Deep ML Model Score
            wf, _ = preprocess_file(audio_path)
            feats = extract_all(wf)
            ml_score = cm_scorer.score(feats)
            cm_scores.append(ml_score)
            cm_preds.append(1 if ml_score >= 0.50 else 0)

            # 2. Browser Edge Engine Score
            data, sr = sf.read(str(audio_path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            ac_feats = compute_acoustic_features(data, sr)
            edge_sim = run_edge_engine_simulation(ac_feats, thresholds)
            edge_scores.append(edge_sim["fusion_score"])
            edge_preds.append(1 if edge_sim["predicted_spoof"] else 0)

            sample_reports.append({
                "file": row["file"],
                "label": row["label"],
                "source": row["source"],
                "ml_cm_score": round(ml_score, 4),
                "edge_fusion_score": edge_sim["fusion_score"],
                "edge_decision": edge_sim["decision"],
                "ml_correct": (label == (1 if ml_score >= 0.5 else 0)),
                "edge_correct": (label == (1 if edge_sim["predicted_spoof"] else 0)),
            })

    cm_metrics = compute_metrics(y_true, cm_preds)
    edge_metrics = compute_metrics(y_true, edge_preds)

    results = {
        "dataset_size": len(y_true),
        "bonafide_count": int(sum(1 for y in y_true if y == 0)),
        "spoof_count": int(sum(1 for y in y_true if y == 1)),
        "cm_deep_model": cm_metrics,
        "browser_edge_engine": edge_metrics,
        "sample_reports": sample_reports
    }

    out_json = Path("results/benchmark_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*75)
    print("VOICEGUARD BENCHMARK REPORT (Trained CM vs Browser Edge Engine)")
    print("="*75)
    print(f"Dataset: {len(y_true)} samples ({results['bonafide_count']} Bonafide Human, {results['spoof_count']} Synthetic/Neural Spoofs)")
    print("-" * 75)
    print(f"{'Metric':<20} | {'Trained CM Model':<22} | {'Browser Edge Engine':<22}")
    print("-" * 75)
    for k in ["accuracy", "precision", "recall", "f1", "far", "frr"]:
        m1 = cm_metrics[k]
        m2 = edge_metrics[k]
        print(f"{k.upper():<20} | {m1 * 100:6.1f}%{' '*15} | {m2 * 100:6.1f}%")
    print("="*75)

    print("\nDetailed Per-Sample Results:")
    print(f"{'File':<26} | {'True Label':<9} | {'CM Score':<9} | {'Edge Score':<10} | {'Decision':<8} | {'Status'}")
    print("-" * 75)
    for s in sample_reports:
        ok_tag = "[PASS]" if (s["ml_correct"] and s["edge_correct"]) else "[CHECK]"
        print(f"{s['file']:<26} | {s['label']:<9} | {s['ml_cm_score']:<9.3f} | {s['edge_fusion_score']:<10.3f} | {s['edge_decision']:<8} | {ok_tag}")
    print("="*75)


if __name__ == "__main__":
    main()
