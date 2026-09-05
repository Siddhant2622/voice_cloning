"""
training/train_fusion.py
Phase 3 — Train the calibrated fusion layer.

Runs the full detection pipeline on a labeled dataset, collects
per-sample score bundles, and trains a logistic regression fusion model.

Usage:
    python training/train_fusion.py --data data/samples --cm-model models/cm.pt --out models/fusion.pkl
    python training/train_fusion.py --data data/ASVspoof2019_LA_train --cm-model models/cm.pt --out models/fusion.pkl
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_fusion")


def main():
    parser = argparse.ArgumentParser(description="Train fusion model.")
    parser.add_argument("--data",       default="data/samples",  help="Labeled data directory.")
    parser.add_argument("--cm-model",   default="models/cm.pt",  help="CM classifier checkpoint.")
    parser.add_argument("--out",        default="models/fusion.pkl", help="Output fusion model path.")
    parser.add_argument("--no-sv",      action="store_true",     help="Disable speaker verification pillar.")
    parser.add_argument("--no-liveness",action="store_true",     help="Disable liveness pillar.")
    parser.add_argument("--enroll-dir", default=None,            help="Directory of reference enrollment audio files (speaker_id.wav).")
    args = parser.parse_args()

    import torch
    from src.preprocessing import preprocess_file
    from src.features import extract_all
    from src.models.cm_classifier import CMScorerWrapper
    from src.models.liveness import LivenessScorer
    from src.models.speaker_verify import SpeakerVerifier
    from src.fusion import FusionModel, ScoreBundle

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Load labeled samples ────────────────────────────────────────────────
    samples = []
    meta_csv = data_dir / "samples_metadata.csv"
    if meta_csv.exists():
        with open(meta_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = 1 if row["label"] == "spoof" else 0
                ap    = data_dir / row["file"]
                if ap.exists():
                    samples.append((ap, label))
    else:
        for subdir, label in [("bonafide", 0), ("spoof", 1)]:
            d = data_dir / subdir
            if not d.exists():
                continue
            for f in sorted(d.glob("*")):
                if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                    samples.append((f, label))

    logger.info("Loaded %d labeled samples.", len(samples))

    # ── Initialize scorers ──────────────────────────────────────────────────
    cm_scorer   = CMScorerWrapper(checkpoint_path=args.cm_model, device=device)
    lv_scorer   = LivenessScorer() if not args.no_liveness else None
    sv          = SpeakerVerifier(device=device) if not args.no_sv else None

    # ── Score all samples ───────────────────────────────────────────────────
    bundles, labels = [], []
    for i, (audio_path, label) in enumerate(samples):
        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(samples))
        try:
            waveform, _ = preprocess_file(audio_path)
            features    = extract_all(waveform, device=device)
            cm_score    = cm_scorer.score(features)

            liveness_score = None
            if lv_scorer:
                try:
                    lv_result      = lv_scorer.score(waveform)
                    liveness_score = lv_result["liveness_score"]
                except Exception:
                    pass

            sv_score = None
            if sv and sv.available and args.enroll_dir:
                enroll_path = Path(args.enroll_dir) / f"speaker_{label}.wav"
                if enroll_path.exists() and audio_path.stem != f"speaker_{label}":
                    ref_wav, _ = preprocess_file(enroll_path)
                    sv.enroll(f"ref_{label}", ref_wav)
                    sv_score = sv.score_as_spoof_probability(waveform, speaker_id=f"ref_{label}")

            bundle = ScoreBundle(
                cm_score=cm_score,
                sv_score=sv_score,
                liveness_score=liveness_score,
            )
            bundles.append(bundle)
            labels.append(label)
        except Exception as exc:
            logger.warning("Skipping %s: %s", audio_path.name, exc)

    if len(bundles) < 4:
        logger.error("Not enough scored samples (%d) to train fusion.", len(bundles))
        sys.exit(1)

    logger.info("Scored %d samples. Training fusion model...", len(bundles))

    # ── Train fusion ────────────────────────────────────────────────────────
    fm = FusionModel()
    # Use calibrate=False if dataset is tiny (< 20 samples) — CV doesn't work well
    calibrate = len(bundles) >= 20
    fm.fit(bundles, labels, calibrate=calibrate)

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    fm.save(out_path)
    print(f"\n[OK] Fusion model saved: {out_path}  (calibrated={calibrate})")


if __name__ == "__main__":
    main()
