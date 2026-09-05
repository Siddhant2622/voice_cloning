"""
training/train_cm.py
Phase 1 — Train the CM (Countermeasure) classifier.

Usage:
    python training/train_cm.py --data data/samples --epochs 30 --out models/cm.pt
    python training/train_cm.py --data data/ASVspoof2019_LA_train --epochs 50 --out models/cm.pt

Data directory format:
    <data_dir>/
        bonafide/   *.wav | *.flac   (label = 0)
        spoof/      *.wav | *.flac   (label = 1)
  OR:
    <data_dir>/samples_metadata.csv  (file, label columns)
    <data_dir>/<audio files>

The model is saved as a PyTorch state dict to --out.
Training runs on GPU if available (RTX 3050).
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_cm")


def load_dataset(data_dir: Path):
    """
    Load (waveform, label) pairs from a data directory.
    Supports both directory structure and metadata CSV.
    """
    from src.preprocessing import preprocess_file

    samples = []

    # Try metadata CSV first
    meta_csv = data_dir / "samples_metadata.csv"
    if meta_csv.exists():
        logger.info("Loading from metadata CSV: %s", meta_csv)
        with open(meta_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = 1 if row["label"] == "spoof" else 0
                audio_path = data_dir / row["file"]
                if audio_path.exists():
                    samples.append((audio_path, label))
                else:
                    logger.warning("Missing file: %s", audio_path)
    else:
        # bonafide/ and spoof/ subdirs
        for subdir, label in [("bonafide", 0), ("spoof", 1)]:
            d = data_dir / subdir
            if not d.exists():
                continue
            for f in sorted(d.glob("*")):
                if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
                    samples.append((f, label))

    logger.info("Dataset: %d samples (%d bona fide, %d spoof)",
                len(samples),
                sum(1 for _, l in samples if l == 0),
                sum(1 for _, l in samples if l == 1))

    if not samples:
        logger.error("No audio files found in %s", data_dir)
        sys.exit(1)

    # Preprocess all files
    logger.info("Preprocessing audio files...")
    processed = []
    for path, label in samples:
        try:
            waveform, _ = preprocess_file(path)
            processed.append((waveform, label))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    return processed


def build_feature_tensors(processed_samples):
    """Extract features for all samples in batch."""
    import torch
    from src.features import extract_all

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Extracting features on %s...", device)

    logmels, ssl_embs, labels = [], [], []
    for i, (waveform, label) in enumerate(processed_samples):
        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(processed_samples))
        try:
            feats = extract_all(waveform, device=device)
            logmels.append(feats["logmel"])
            ssl_embs.append(feats["ssl_embedding"])
            labels.append(label)
        except Exception as exc:
            logger.warning("Feature extraction failed for sample %d: %s", i, exc)

    return logmels, ssl_embs, labels


def pad_logmels(logmels):
    """Pad log-mel tensors to the same time dimension."""
    import torch
    max_t = max(m.shape[1] for m in logmels)
    padded = []
    for m in logmels:
        if m.shape[1] < max_t:
            pad = torch.zeros(m.shape[0], max_t - m.shape[1])
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    return torch.stack(padded)     # [N, n_mels, T]


def main():
    parser = argparse.ArgumentParser(description="Train CM classifier.")
    parser.add_argument("--data",    default="data/samples",  help="Data directory.")
    parser.add_argument("--epochs",  type=int, default=30,     help="Training epochs.")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch",   type=int, default=8,      help="Batch size.")
    parser.add_argument("--out",     default="models/cm.pt",   help="Output checkpoint path.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from src.models.cm_classifier import build_cm_classifier

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.error("Run: python data/generate_samples.py  to create sample data.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device)

    # ── Load and preprocess ─────────────────────────────────────────────────
    processed = load_dataset(data_dir)
    logmels, ssl_embs, labels = build_feature_tensors(processed)

    if len(labels) == 0:
        logger.error("No features extracted. Check your audio files.")
        sys.exit(1)

    # ── Build tensors ───────────────────────────────────────────────────────
    X_logmel = pad_logmels(logmels)
    X_ssl    = torch.stack(ssl_embs)
    y        = torch.tensor(labels, dtype=torch.float32)

    logger.info("Feature shapes: logmel=%s  ssl=%s  labels=%s",
                X_logmel.shape, X_ssl.shape, y.shape)

    # ── Train/val split ─────────────────────────────────────────────────────
    n_total = len(y)
    n_val   = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    dataset = TensorDataset(X_logmel, X_ssl, y)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    # ── Model, optimizer, loss ──────────────────────────────────────────────
    model     = build_cm_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger.info("Starting training: %d epochs, batch=%d, lr=%.1e",
                args.epochs, args.batch, args.lr)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_logmel, batch_ssl, batch_y in train_loader:
            batch_logmel = batch_logmel.to(device)
            batch_ssl    = batch_ssl.to(device)
            batch_y      = batch_y.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            logits = model(batch_logmel, batch_ssl)
            loss   = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_logmel, batch_ssl, batch_y in val_loader:
                batch_logmel = batch_logmel.to(device)
                batch_ssl    = batch_ssl.to(device)
                batch_y      = batch_y.to(device).unsqueeze(-1)
                logits = model(batch_logmel, batch_ssl)
                val_loss += criterion(logits, batch_y).item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()

        val_loss /= max(1, len(val_loader))
        val_acc   = correct / max(1, total)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.1f%%",
                epoch, args.epochs, train_loss, val_loss, val_acc * 100,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Save best checkpoint ────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(out_path))
    logger.info("Best model saved: %s  (val_loss=%.4f)", out_path, best_val_loss)
    print(f"\n[OK] Training complete. Checkpoint: {out_path}")


if __name__ == "__main__":
    main()
