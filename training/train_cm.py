"""
training/train_cm.py
=====================
Phase 1 — Train the CM (Countermeasure) classifier.
Upgraded to Resemble AI DETECT-2B inspired architecture:
  - Dual SSL ensemble: WavLM + Wav2Vec2
  - Mamba-SSM temporal encoder (pure-PyTorch S4-lite)
  - Frame-level output head with auxiliary BCE loss
  - SSL agreement auxiliary loss (cross-SSL consistency)
  - Resemble Enhance data augmentation support
  - Multi-generator training data support (pyttsx3, gTTS, edge-tts, WaveFake)

Usage:
    # Quick start (small dataset):
    python training/train_cm.py --data data/samples --epochs 80

    # Full training (ASVspoof 2019 / WaveFake):
    python training/train_cm.py --data data/WaveFake --epochs 150 --batch 16

    # With Resemble Enhance augmentation:
    python training/train_cm.py --data data/samples --enhance --epochs 100

Data directory format:
    <data_dir>/bonafide/  *.wav|*.flac  (label = 0)
    <data_dir>/spoof/     *.wav|*.flac  (label = 1)
  OR:
    <data_dir>/samples_metadata.csv  (file, label columns)
    <data_dir>/<audio files>
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


# ---------------------------------------------------------------------------
# SSL Agreement Loss
# Resemble AI DETECT-World insight: real speech produces consistent embeddings
# across different SSL models (both capture the same phonetic reality).
# Synthetic speech creates subtle inconsistencies — each model detects
# slightly different artifacts of the generator.
# We exploit this cross-SSL disagreement as an auxiliary training signal.
# ---------------------------------------------------------------------------
def compute_ssl_agreement_loss(
    wavlm_feats:    "torch.Tensor",  # [B, D]
    wav2vec2_feats: "torch.Tensor",  # [B, D]
    labels:         "torch.Tensor",  # [B, 1]  0=bonafide, 1=spoof
) -> "torch.Tensor":
    """
    SSL Agreement Loss:
      - For BONAFIDE pairs: maximise cosine similarity (WavLM ≈ Wav2Vec2)
      - For SPOOF pairs:    minimise cosine similarity (WavLM ≠ Wav2Vec2)

    This teaches the model that synthetic audio creates cross-SSL inconsistency.

    Returns scalar loss tensor.
    """
    import torch
    import torch.nn.functional as F

    # Cosine similarity between WavLM and Wav2Vec2 features: [B]
    sim = F.cosine_similarity(wavlm_feats, wav2vec2_feats, dim=-1)

    # For bonafide (label=0): sim should be HIGH → loss = (1 - sim)
    # For spoof (label=1):    sim should be LOW  → loss = sim
    labels_flat = labels.squeeze(-1)  # [B]
    agreement_loss = labels_flat * (sim) + (1 - labels_flat) * (1 - sim)
    return agreement_loss.mean()


# ---------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------
def apply_spec_augment(
    mel: "torch.Tensor",
    max_time_mask: int = 20,
    max_freq_mask: int = 10,
) -> "torch.Tensor":
    """
    Apply SpecAugment: random frequency and time masking to log-mel spectrogram.
    mel shape: [B, n_mels, T]
    """
    import random
    import torch
    aug_mel = mel.clone()
    B, n_mels, T = aug_mel.shape

    for b in range(B):
        # Frequency masking
        if max_freq_mask > 0 and n_mels > max_freq_mask:
            f_len = random.randint(1, max_freq_mask)
            f0 = random.randint(0, n_mels - f_len)
            aug_mel[b, f0: f0 + f_len, :] = 0.0

        # Time masking
        if max_time_mask > 0 and T > max_time_mask:
            t_len = random.randint(1, max_time_mask)
            t0 = random.randint(0, T - t_len)
            aug_mel[b, :, t0: t0 + t_len] = 0.0

    return aug_mel


# ---------------------------------------------------------------------------
# Resemble Enhance augmentation
# ---------------------------------------------------------------------------
def apply_resemble_enhance(waveform_np, sr=16000):
    """
    Denoise a waveform using resemble-enhance.
    Returns enhanced waveform numpy array at 16kHz.
    Falls back to original if resemble-enhance not installed.
    """
    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import enhance as resemble_enhance_fn

        wav_t = torch.from_numpy(waveform_np).float()
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)  # [1, T]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        enhanced, out_sr = resemble_enhance_fn(
            wav_t, sr, device=device, nfe=16, denoise_only=True  # fast denoise
        )
        # Resample back to 16kHz if needed
        if out_sr != 16000:
            enhanced = torchaudio.functional.resample(
                enhanced if enhanced.dim() == 2 else enhanced.unsqueeze(0),
                out_sr, 16000
            ).squeeze(0)
        return enhanced.squeeze().cpu().numpy()
    except Exception as exc:
        logger.debug("resemble-enhance unavailable (%s); using original.", exc)
        return waveform_np


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset(data_dir: Path, use_enhance: bool = False):
    """
    Load (waveform, label) pairs.
    Supports directory structure (bonafide/spoof) and metadata CSV.
    Also supports WaveFake directory structure.
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
            for f in sorted(d.glob("**/*")):
                if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
                    samples.append((f, label))

        # WaveFake layout: folders named by generator, all spoof
        if not samples:
            logger.info("Trying WaveFake directory layout...")
            wavefake_spoof_dirs = [
                "ljspeech_full_band_melgan",
                "ljspeech_hifiGAN",
                "ljspeech_melgan",
                "ljspeech_melgan_large",
                "ljspeech_multi_band_melgan",
                "ljspeech_parallel_wavegan",
                "ljspeech_waveflow",
                "jsut_full_band_melgan",
                "jsut_melgan",
                "ljspeech_waveglow",
            ]
            for spoof_dir in wavefake_spoof_dirs:
                d = data_dir / spoof_dir
                if d.exists():
                    for f in sorted(d.glob("**/*.wav")):
                        samples.append((f, 1))
                    logger.info("  Found WaveFake generator: %s (%d files)", spoof_dir,
                                sum(1 for p, l in samples if l == 1))

            # Also look for LJSpeech real audio as bonafide
            for bf_name in ["LJSpeech-1.1", "wavs", "ljspeech_real"]:
                d = data_dir / bf_name
                if d.exists():
                    for f in sorted(d.glob("**/*.wav"))[:3000]:  # cap at 3000
                        samples.append((f, 0))

    logger.info("Dataset: %d samples (%d bonafide, %d spoof)",
                len(samples),
                sum(1 for _, l in samples if l == 0),
                sum(1 for _, l in samples if l == 1))

    if not samples:
        logger.error("No audio files found in %s", data_dir)
        sys.exit(1)

    # Preprocess all files
    logger.info("Preprocessing audio files%s...",
                " + Resemble Enhance denoising" if use_enhance else "")
    processed = []
    for i, (path, label) in enumerate(samples):
        if (i + 1) % 50 == 0:
            logger.info("  %d / %d preprocessed", i + 1, len(samples))
        try:
            waveform, _ = preprocess_file(path)

            if use_enhance and label == 0:
                # Enhance bonafide samples for cleaner training signal
                import numpy as np
                wav_np = waveform.numpy() if hasattr(waveform, "numpy") else waveform
                enhanced_np = apply_resemble_enhance(wav_np)
                # Add original AND enhanced version for 2x bonafide data
                processed.append((waveform, label))
                import torch
                processed.append((torch.from_numpy(enhanced_np.astype("float32")), label))
            else:
                processed.append((waveform, label))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    return processed


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def build_feature_tensors(processed_samples, extract_wav2vec2=True):
    """Extract features for all samples."""
    import torch
    from src.features import extract_all

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Extracting features on %s (wav2vec2=%s)...", device, extract_wav2vec2)

    logmels, ssl_embs, w2v2_embs, labels = [], [], [], []
    for i, (waveform, label) in enumerate(processed_samples):
        if (i + 1) % 10 == 0:
            logger.info("  %d / %d", i + 1, len(processed_samples))
        try:
            feats = extract_all(waveform, device=device, extract_wav2vec2=extract_wav2vec2)
            logmels.append(feats["logmel"])
            ssl_embs.append(feats["ssl_embedding"])
            w2v2_embs.append(feats.get("wav2vec2_embedding", torch.zeros(768)))
            labels.append(label)
        except Exception as exc:
            logger.warning("Feature extraction failed for sample %d: %s", i, exc)

    return logmels, ssl_embs, w2v2_embs, labels


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
    return torch.stack(padded)   # [N, n_mels, T]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train CM classifier (DETECT-2B style).")
    parser.add_argument("--data",       default="data/samples",          help="Data directory.")
    parser.add_argument("--epochs",     type=int,   default=80,          help="Training epochs.")
    parser.add_argument("--lr",         type=float, default=5e-4,        help="Learning rate.")
    parser.add_argument("--batch",      type=int,   default=4,           help="Batch size.")
    parser.add_argument("--out",        default="models/cm.pt",          help="Output checkpoint path.")
    parser.add_argument("--val-split",  type=float, default=0.2,         help="Validation split fraction.")
    parser.add_argument("--cache",      default="data/features_cache.pt", help="Feature cache path.")
    parser.add_argument("--enhance",    action="store_true",             help="Use Resemble Enhance on bonafide samples.")
    parser.add_argument("--no-mamba",   action="store_true",             help="Disable Mamba-SSM (faster but less accurate).")
    parser.add_argument("--no-wav2vec2",action="store_true",             help="Disable Wav2Vec2 branch (saves ~95MB VRAM).")
    parser.add_argument("--agreement-weight", type=float, default=0.3,  help="Weight of SSL agreement loss.")
    parser.add_argument("--frame-weight",     type=float, default=0.2,  help="Weight of frame-level auxiliary loss.")
    args = parser.parse_args()

    # Auto-detect Google Drive root if present
    import os
    drive_root = os.environ.get("VOICEGUARD_DRIVE_ROOT", os.environ.get("DRIVE_ROOT", ""))
    if not drive_root and Path("/content/drive/MyDrive/voiceguard").exists():
        drive_root = "/content/drive/MyDrive/voiceguard"

    if drive_root:
        if args.out == "models/cm.pt":
            args.out = f"{drive_root}/models/cm.pt"
        if args.cache == "data/features_cache.pt":
            args.cache = f"{drive_root}/cache/features_cache.pt"
        if args.data == "data/samples" and Path(f"{drive_root}/data/samples").exists():
            args.data = f"{drive_root}/data/samples"

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from src.models.cm_classifier import build_cm_classifier

    use_wav2vec2 = not args.no_wav2vec2
    use_mamba    = not args.no_mamba

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.error("Run: python data/generate_samples.py  to create sample data.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s | Mamba=%s | Wav2Vec2=%s | Enhance=%s",
                device, use_mamba, use_wav2vec2, args.enhance)

    # ── Load or extract features ─────────────────────────────────────────────
    cache_path = Path(args.cache)
    # Invalidate old cache if it doesn't have wav2vec2 key
    if cache_path.exists():
        cache = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        if "w2v2" not in cache and use_wav2vec2:
            logger.info("Cache missing wav2vec2 features; re-extracting...")
            cache_path.unlink()

    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        cache   = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        X_logmel = cache["logmel"]
        X_ssl    = cache["ssl"]
        X_w2v2   = cache.get("w2v2", torch.zeros(len(cache["ssl"]), 768))
        y        = cache["labels"]
        logger.info("Loaded %d cached samples (%d bonafide, %d spoof)",
                    len(y), (y == 0).sum().item(), (y == 1).sum().item())
    else:
        processed = load_dataset(data_dir, use_enhance=args.enhance)
        logmels, ssl_embs, w2v2_embs, labels = build_feature_tensors(
            processed, extract_wav2vec2=use_wav2vec2)
        if len(labels) == 0:
            logger.error("No features extracted. Check your audio files.")
            sys.exit(1)
        X_logmel = pad_logmels(logmels)
        X_ssl    = torch.stack(ssl_embs)
        X_w2v2   = torch.stack(w2v2_embs)
        y        = torch.tensor(labels, dtype=torch.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logmel": X_logmel, "ssl": X_ssl, "w2v2": X_w2v2, "labels": y},
                   str(cache_path))
        logger.info("Saved features to cache: %s", cache_path)

    logger.info("Feature shapes: logmel=%s  ssl=%s  wav2vec2=%s  labels=%s",
                X_logmel.shape, X_ssl.shape, X_w2v2.shape, y.shape)

    # ── Train/val split ──────────────────────────────────────────────────────
    n_total = len(y)
    n_val   = max(2, int(n_total * args.val_split))
    n_train = n_total - n_val

    dataset = TensorDataset(X_logmel, X_ssl, X_w2v2, y)
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)

    # ── Model, optimizer, loss ───────────────────────────────────────────────
    model     = build_cm_classifier(use_mamba=use_mamba).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Class imbalance: compute positive weight
    n_spoof   = float((y == 1).sum())
    n_bonafide= float((y == 0).sum())
    if n_spoof > 0 and n_bonafide > 0:
        pos_weight = torch.tensor([n_bonafide / n_spoof], device=device)
        weighted_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        weighted_criterion = criterion

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", n_params)
    logger.info("Starting training: %d epochs, batch=%d, lr=%.1e",
                args.epochs, args.batch, args.lr)

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_state    = None
    patience      = 40
    no_improve    = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train loop ───────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_logmel, batch_ssl, batch_w2v2, batch_y in train_loader:
            batch_logmel = batch_logmel.to(device)
            batch_ssl    = batch_ssl.to(device)
            batch_w2v2   = batch_w2v2.to(device)
            batch_y      = batch_y.to(device).unsqueeze(-1)

            # ── Data augmentation ─────────────────────────────────────────
            if torch.rand(1).item() > 0.3:
                batch_logmel = apply_spec_augment(batch_logmel, max_time_mask=20, max_freq_mask=10)
            if torch.rand(1).item() > 0.5:
                # Noise injection on SSL embeddings
                batch_ssl  = batch_ssl  + torch.randn_like(batch_ssl)  * 0.015
                batch_w2v2 = batch_w2v2 + torch.randn_like(batch_w2v2) * 0.015

            optimizer.zero_grad()

            # ── Forward: global + frame logits ────────────────────────────
            try:
                global_logit, frame_logits = model.forward_with_frames(
                    batch_logmel, batch_ssl,
                    batch_w2v2 if use_wav2vec2 else None,
                )
            except Exception:
                # Fallback: global only (if frame head fails on edge case)
                global_logit = model(batch_logmel, batch_ssl,
                                     batch_w2v2 if use_wav2vec2 else None)
                frame_logits = None

            # ── Primary BCE loss (global) ──────────────────────────────────
            cls_loss = weighted_criterion(global_logit, batch_y)

            # ── Frame-level auxiliary loss ────────────────────────────────
            frame_loss = torch.tensor(0.0, device=device)
            if frame_logits is not None and args.frame_weight > 0:
                # Broadcast label across all frames
                T = frame_logits.shape[1]
                frame_labels = batch_y.unsqueeze(1).expand(-1, T, -1)
                frame_loss = criterion(frame_logits, frame_labels)

            # ── SSL Agreement loss (Resemble AI insight) ──────────────────
            agreement_loss = torch.tensor(0.0, device=device)
            if use_wav2vec2 and args.agreement_weight > 0:
                # Get branch-level SSL features (pre-fusion)
                # Proxy: use projected features from each SSL branch
                wavlm_feat  = model.wavlm_branch(batch_ssl)    # [B, hidden]
                w2v2_feat   = model.wav2vec2_branch(batch_w2v2) # [B, hidden]
                agreement_loss = compute_ssl_agreement_loss(
                    wavlm_feat, w2v2_feat, batch_y)

            # ── Total loss ────────────────────────────────────────────────
            total_loss = (cls_loss
                          + args.frame_weight    * frame_loss
                          + args.agreement_weight * agreement_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += cls_loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validation loop ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        bf_correct, bf_total = 0, 0
        sp_correct, sp_total = 0, 0

        with torch.no_grad():
            for batch_logmel, batch_ssl, batch_w2v2, batch_y in val_loader:
                batch_logmel = batch_logmel.to(device)
                batch_ssl    = batch_ssl.to(device)
                batch_w2v2   = batch_w2v2.to(device)
                batch_y      = batch_y.to(device).unsqueeze(-1)

                logits = model(batch_logmel, batch_ssl,
                               batch_w2v2 if use_wav2vec2 else None)
                val_loss += criterion(logits, batch_y).item()
                preds = (torch.sigmoid(logits) >= 0.5).float()

                for p, t in zip(preds.flatten(), batch_y.flatten()):
                    if t.item() == 0:
                        bf_total += 1
                        if p.item() == 0:
                            bf_correct += 1
                    else:
                        sp_total += 1
                        if p.item() == 1:
                            sp_correct += 1

        val_loss   /= max(1, len(val_loader))
        bf_acc      = (bf_correct / bf_total)  if bf_total > 0 else 1.0
        sp_acc      = (sp_correct / sp_total)  if sp_total > 0 else 1.0
        overall_acc = (bf_correct + sp_correct) / max(1, bf_total + sp_total)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d | cls=%.4f | val_acc=%.1f%% (BF=%.1f%%, SP=%.1f%%)",
                epoch, args.epochs, train_loss, overall_acc * 100, bf_acc * 100, sp_acc * 100,
            )

        if val_loss < best_val_loss or (val_loss <= best_val_loss * 1.05 and overall_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc  = overall_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            # Persist best model immediately to Drive/disk so progress survives disconnects
            try:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, str(out_path))
                logger.info("✓ [Drive Saved] Best model updated: %s (val_loss=%.4f, val_acc=%.1f%%)",
                            out_path, best_val_loss, best_val_acc * 100)
            except Exception as save_err:
                logger.debug("Could not persist mid-training checkpoint: %s", save_err)
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 40:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # ── Save best checkpoint ─────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_state = best_state if best_state is not None else model.state_dict()
    torch.save(save_state, str(out_path))
    logger.info("Best model saved: %s  (val_loss=%.4f, val_acc=%.1f%%)",
                out_path, best_val_loss, best_val_acc * 100)

    # Also save training metadata alongside checkpoint
    meta_path = out_path.with_suffix(".json")
    import json
    meta = {
        "val_acc":     round(best_val_acc * 100, 2),
        "val_loss":    round(best_val_loss, 4),
        "architecture": "DETECT-2B-lite",
        "ssl_models":  ["wavlm-base", "wav2vec2-base-960h"] if use_wav2vec2 else ["wavlm-base"],
        "mamba_enabled": use_mamba,
        "data_dir":    str(data_dir),
        "epochs":      args.epochs,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Training metadata saved: %s", meta_path)

    print(f"\n[OK] Training complete.")
    print(f"     Checkpoint:  {out_path}")
    print(f"     Val Acc:     {best_val_acc*100:.1f}%")
    print(f"     Architecture: DETECT-2B-lite (WavLM + Wav2Vec2 + Mamba-SSM)")


if __name__ == "__main__":
    main()
