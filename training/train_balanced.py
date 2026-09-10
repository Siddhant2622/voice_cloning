"""
training/train_balanced.py
==========================
Trains the DETECT-2B-lite classifier on the balanced dataset:
  162 Bonafide human speech clips (from LibriSpeech 40 speakers)
  162 Spoof synthetic voice clips (from 8 Edge-TTS neural voices + pyttsx3)
Total: 324 balanced samples (50/50 ratio).
"""

import sys
import time
import json
import random
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import to_mono_16k
from src.features import extract_logmel, extract_ssl_embedding, extract_wav2vec2_embedding
from src.models.cm_classifier import build_cm_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_balanced")

CACHE_FILE = Path("data/balanced_features_cache.pt")
MODEL_OUT  = Path("models/cm_detect2b_v2.pt")
MAX_SECONDS = 3.0
TARGET_SR   = 16000


def load_and_trim(path: Path) -> torch.Tensor:
    """Load audio, resample to 16kHz mono, and trim/pad to MAX_SECONDS."""
    data, sr = sf.read(str(path))
    wav = torch.from_numpy(data).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav_16k = to_mono_16k(wav, sr)  # 1D tensor [T]

    max_len = int(MAX_SECONDS * TARGET_SR)
    if len(wav_16k) > max_len:
        # Take center chunk
        start = (len(wav_16k) - max_len) // 2
        wav_16k = wav_16k[start:start + max_len]
    elif len(wav_16k) < max_len:
        # Zero-pad
        wav_16k = F.pad(wav_16k, (0, max_len - len(wav_16k)))
    return wav_16k


def build_or_load_dataset(device="cpu", replay_ratio: float = 0.5, force_rebuild: bool = False):
    cache_path = Path(f"data/balanced_features_cache_replay{int(replay_ratio * 100)}.pt")
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached features from %s...", cache_path)
        data = torch.load(str(cache_path), weights_only=False)
        return data["logmel"], data["ssl"], data["w2v2"], data["labels"]

    logger.info("Building replay-augmented feature dataset (replay_ratio=%.2f)...", replay_ratio)
    bf_files = sorted(list(Path("data/large_dataset/bonafide").glob("*.flac")) + list(Path("data/large_dataset/bonafide").glob("*.wav")))[:162]
    
    clean_sp_files = sorted(list(Path("data/large_dataset/spoof").glob("*.wav")) + list(Path("data/large_dataset/spoof").glob("*.mp3")))
    replayed_sp_files = sorted(list(Path("data/large_dataset/replayed_spoof").glob("*.wav")))
    
    if len(replayed_sp_files) == 0:
        logger.warning("No replayed_spoof samples found! Run data/replay_augment.py first. Falling back to clean spoof.")
        sp_files = clean_sp_files[:len(bf_files)]
    else:
        total_sp = len(bf_files)  # 162
        num_replayed = int(round(total_sp * replay_ratio))
        num_clean = total_sp - num_replayed
        sp_files = clean_sp_files[:num_clean] + replayed_sp_files[:num_replayed]
        logger.info("Spoof composition: %d clean + %d replayed (ratio=%.2f)", num_clean, num_replayed, replay_ratio)

    logger.info("Bonafide files: %d | Total Spoof files: %d", len(bf_files), len(sp_files))
    all_files = [(f, 0) for f in bf_files] + [(f, 1) for f in sp_files]
    random.seed(42)
    random.shuffle(all_files)

    logmels, ssls, w2v2s, labels = [], [], [], []
    t0 = time.time()

    for idx, (path, label) in enumerate(all_files):
        try:
            wav = load_and_trim(path)
            mel = extract_logmel(wav, device="cpu")
            ssl = extract_ssl_embedding(wav, device=device)
            w2v = extract_wav2vec2_embedding(wav, device=device)

            logmels.append(mel)
            ssls.append(ssl)
            w2v2s.append(w2v)
            labels.append(label)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(all_files):
                elapsed = time.time() - t0
                per_sample = elapsed / (idx + 1)
                remaining = (len(all_files) - idx - 1) * per_sample
                logger.info(
                    "Features: %d/%d (%.1f%%) — %.2fs/sample, ~%.0fs remaining",
                    idx + 1, len(all_files), 100 * (idx + 1) / len(all_files), per_sample, remaining
                )
        except Exception as e:
            logger.warning("Error processing %s: %s", path.name, e)

    # Pad/crop logmel to common length
    target_frames = int(MAX_SECONDS * 100)  # ~300 frames
    fixed_logmels = []
    for m in logmels:
        if m.shape[-1] < target_frames:
            m = F.pad(m, (0, target_frames - m.shape[-1]))
        else:
            m = m[:, :target_frames]
        fixed_logmels.append(m)

    X_logmel = torch.stack(fixed_logmels).float()
    X_ssl    = torch.stack(ssls).float()
    X_w2v2   = torch.stack(w2v2s).float()
    y        = torch.tensor(labels, dtype=torch.float32)

    logger.info("Saving features cache to %s...", cache_path)
    torch.save({"logmel": X_logmel, "ssl": X_ssl, "w2v2": X_w2v2, "labels": y}, str(cache_path))
    return X_logmel, X_ssl, X_w2v2, y


def train(replay_ratio: float = 0.5, force_rebuild: bool = False, epochs: int = 80):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s | Replay Ratio: %.2f", device, replay_ratio)

    X_logmel, X_ssl, X_w2v2, y = build_or_load_dataset(device=device, replay_ratio=replay_ratio, force_rebuild=force_rebuild)
    logger.info("Dataset shape: logmel=%s, ssl=%s, w2v2=%s, y=%s",
                X_logmel.shape, X_ssl.shape, X_w2v2.shape, y.shape)

    # Split 80/20 train/val
    dataset = TensorDataset(X_logmel, X_ssl, X_w2v2, y)
    n_val = int(len(y) * 0.2)
    n_train = len(y) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = build_cm_classifier(use_mamba=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None

    logger.info("Training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for b_mel, b_ssl, b_w2v, b_y in train_loader:
            b_mel = b_mel.to(device)
            b_ssl = b_ssl.to(device)
            b_w2v = b_w2v.to(device)
            b_y   = b_y.to(device).unsqueeze(-1)

            # SSL agreement auxiliary loss
            sim = F.cosine_similarity(b_ssl, b_w2v, dim=-1)
            agr_loss = (b_y.squeeze() * sim + (1 - b_y.squeeze()) * (1 - sim)).mean()

            optimizer.zero_grad()
            global_logits, frame_logits = model.forward_with_frames(b_mel, b_ssl, b_w2v)
            cls_loss = bce(global_logits, b_y)

            # Frame loss
            frame_labels = b_y.view(-1, 1, 1).expand(-1, frame_logits.shape[1], 1)
            frame_loss = bce(frame_logits, frame_labels)

            total_loss = cls_loss + 0.2 * frame_loss + 0.3 * agr_loss
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        bf_correct, bf_total = 0, 0
        sp_correct, sp_total = 0, 0

        with torch.no_grad():
            for b_mel, b_ssl, b_w2v, b_y in val_loader:
                b_mel = b_mel.to(device)
                b_ssl = b_ssl.to(device)
                b_w2v = b_w2v.to(device)
                b_y   = b_y.to(device).unsqueeze(-1)

                g_logits = model(b_mel, b_ssl, b_w2v)
                loss = bce(g_logits, b_y)
                val_loss += loss.item()

                probs = torch.sigmoid(g_logits).squeeze(-1)
                preds = (probs >= 0.5).float()

                for pred, target in zip(preds, b_y.squeeze(-1)):
                    if target.item() == 0:
                        bf_total += 1
                        if pred.item() == 0:
                            bf_correct += 1
                    else:
                        sp_total += 1
                        if pred.item() == 1:
                            sp_correct += 1

        val_loss /= len(val_loader)
        bf_acc = bf_correct / max(1, bf_total)
        sp_acc = sp_correct / max(1, sp_total)
        val_acc = (bf_correct + sp_correct) / max(1, bf_total + sp_total)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.1f%% (BF=%.1f%%, SP=%.1f%%)",
                epoch, epochs, train_loss / len(train_loader), val_loss,
                val_acc * 100, bf_acc * 100, sp_acc * 100
            )

        if val_loss < best_val_loss and (bf_acc >= 0.90 and sp_acc >= 0.90):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save checkpoint
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    save_state = best_state if best_state is not None else model.state_dict()
    torch.save(save_state, str(MODEL_OUT))
    logger.info("Saved best model to %s (val_acc=%.1f%%)", MODEL_OUT, best_val_acc * 100)

    # Save json metadata
    meta = {
        "val_acc": round(best_val_acc * 100, 2),
        "val_loss": round(best_val_loss, 4),
        "architecture": "DETECT-2B-lite (Presentation-Gap Hardened)",
        "ssl_models": ["wavlm-base", "wav2vec2-base-960h"],
        "mamba_enabled": False,
        "replay_ratio": replay_ratio,
        "presentation_gap_mitigation": "hybrid_replay_augmentation",
        "data_dir": f"data/large_dataset (balanced 162 BF + 162 SP [ratio={replay_ratio}])",
        "epochs": epochs
    }
    with open(MODEL_OUT.with_suffix(".json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    print("\n" + "="*50)
    print("TRAINING SUCCESSFUL!")
    print(f"Model saved: {MODEL_OUT}")
    print(f"Val Accuracy: {best_val_acc * 100:.1f}%")
    print(f"Replay Ratio: {replay_ratio * 100:.0f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train balanced DETECT-2B classifier with replay augmentation")
    parser.add_argument("--replay-ratio", type=float, default=0.5, help="Ratio of replayed spoof samples (0.0 to 1.0)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding feature cache")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    args = parser.parse_args()

    train(replay_ratio=args.replay_ratio, force_rebuild=args.force_rebuild, epochs=args.epochs)
