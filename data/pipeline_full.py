"""
data/pipeline_full.py
======================
Master pipeline: build large dataset → upload to Drive → retrain.

Run this ONE script to do everything:

    python data/pipeline_full.py

It will:
  1. Download LibriSpeech dev-clean (~337 MB, 2700 real speech clips)
  2. Generate 160+ synthetic clips (edge-tts × 8 voices, gTTS, pyttsx3)
  3. Upload everything to Google Drive → VoiceGuard-Dataset/
  4. Retrain CM model on the full dataset (DETECT-2B-lite architecture)
  5. Save new checkpoint as models/cm_detect2b_v2.pt

No flags needed — just run it.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

DATA_DIR      = Path("data/large_dataset")
CHECKPOINT    = Path("models/cm_detect2b_v2.pt")
CACHE_PATH    = Path("data/features_cache_large.pt")
SECRETS_FILE  = Path("client_secrets.json")
EPOCHS        = 150
BATCH_SIZE    = 8


async def step1_build_dataset():
    """Download LibriSpeech + generate synthetic data."""
    from data.build_large_dataset import (
        download_librispeech_dev_clean,
        generate_all_edge_tts,
        generate_gtts_clips,
        generate_pyttsx3_clips,
        write_metadata_csv,
    )
    import shutil

    out_dir   = DATA_DIR
    spoof_dir = out_dir / "spoof"
    bf_dir    = out_dir / "bonafide"
    out_dir.mkdir(parents=True, exist_ok=True)
    bf_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1: Building large dataset")
    print("="*60)

    # ── Bonafide: LibriSpeech ────────────────────────────────────────────
    print("\n[1/4] Downloading LibriSpeech dev-clean (real human speech)...")
    flac_files = download_librispeech_dev_clean(out_dir)
    flac_files = flac_files[:3000]
    print(f"  → {len(flac_files)} bonafide clips ready")

    # ── Spoof: edge-tts ──────────────────────────────────────────────────
    print("\n[2/4] Generating edge-tts clips (8 voices × 20 sentences)...")
    spoof_records = []
    edge_records = await generate_all_edge_tts(spoof_dir)
    spoof_records.extend(edge_records)
    print(f"  → {len(edge_records)} edge-tts clips")

    # ── Spoof: gTTS ──────────────────────────────────────────────────────
    print("\n[3/4] Generating gTTS clips...")
    gtts_records = generate_gtts_clips(spoof_dir)
    spoof_records.extend(gtts_records)
    print(f"  → {len(gtts_records)} gTTS clips")

    # ── Spoof: pyttsx3 ───────────────────────────────────────────────────
    print("\n[4/4] Generating pyttsx3 clips...")
    py_records = generate_pyttsx3_clips(spoof_dir)
    spoof_records.extend(py_records)
    print(f"  → {len(py_records)} pyttsx3 clips")

    # ── Copy existing spoof from enhanced_samples ─────────────────────────
    existing_spoof = Path("data/enhanced_samples/spoof")
    if existing_spoof.exists():
        for f in existing_spoof.glob("*.wav"):
            dst = spoof_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                spoof_records.append((dst, "", "existing"))

    # ── Write metadata ────────────────────────────────────────────────────
    csv_path = out_dir / "samples_metadata.csv"
    rows = write_metadata_csv(csv_path, flac_files, spoof_records)

    n_bf = sum(1 for r in rows if r["label"] == "bonafide")
    n_sp = sum(1 for r in rows if r["label"] == "spoof")
    print(f"\n  ✓ Dataset ready: {n_bf} bonafide + {n_sp} spoof = {n_bf+n_sp} total")
    return n_bf, n_sp


def step2_upload_to_drive():
    """Upload dataset to Google Drive (requires client_secrets.json)."""
    print("\n" + "="*60)
    print("STEP 2: Uploading to Google Drive")
    print("="*60)

    if not SECRETS_FILE.exists():
        print("\n  ⚠  client_secrets.json not found.")
        print("  Skipping Drive upload.")
        print()
        print("  To upload later, follow these steps:")
        print("  1. Go to: https://console.cloud.google.com/apis/credentials")
        print("  2. Create Project → Enable Google Drive API")
        print("  3. Create OAuth 2.0 Desktop credentials → Download JSON")
        print("  4. Rename to 'client_secrets.json' and place in:")
        print(f"     {Path.cwd()}")
        print("  5. Run: python data/upload_to_drive.py --data data/large_dataset")
        return None

    try:
        from data.upload_to_drive import get_drive_client, upload_dataset
        drive = get_drive_client(str(SECRETS_FILE))
        link = upload_dataset(drive, DATA_DIR, root_folder_name="VoiceGuard-Dataset")
        print(f"\n  ✓ Uploaded to Google Drive: {link}")
        return link
    except Exception as exc:
        logger.error("Drive upload failed: %s", exc)
        print(f"  ✗ Drive upload failed: {exc}")
        print("  Continuing to training...")
        return None


def step3_train():
    """Retrain DETECT-2B model on the large dataset."""
    print("\n" + "="*60)
    print("STEP 3: Training DETECT-2B-lite on large dataset")
    print("="*60)

    cmd = [
        sys.executable, "training/train_cm.py",
        "--data",        str(DATA_DIR),
        "--epochs",      str(EPOCHS),
        "--batch",       str(BATCH_SIZE),
        "--out",         str(CHECKPOINT),
        "--cache",       str(CACHE_PATH),
        "--no-mamba",                    # faster on CPU; remove for GPU with mamba-ssm
        "--agreement-weight", "0.3",
        "--frame-weight",     "0.2",
    ]

    print(f"  Running: {' '.join(cmd)}\n")

    # Stream output in real-time
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode == 0 or CHECKPOINT.exists():
        print(f"\n  ✓ Training complete! Checkpoint: {CHECKPOINT}")
        return True
    else:
        print(f"\n  ✗ Training failed (exit code {proc.returncode})")
        return False


async def main():
    print("\n" + "█"*60)
    print("  VoiceGuard — Full Training Pipeline")
    print("  Dataset: LibriSpeech + edge-tts + gTTS + pyttsx3")
    print("  Architecture: DETECT-2B-lite (WavLM + Wav2Vec2)")
    print("█"*60)

    # Step 1: Build dataset
    n_bf, n_sp = await step1_build_dataset()

    # Step 2: Upload to Drive
    drive_link = step2_upload_to_drive()

    # Step 3: Train
    success = step3_train()

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Dataset:     {n_bf} bonafide + {n_sp} spoof clips")
    if drive_link:
        print(f"  Drive URL:   {drive_link}")
    print(f"  Checkpoint:  {CHECKPOINT}")
    print()
    if success:
        print("  ► To use the new model, update your .env:")
        print(f"    CM_CHECKPOINT={CHECKPOINT}")
        print()
        print("  ► Start the backend:")
        print("    python backend/server.py")


if __name__ == "__main__":
    asyncio.run(main())
