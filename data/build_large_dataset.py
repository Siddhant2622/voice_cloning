"""
data/build_large_dataset.py
============================
Downloads and assembles a large training dataset:

  Bonafide (Real Human Speech):
    - LibriSpeech dev-clean  (~337 MB, 2703 clips, public domain)
    - Common Voice EN subset (optional, requires Mozilla account)

  Spoof (Synthetic AI Speech):
    - edge-tts: 5 neural voices × 15 sentences = 75 clips
    - gTTS: 15 clips
    - pyttsx3: 15 clips  (already in data/samples/)

Total target: ~3000 bonafide + 100+ spoof clips
"""

import argparse
import asyncio
import csv
import io
import logging
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

DATASET_DIR = Path(__file__).parent / "large_dataset"

SENTENCES = [
    "The quarterly earnings report showed a significant increase in revenue this year.",
    "Please transfer twenty five thousand dollars to the following account number.",
    "My voice is my passport verify me for secure access to the financial system.",
    "The weather forecast predicts heavy rainfall throughout the eastern region tomorrow.",
    "Artificial intelligence systems require careful monitoring and human oversight.",
    "The meeting has been rescheduled to three o clock on Thursday afternoon next week.",
    "Please reset my password and send a confirmation link to my email address.",
    "The authorization code for this transaction is seven four two nine one six.",
    "Security systems must detect synthetic speech with the highest possible accuracy.",
    "Voice cloning technology requires robust countermeasure detection systems.",
    "Generative models produce audio with subtle artifacts in the frequency spectrum.",
    "Real human speech has natural prosodic variation and vocal tract resonance patterns.",
    "The algorithm achieved ninety nine percent accuracy on the standard benchmark dataset.",
    "Please confirm your identity before proceeding with any financial transaction.",
    "Machine learning models require diverse training data for robust generalization.",
    "The neural network was trained on thousands of hours of clean speech recordings.",
    "Biometric authentication systems are becoming increasingly popular in banking.",
    "The research team published their findings in a peer-reviewed scientific journal.",
    "Advanced voice synthesis technology can replicate any speaker with high fidelity.",
    "Deepfake detection requires analysis of both spectral and temporal audio features.",
]

EDGE_TTS_VOICES = [
    ("en-US-GuyNeural",      "edge_us_male"),
    ("en-US-JennyNeural",    "edge_us_female"),
    ("en-GB-RyanNeural",     "edge_gb_male"),
    ("en-GB-SoniaNeural",    "edge_gb_female"),
    ("en-AU-WilliamNeural",  "edge_au_male"),
    ("en-AU-NatashaNeural",  "edge_au_female"),
    ("en-IN-NeerjaNeural",   "edge_in_female"),
    ("en-CA-LiamNeural",     "edge_ca_male"),
]


def download_with_progress(url: str, dest: Path) -> bool:
    """Download with progress bar using urllib."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading: %s", url.split("/")[-1])

        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                mb = count * block_size / 1_048_576
                total_mb = total_size / 1_048_576
                print(f"\r  {pct:3d}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=progress)
        print()
        return True
    except Exception as exc:
        print(f"\n  [FAIL] {exc}")
        return False


def download_librispeech_dev_clean(out_dir: Path) -> list:
    """
    Download LibriSpeech dev-clean (~337 MB).
    Returns list of (wav_path, label=0) tuples.
    Public domain (CC0). No registration required.
    """
    tar_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    tar_path = out_dir / "dev-clean.tar.gz"
    extract_dir = out_dir / "LibriSpeech"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not extract_dir.exists():
        if not tar_path.exists():
            logger.info("Downloading LibriSpeech dev-clean (~337 MB)...")
            ok = download_with_progress(tar_url, tar_path)
            if not ok:
                logger.error("LibriSpeech download failed. Check your internet connection.")
                return []

        logger.info("Extracting LibriSpeech dev-clean...")
        with tarfile.open(str(tar_path), "r:gz") as tf:
            tf.extractall(str(out_dir))
        logger.info("[OK] LibriSpeech dev-clean extracted.")

    # Find all FLAC files
    flac_files = list(extract_dir.glob("**/*.flac"))
    logger.info("Found %d LibriSpeech bonafide clips.", len(flac_files))
    return flac_files


async def generate_edge_tts_clip(text: str, voice: str, out_path: Path) -> bool:
    """Generate one Edge TTS clip using soundfile for saving (avoids torchaudio/torchcodec)."""
    try:
        import edge_tts
        import numpy as np
        import soundfile as sf
        import io as io_module

        # Generate MP3 bytes in memory
        comm = edge_tts.Communicate(text=text, voice=voice)
        mp3_path = out_path.with_suffix(".mp3")
        await comm.save(str(mp3_path))

        if not mp3_path.exists():
            return False

        # Convert MP3 -> WAV using soundfile (or pydub if available)
        try:
            # Try soundfile directly (works with some MP3s)
            data, sr = sf.read(str(mp3_path))
            # Resample to 16kHz if needed
            if sr != 16000:
                import scipy.signal
                n_samples = int(len(data) * 16000 / sr)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = scipy.signal.resample(data, n_samples)
            elif data.ndim > 1:
                data = data.mean(axis=1)
            sf.write(str(out_path), data.astype("float32"), 16000)
            mp3_path.unlink(missing_ok=True)
            return True
        except Exception:
            # Fallback: use pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(str(mp3_path))
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(str(out_path), format="wav")
                mp3_path.unlink(missing_ok=True)
                return True
            except Exception as e2:
                # Last resort: keep MP3, rename as mp3
                logger.warning("Could not convert %s to WAV: %s", mp3_path.name, e2)
                mp3_path.rename(out_path.with_suffix(".mp3"))
                return False
    except Exception as exc:
        logger.warning("edge-tts failed for %s: %s", out_path.name, exc)
        return False


async def generate_all_edge_tts(spoof_dir: Path, voices=None, sentences=None) -> list:
    """Generate edge-tts clips for all voice × sentence combinations."""
    if voices is None:
        voices = EDGE_TTS_VOICES
    if sentences is None:
        sentences = SENTENCES

    spoof_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for voice_name, prefix in voices:
        logger.info("Generating edge-tts clips: %s (%s)...", prefix, voice_name)
        for i, text in enumerate(sentences):
            out_path = spoof_dir / f"{prefix}_{i:03d}.wav"
            if out_path.exists():
                records.append((out_path, text, prefix))
                continue
            ok = await generate_edge_tts_clip(text, voice_name, out_path)
            if ok:
                records.append((out_path, text, prefix))
                print(f"  [OK] {out_path.name}")
            else:
                # Check for MP3 fallback
                mp3_fb = out_path.with_suffix(".mp3")
                if mp3_fb.exists():
                    records.append((mp3_fb, text, prefix))

    logger.info("edge-tts: %d clips generated.", len(records))
    return records


def generate_gtts_clips(spoof_dir: Path, sentences=None) -> list:
    """Generate gTTS clips."""
    if sentences is None:
        sentences = SENTENCES

    spoof_dir.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        from gtts import gTTS
        import soundfile as sf
        import scipy.signal

        logger.info("Generating gTTS clips...")
        for i, text in enumerate(sentences):
            out_path = spoof_dir / f"gtts_{i:03d}.wav"
            if out_path.exists():
                records.append((out_path, text, "gtts"))
                continue
            try:
                mp3_path = out_path.with_suffix(".mp3")
                tts = gTTS(text=text, lang="en", slow=False)
                tts.save(str(mp3_path))
                try:
                    data, sr = sf.read(str(mp3_path))
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != 16000:
                        n = int(len(data) * 16000 / sr)
                        data = scipy.signal.resample(data, n)
                    sf.write(str(out_path), data.astype("float32"), 16000)
                    mp3_path.unlink(missing_ok=True)
                    records.append((out_path, text, "gtts"))
                    print(f"  [OK] {out_path.name}")
                except Exception:
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_mp3(str(mp3_path))
                        audio = audio.set_frame_rate(16000).set_channels(1)
                        audio.export(str(out_path), format="wav")
                        mp3_path.unlink(missing_ok=True)
                        records.append((out_path, text, "gtts"))
                    except Exception:
                        mp3_path.rename(out_path.with_suffix(".mp3"))
            except Exception as exc:
                logger.warning("gTTS failed for clip %d: %s", i, exc)
    except ImportError:
        logger.warning("gTTS not installed.")

    logger.info("gTTS: %d clips generated.", len(records))
    return records


def generate_pyttsx3_clips(spoof_dir: Path, sentences=None) -> list:
    """Generate pyttsx3 classical vocoder clips."""
    if sentences is None:
        sentences = SENTENCES

    spoof_dir.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")

        logger.info("Generating pyttsx3 clips (%d voices available)...", len(voices) if voices else 0)

        # Use up to 2 different voices for diversity
        voice_ids = [voices[0].id] if voices else [None]
        if len(voices) > 1:
            voice_ids.append(voices[1].id)

        for vi, vid in enumerate(voice_ids):
            if vid:
                engine.setProperty("voice", vid)
            for i, text in enumerate(sentences):
                out_path = spoof_dir / f"pyttsx3_v{vi}_{i:03d}.wav"
                if out_path.exists():
                    records.append((out_path, text, f"pyttsx3_v{vi}"))
                    continue
                try:
                    engine.setProperty("rate", 140 + i * 3)
                    engine.save_to_file(text, str(out_path))
                    engine.runAndWait()
                    if out_path.exists():
                        records.append((out_path, text, f"pyttsx3_v{vi}"))
                        print(f"  [OK] {out_path.name}")
                except Exception as exc:
                    logger.warning("pyttsx3 failed for clip %d: %s", i, exc)
    except Exception as exc:
        logger.warning("pyttsx3 failed: %s", exc)

    logger.info("pyttsx3: %d clips generated.", len(records))
    return records


def write_metadata_csv(csv_path: Path, bonafide_paths: list, spoof_records: list):
    """Write samples_metadata.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for p in bonafide_paths:
        rows.append({
            "file": str(p.relative_to(csv_path.parent)),
            "label": "bonafide",
            "source": "librispeech_dev_clean",
            "text": "",
        })

    for p, text, source in spoof_records:
        rel = str(p.relative_to(csv_path.parent)) if p.is_absolute() else str(p)
        try:
            rel = str(p.relative_to(csv_path.parent))
        except ValueError:
            rel = f"spoof/{p.name}"
        rows.append({
            "file": rel,
            "label": "spoof",
            "source": source,
            "text": text,
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "text"])
        writer.writeheader()
        writer.writerows(rows)

    n_bf = sum(1 for r in rows if r["label"] == "bonafide")
    n_sp = sum(1 for r in rows if r["label"] == "spoof")
    logger.info("Metadata CSV written: %s", csv_path)
    logger.info("  Bonafide: %d | Spoof: %d | Total: %d", n_bf, n_sp, len(rows))
    return rows


async def main():
    parser = argparse.ArgumentParser(description="Build large training dataset.")
    parser.add_argument("--out",            default="data/large_dataset",  help="Output directory.")
    parser.add_argument("--no-librispeech", action="store_true",           help="Skip LibriSpeech download.")
    parser.add_argument("--no-edge-tts",    action="store_true",           help="Skip edge-tts generation.")
    parser.add_argument("--no-gtts",        action="store_true",           help="Skip gTTS generation.")
    parser.add_argument("--max-bonafide",   type=int, default=3000,        help="Max LibriSpeech clips to use.")
    args = parser.parse_args()

    out_dir   = Path(args.out)
    spoof_dir = out_dir / "spoof"
    bf_dir    = out_dir / "bonafide"
    out_dir.mkdir(parents=True, exist_ok=True)
    bf_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    bonafide_paths = []
    spoof_records  = []

    # ── Step 1: LibriSpeech real human speech ──────────────────────────────
    if not args.no_librispeech:
        logger.info("=== Downloading LibriSpeech dev-clean (real human speech) ===")
        flac_files = download_librispeech_dev_clean(out_dir)
        # Cap at max_bonafide
        flac_files = flac_files[:args.max_bonafide]
        logger.info("Using %d bonafide clips from LibriSpeech.", len(flac_files))
        bonafide_paths = flac_files
    else:
        # Look for existing bonafide files
        existing = list(bf_dir.glob("**/*.wav")) + list(bf_dir.glob("**/*.flac"))
        bonafide_paths = existing[:args.max_bonafide]
        logger.info("Using %d existing bonafide clips.", len(bonafide_paths))

    # ── Step 2: edge-tts synthetic clips ──────────────────────────────────
    if not args.no_edge_tts:
        logger.info("=== Generating edge-tts synthetic clips (8 voices × 20 sentences) ===")
        edge_records = await generate_all_edge_tts(spoof_dir)
        spoof_records.extend(edge_records)

    # ── Step 3: gTTS clips ────────────────────────────────────────────────
    if not args.no_gtts:
        logger.info("=== Generating gTTS clips ===")
        gtts_records = generate_gtts_clips(spoof_dir)
        spoof_records.extend(gtts_records)

    # ── Step 4: pyttsx3 clips ─────────────────────────────────────────────
    logger.info("=== Generating pyttsx3 clips ===")
    py_records = generate_pyttsx3_clips(spoof_dir)
    spoof_records.extend(py_records)

    # ── Step 5: Copy existing enhanced_samples spoof ──────────────────────
    existing_spoof = Path("data/enhanced_samples/spoof")
    if existing_spoof.exists():
        for f in existing_spoof.glob("*.wav"):
            dst = spoof_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
                spoof_records.append((dst, "", "existing"))

    # ── Step 6: Write metadata CSV ─────────────────────────────────────────
    csv_path = out_dir / "samples_metadata.csv"
    rows = write_metadata_csv(csv_path, bonafide_paths, spoof_records)

    print(f"\n[OK] Large dataset built at: {out_dir}")
    print(f"     Bonafide: {sum(1 for r in rows if r['label']=='bonafide')}")
    print(f"     Spoof:    {sum(1 for r in rows if r['label']=='spoof')}")
    print(f"\nNext: python training/train_cm.py --data {out_dir} --epochs 150 --batch 16")


if __name__ == "__main__":
    asyncio.run(main())
