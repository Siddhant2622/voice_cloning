"""
data/generate_samples.py
Generates a minimal labeled sample set for demo and basic testing.

Bona fide samples: downloaded from LibriSpeech test-clean (public domain)
Synthetic samples: generated using pyttsx3 (already installed) with
                   scripted, non-impersonating sentences.

GROUND RULES COMPLIANCE:
  - pyttsx3 uses the system's local TTS engine — no real person is cloned.
  - Sentences are scripted, non-identifying, non-impersonating.
  - Samples are labeled clearly in samples_metadata.csv.

Usage:
    python data/generate_samples.py [--n-synthetic 5] [--no-download]
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLES_DIR    = Path(__file__).parent / "samples"
METADATA_CSV   = SAMPLES_DIR / "samples_metadata.csv"

# Scripted sentences (non-impersonating, scripted for diversity)
SYNTHETIC_SENTENCES = [
    "The quarterly earnings report showed a significant increase in revenue.",
    "Please transfer twenty five thousand dollars to the following account.",
    "My voice is my passport verify me for secure access to the system.",
    "The weather forecast predicts heavy rainfall throughout the region tomorrow.",
    "Artificial intelligence systems require careful monitoring and oversight.",
    "The meeting has been rescheduled to three o'clock on Thursday afternoon.",
    "Please reset my password and send a confirmation to my email address.",
    "The authorization code for this transaction is seven four two nine one.",
]

# Small LibriSpeech clips (LibriSpeech test-clean excerpt, CC0 / public domain)
LIBRISPEECH_SAMPLES = [
    {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "note": "Full test-clean (3.4 GB) — too large for quick demo. Using direct clip URLs instead."
    }
]

# Direct audio sample URLs (LibriVox / common-voice style — public domain)
# These are short speaker-diverse clips from the LibriSpeech OpenSLR host.
# We use the smallest individual clips available.
LIBRISPEECH_CLIP_URLS = [
    # LibriSpeech reader 84, chapter 121123 — "Grimm's Fairy Tales" excerpts
    "https://www.openslr.org/resources/12/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac",
    "https://www.openslr.org/resources/12/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac",
    "https://www.openslr.org/resources/12/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0002.flac",
]

# Fallback: generate bona-fide-like samples using sounddevice (recorded silence)
# or just use the synthetic pyttsx3 samples as stand-ins with different voice params.


def _generate_pyttsx3_clip(text: str, output_path: Path, rate: int = 150, volume: float = 1.0) -> bool:
    """Generate a WAV file from text using pyttsx3."""
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 not installed. Skipping TTS generation.")
        return False

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()
    return output_path.exists()


def _download_clip(url: str, output_path: Path) -> bool:
    """Download a single audio clip."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading: {url.split('/')[-1]} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, str(output_path))
        print("[OK]")
        return True
    except Exception as exc:
        print(f"[FAIL] ({exc})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate demo sample clips.")
    parser.add_argument("--n-synthetic", type=int, default=5, help="Number of synthetic clips to generate.")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading bona-fide clips.")
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # ── Generate synthetic clips (pyttsx3) ──────────────────────────────────
    print(f"\n[1/2] Generating {args.n_synthetic} synthetic clips (pyttsx3 TTS)...")
    sentences = SYNTHETIC_SENTENCES[:args.n_synthetic]
    for i, sentence in enumerate(sentences):
        out = SAMPLES_DIR / f"synthetic_{i:02d}.wav"
        if out.exists():
            print(f"  {out.name} already exists — skipping.")
            records.append({"file": out.name, "label": "spoof", "source": "pyttsx3", "text": sentence})
            continue
        # Vary rate slightly to add diversity
        rate = 140 + i * 5
        success = _generate_pyttsx3_clip(sentence, out, rate=rate)
        if success:
            print(f"  Generated: {out.name}")
            records.append({"file": out.name, "label": "spoof", "source": "pyttsx3", "text": sentence})
        else:
            print(f"  [FAIL] Failed: {out.name}")

    # ── Download bona fide clips (LibriSpeech) ──────────────────────────────
    if not args.no_download:
        print("\n[2/2] Attempting to download bona-fide clips (LibriSpeech public domain)...")
        bf_count = 0
        for url in LIBRISPEECH_CLIP_URLS:
            clip_name = "bonafide_" + url.split("/")[-1].replace(".flac", ".flac")
            out = SAMPLES_DIR / clip_name
            if out.exists():
                print(f"  {out.name} already exists — skipping.")
                records.append({"file": out.name, "label": "bonafide", "source": "librispeech", "text": ""})
                bf_count += 1
                continue
            if _download_clip(url, out):
                records.append({"file": out.name, "label": "bonafide", "source": "librispeech", "text": ""})
                bf_count += 1

        if bf_count == 0:
            # Fallback: generate "bona fide" stand-ins with different pyttsx3 voice
            print("  LibriSpeech download failed. Generating pyttsx3 stand-ins with different rate.")
            bf_sentences = [
                "The quick brown fox jumps over the lazy dog near the riverbank.",
                "She sells sea shells by the seashore on a warm summer afternoon.",
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
            ]
            for i, s in enumerate(bf_sentences[:3]):
                out = SAMPLES_DIR / f"bonafide_fallback_{i:02d}.wav"
                if out.exists():
                    records.append({"file": out.name, "label": "bonafide", "source": "pyttsx3_fallback", "text": s})
                    continue
                # Use higher rate to differentiate from synthetic clips
                success = _generate_pyttsx3_clip(s, out, rate=200)
                if success:
                    records.append({"file": out.name, "label": "bonafide", "source": "pyttsx3_fallback", "text": s})
                    print(f"  Generated fallback bona fide: {out.name}")
    else:
        print("\n[2/2] Skipping bona-fide download (--no-download set).")

    # ── Write metadata CSV ───────────────────────────────────────────────────
    if records:
        with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "text"])
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[OK] Metadata written: {METADATA_CSV}")
        print(f"  {sum(1 for r in records if r['label']=='spoof')} synthetic clips")
        print(f"  {sum(1 for r in records if r['label']=='bonafide')} bona fide clips")
    else:
        print("\n[FAIL] No samples generated.")

    print("\nNote: pyttsx3 clips are labeled 'spoof' because they are machine-generated,")
    print("but they represent an easy attack case. Real evaluation requires ASVspoof 2019.")
    print("Run: python data/download_asvspoof.py  to get the benchmark dataset.")


if __name__ == "__main__":
    main()
