"""
training/prepare_data_with_enhance.py
=======================================
Prepares an enhanced training dataset using Resemble Enhance.

This script:
  1. Runs Resemble Enhance denoiser on all bonafide training clips
  2. Creates clean + augmented versions for richer training distribution
  3. Generates diverse synthetic clips using multiple TTS systems
     (gTTS, edge-tts, pyttsx3, Coqui-TTS) to train against many generators

Usage:
    # Install enhance first:
    pip install resemble-enhance --upgrade

    # Prepare dataset:
    python training/prepare_data_with_enhance.py --in-dir data/samples --out-dir data/enhanced_samples

    # Generate synthetic data from multiple TTS:
    python training/prepare_data_with_enhance.py --generate-synth --out-dir data/samples

Inspired by Resemble AI's approach of training on 44.1kHz high-quality
clean speech with multi-generator synthetic coverage.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")

SYNTH_SENTENCES = [
    "The quarterly earnings report showed a significant increase in revenue.",
    "Please transfer twenty five thousand dollars to the following account.",
    "My voice is my passport verify me for secure access to the system.",
    "The weather forecast predicts heavy rainfall throughout the region tomorrow.",
    "Artificial intelligence systems require careful monitoring and oversight.",
    "The meeting has been rescheduled to three o clock on Thursday afternoon.",
    "Please reset my password and send a confirmation to my email address.",
    "The authorization code for this transaction is seven four two nine one.",
    "Security systems must detect synthetic speech with high accuracy.",
    "Voice cloning technology requires robust countermeasure systems.",
    "Generative models produce audio with subtle artifacts in the spectrum.",
    "Real human speech has natural prosodic variation and vocal tract resonance.",
    "The algorithm achieved ninety nine percent accuracy on the benchmark dataset.",
    "Please confirm your identity before proceeding with the transaction.",
    "Machine learning models require diverse training data for generalization.",
]


def denoise_with_resemble_enhance(wav_path: Path, out_path: Path, sr_out: int = 16000) -> bool:
    """
    Denoise a WAV file using resemble-enhance and save to out_path.
    Returns True on success.
    """
    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import enhance as resemble_enhance_fn

        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # stereo → mono

        device = "cuda" if torch.cuda.is_available() else "cpu"
        enhanced, out_sr = resemble_enhance_fn(
            wav, sr, device=device, nfe=16, denoise_only=True
        )

        # Resample to 16kHz if needed
        if out_sr != sr_out:
            enhanced = torchaudio.functional.resample(
                enhanced if enhanced.dim() == 2 else enhanced.unsqueeze(0),
                out_sr, sr_out
            )

        if enhanced.dim() == 1:
            enhanced = enhanced.unsqueeze(0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), enhanced.cpu(), sr_out)
        return True
    except Exception as exc:
        logger.warning("resemble-enhance failed for %s: %s", wav_path.name, exc)
        return False


def generate_gtts(text: str, out_path: Path, lang: str = "en") -> bool:
    """Generate speech using Google TTS (gTTS)."""
    try:
        from gtts import gTTS
        import torchaudio
        import io
        import torch

        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as mp3 then convert
        mp3_path = out_path.with_suffix(".mp3")
        with open(mp3_path, "wb") as f:
            f.write(mp3_buf.read())

        # Convert to 16kHz wav
        wav, sr = torchaudio.load(str(mp3_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        torchaudio.save(str(out_path), wav, 16000)
        mp3_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.warning("gTTS failed: %s", exc)
        return False


def generate_edge_tts(text: str, out_path: Path, voice: str = "en-US-GuyNeural") -> bool:
    """Generate speech using Microsoft Edge TTS (neural HiFi-GAN quality)."""
    try:
        import subprocess
        import sys
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path = out_path.with_suffix(".mp3")
        result = subprocess.run(
            [sys.executable, "-m", "edge_tts",
             "--voice", voice, "--text", text, "--write-media", str(mp3_path)],
            capture_output=True, timeout=30
        )
        if result.returncode != 0 or not mp3_path.exists():
            return False

        import torchaudio
        wav, sr = torchaudio.load(str(mp3_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        torchaudio.save(str(out_path), wav, 16000)
        mp3_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.warning("edge-tts failed: %s", exc)
        return False


def generate_pyttsx3(text: str, out_path: Path, rate: int = 150) -> bool:
    """Generate speech using pyttsx3 (classical vocoder)."""
    try:
        import pyttsx3
        out_path.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        return out_path.exists()
    except Exception as exc:
        logger.warning("pyttsx3 failed: %s", exc)
        return False


def generate_coqui_tts(text: str, out_path: Path) -> bool:
    """Generate speech using Coqui TTS (VITS neural vocoder)."""
    try:
        from TTS.api import TTS
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tts = TTS("tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
        tts.tts_to_file(text=text, file_path=str(out_path))
        return out_path.exists()
    except Exception as exc:
        logger.warning("Coqui TTS failed: %s", exc)
        return False


def prepare_enhanced_bonafide(in_dir: Path, out_dir: Path) -> list:
    """
    Process bonafide samples through Resemble Enhance.
    Returns list of (out_path, label) records.
    """
    records = []
    bonafide_dir = in_dir / "bonafide"
    if not bonafide_dir.exists():
        # Try metadata CSV
        meta_csv = in_dir / "samples_metadata.csv"
        if meta_csv.exists():
            with open(meta_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                bonafide_paths = [
                    in_dir / row["file"]
                    for row in reader
                    if row["label"] == "bonafide" and (in_dir / row["file"]).exists()
                ]
        else:
            bonafide_paths = []
    else:
        bonafide_paths = list(bonafide_dir.glob("**/*.wav")) + list(bonafide_dir.glob("**/*.flac"))

    logger.info("Enhancing %d bonafide files with Resemble Enhance...", len(bonafide_paths))
    enhanced_dir = out_dir / "bonafide"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in bonafide_paths:
        # Copy original
        import shutil
        dst_orig = enhanced_dir / wav_path.name
        shutil.copy2(str(wav_path), str(dst_orig))
        records.append({"file": f"bonafide/{wav_path.name}", "label": "bonafide",
                        "source": "original", "text": ""})

        # Enhanced version
        dst_enh = enhanced_dir / (wav_path.stem + "_enhanced.wav")
        if not dst_enh.exists():
            success = denoise_with_resemble_enhance(wav_path, dst_enh)
            if success:
                logger.info("  Enhanced: %s", dst_enh.name)
                records.append({"file": f"bonafide/{dst_enh.name}", "label": "bonafide",
                                "source": "resemble_enhance", "text": ""})
        else:
            records.append({"file": f"bonafide/{dst_enh.name}", "label": "bonafide",
                            "source": "resemble_enhance", "text": ""})

    return records


def generate_multi_tts_synthetic(out_dir: Path, n_per_engine: int = 8) -> list:
    """
    Generate synthetic samples from multiple TTS engines.
    Returns list of record dicts.
    """
    records = []
    spoof_dir = out_dir / "spoof"
    spoof_dir.mkdir(parents=True, exist_ok=True)

    sentences = SYNTH_SENTENCES[:n_per_engine]

    # Edge TTS voices (neural HiFi-GAN quality — hardest to detect)
    edge_voices = [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-RyanNeural",
        "en-AU-WilliamNeural",
    ]

    generators = [
        ("pyttsx3",    lambda text, i: generate_pyttsx3(
            text, spoof_dir / f"pyttsx3_{i:03d}.wav", rate=140 + i * 5)),
        ("gtts",       lambda text, i: generate_gtts(
            text, spoof_dir / f"gtts_{i:03d}.wav")),
        ("edge_tts_m", lambda text, i: generate_edge_tts(
            text, spoof_dir / f"edge_male_{i:03d}.wav", voice=edge_voices[0])),
        ("edge_tts_f", lambda text, i: generate_edge_tts(
            text, spoof_dir / f"edge_female_{i:03d}.wav", voice=edge_voices[1])),
        ("edge_tts_gb", lambda text, i: generate_edge_tts(
            text, spoof_dir / f"edge_gb_{i:03d}.wav", voice=edge_voices[2])),
        ("coqui_vits", lambda text, i: generate_coqui_tts(
            text, spoof_dir / f"coqui_{i:03d}.wav")),
    ]

    for gen_name, gen_fn in generators:
        logger.info("Generating synthetic clips with %s...", gen_name)
        for i, text in enumerate(sentences):
            fname = list(spoof_dir.glob(f"{gen_name.split('_')[0]}*_{i:03d}.wav"))
            if fname:
                # Already generated
                records.append({"file": f"spoof/{fname[0].name}", "label": "spoof",
                                "source": gen_name, "text": text})
                continue
            success = gen_fn(text, i)
            out_file = spoof_dir / f"{gen_name}_{i:03d}.wav"
            if success or out_file.exists():
                records.append({"file": f"spoof/{out_file.name}", "label": "spoof",
                                "source": gen_name, "text": text})
                logger.info("  [OK] %s: %s", gen_name, out_file.name)

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare enhanced training data.")
    parser.add_argument("--in-dir",       default="data/samples",          help="Input data directory.")
    parser.add_argument("--out-dir",      default="data/enhanced_samples", help="Output data directory.")
    parser.add_argument("--enhance",      action="store_true", default=True, help="Run Resemble Enhance on bonafide.")
    parser.add_argument("--generate-synth", action="store_true",            help="Generate multi-TTS synthetic clips.")
    parser.add_argument("--n-per-engine", type=int, default=8,             help="Sentences per TTS engine.")
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    if args.enhance:
        logger.info("=== Step 1: Resemble Enhance bonafide augmentation ===")
        records = prepare_enhanced_bonafide(in_dir, out_dir)
        all_records.extend(records)
        logger.info("Enhanced bonafide: %d records", len(records))

    # Copy existing spoof samples
    spoof_src = in_dir / "spoof"
    if spoof_src.exists():
        import shutil
        spoof_dst = out_dir / "spoof"
        spoof_dst.mkdir(parents=True, exist_ok=True)
        for f in spoof_src.glob("**/*.wav"):
            dst = spoof_dst / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            all_records.append({"file": f"spoof/{f.name}", "label": "spoof",
                                "source": "original", "text": ""})

    if args.generate_synth:
        logger.info("=== Step 2: Multi-TTS synthetic data generation ===")
        records = generate_multi_tts_synthetic(out_dir, n_per_engine=args.n_per_engine)
        all_records.extend(records)
        logger.info("Multi-TTS synthetic: %d records", len(records))

    # Write metadata CSV
    if all_records:
        meta_csv = out_dir / "samples_metadata.csv"
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "text"])
            writer.writeheader()
            writer.writerows(all_records)

        n_bf = sum(1 for r in all_records if r["label"] == "bonafide")
        n_sp = sum(1 for r in all_records if r["label"] == "spoof")
        logger.info("Saved metadata: %s (%d bonafide, %d spoof)", meta_csv, n_bf, n_sp)

    print(f"\n[OK] Dataset prepared at: {out_dir}")
    print(f"     Bonafide: {sum(1 for r in all_records if r['label']=='bonafide')}")
    print(f"     Spoof:    {sum(1 for r in all_records if r['label']=='spoof')}")
    print(f"\nNext step: python training/train_cm.py --data {out_dir}")


if __name__ == "__main__":
    main()
