"""
data/generate_modern_spoof.py
==============================
Generate a diverse, modern synthetic speech dataset using edge-tts.

Uses 30+ different Microsoft Edge neural TTS voices across multiple
languages and accents to produce a variety of spoof training samples
that are more representative of modern TTS quality (2024-2026 era).

Also applies codec augmentation to create realistic degraded versions.

Usage:
    python data/generate_modern_spoof.py --out data/modern_dataset --count 500
    python data/generate_modern_spoof.py --out data/modern_dataset --count 300 --codec-aug
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import scipy.signal as signal
import soundfile as sf

logger = logging.getLogger("gen_modern_spoof")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Modern neural TTS voices (edge-tts, 2024-2026 quality)
# These cover diverse accents, genders, and languages
# ---------------------------------------------------------------------------
NEURAL_VOICES = [
    # US English
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AmberNeural", "en-US-AnaNeural",
    "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural",
    "en-US-JasonNeural", "en-US-SaraNeural", "en-US-TonyNeural",
    "en-US-NancyNeural", "en-US-JaneNeural", "en-US-RogerNeural",
    "en-US-SteffanNeural",
    # UK English
    "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-GB-AbbiNeural", "en-GB-AlfieNeural", "en-GB-BellaNeural",
    "en-GB-ElliotNeural", "en-GB-EthanNeural", "en-GB-HollieNeural",
    "en-GB-MaisieNeural", "en-GB-NoahNeural", "en-GB-OliverNeural",
    "en-GB-OliviaNeural", "en-GB-ThomasNeural",
    # Australian English
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    "en-AU-AnnetteNeural", "en-AU-CarlyNeural", "en-AU-DarrenNeural",
    # Indian English
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "en-IN-AashiNeural", "en-IN-AnanyaNeural",
    # Canadian English
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    # Irish English
    "en-IE-EmilyNeural", "en-IE-ConnorNeural",
    # Other languages (testing cross-lingual robustness)
    "de-DE-KatjaNeural", "de-DE-ConradNeural",
    "fr-FR-DeniseNeural", "fr-FR-HenriNeural",
    "es-ES-ElviraNeural", "es-ES-AlvaroNeural",
    "ja-JP-NanamiNeural", "ja-JP-KeitaNeural",
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
    "hi-IN-SwaraNeural", "hi-IN-MadhurNeural",
    "ko-KR-SunHiNeural", "ko-KR-InJoonNeural",
    "pt-BR-FranciscaNeural", "pt-BR-AntonioNeural",
    "it-IT-ElsaNeural", "it-IT-DiegoNeural",
    "ar-SA-ZariyahNeural", "ar-SA-HamedNeural",
]

# Diverse text corpus for TTS generation
TEXT_CORPUS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Every morning I walk through the park and listen to the birds singing.",
    "Technology is advancing at an unprecedented rate in the modern world.",
    "She sold seashells by the seashore on a warm summer afternoon.",
    "The old library on the corner has been standing for over a hundred years.",
    "Music has the power to bring people together across all cultures.",
    "The scientist discovered a new species of butterfly in the Amazon rainforest.",
    "Education is the most powerful weapon which you can use to change the world.",
    "The sunset painted the sky in brilliant shades of orange and purple.",
    "A good book is a garden carried in the pocket of every reader.",
    "The train arrived at the station exactly on time despite the heavy snow.",
    "Artificial intelligence is transforming how we interact with technology.",
    "The children played happily in the garden throughout the long summer day.",
    "Knowledge is the key that opens many doors in our lives.",
    "The ocean waves crashed against the rocky shore under the moonlit sky.",
    "Communication is the foundation of every successful relationship.",
    "The ancient castle stood majestically on top of the green hill.",
    "Innovation drives progress and shapes the future of humanity.",
    "The baker woke up early every morning to prepare fresh bread.",
    "Nature provides us with everything we need to survive and thrive.",
    "The concert hall was filled with the beautiful sound of the orchestra.",
    "Learning a new language opens up a world of new possibilities.",
    "The mountain trail offered breathtaking views of the valley below.",
    "Creativity is intelligence having fun with the world around us.",
    "The farmer harvested the golden wheat fields before the autumn rain.",
    "Science and art are two sides of the same coin of human expression.",
    "The city streets were bustling with people heading to work each morning.",
    "Patience is not the ability to wait but how you act while waiting.",
    "The lighthouse guided ships safely through the treacherous waters at night.",
    "History teaches us lessons that help shape a better tomorrow.",
    "Please call the number on your screen to verify your identity now.",
    "Your account has been temporarily locked for security purposes.",
    "The meeting is scheduled for three o'clock in the afternoon today.",
    "Can you please confirm your date of birth and postal code?",
    "The weather forecast predicts rain throughout the coming weekend.",
    "We are experiencing higher than normal call volumes at this time.",
    "Press one for billing inquiries or two for technical support.",
    "Your order has been shipped and will arrive within three business days.",
    "The annual report shows a significant increase in revenue this quarter.",
    "Please leave your message after the tone and we will get back to you.",
]


async def generate_single_sample(
    voice: str,
    text: str,
    output_path: Path,
) -> bool:
    """Generate a single TTS sample using edge-tts."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        return True
    except Exception as e:
        logger.warning("Failed to generate with %s: %s", voice, e)
        return False


def apply_codec_g711(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """G.711 mu-law codec simulation."""
    if sr != 8000:
        gcd = math.gcd(sr, 8000)
        wav_8k = signal.resample_poly(wav, 8000 // gcd, sr // gcd).astype(np.float32)
    else:
        wav_8k = wav.copy()
    mu = 255.0
    wav_norm = wav_8k / (np.abs(wav_8k).max() + 1e-8)
    compressed = np.sign(wav_norm) * np.log1p(mu * np.abs(wav_norm)) / np.log1p(mu)
    quantized = np.round(compressed * 128) / 128.0
    expanded = np.sign(quantized) * ((1 + mu) ** np.abs(quantized) - 1) / mu
    if sr != 8000:
        gcd = math.gcd(8000, sr)
        expanded = signal.resample_poly(expanded, sr // gcd, 8000 // gcd).astype(np.float32)
    return expanded[:len(wav)]


def apply_codec_lowpass(wav: np.ndarray, sr: int = 16000, cutoff: float = 4000) -> np.ndarray:
    """Low-pass filter simulating narrow-band codec."""
    nyq = sr / 2
    if cutoff >= nyq:
        return wav
    b, a = signal.butter(5, cutoff / nyq, btype='low')
    filtered = signal.filtfilt(b, a, wav).astype(np.float32)
    noise_level = 0.002
    filtered += np.random.randn(len(filtered)).astype(np.float32) * noise_level
    return filtered


def apply_replay_sim(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Lightweight replay simulation."""
    rt60 = random.uniform(0.15, 0.45)
    num_samples = int(rt60 * sr)
    t = np.linspace(0, rt60, num_samples, endpoint=False)
    rir = np.zeros(num_samples, dtype=np.float32)
    direct_idx = int(0.003 * sr)
    if direct_idx < num_samples:
        rir[direct_idx] = 1.0
    for delay_ms in [8, 12, 18, 25]:
        idx = int(delay_ms * sr / 1000)
        if idx < num_samples:
            rir[idx] = random.uniform(0.1, 0.3)
    decay = np.exp(-6.908 * t / rt60)
    rir += np.random.randn(num_samples).astype(np.float32) * 0.015 * decay
    rir /= np.abs(rir).max() + 1e-8
    out = signal.fftconvolve(wav, rir, mode="full")[:len(wav)].astype(np.float32)
    snr_db = random.uniform(22, 35)
    noise = np.random.randn(len(out)).astype(np.float32)
    sig_power = np.mean(out ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    out += noise * np.sqrt(noise_power)
    mx = np.abs(out).max()
    if mx > 0:
        out = out / mx * 0.95
    return out


async def generate_dataset(
    out_dir: Path,
    count: int = 500,
    codec_aug: bool = True,
    include_existing_spoof: bool = True,
):
    """Generate diverse modern spoof dataset."""
    import edge_tts
    
    spoof_dir = out_dir / "spoof"
    bonafide_dir = out_dir / "bonafide"
    spoof_dir.mkdir(parents=True, exist_ok=True)
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # 1. Copy existing bonafide
    existing_bf = Path("data/large_dataset/bonafide")
    if existing_bf.exists():
        bf_files = sorted(list(existing_bf.glob("*.flac")) + list(existing_bf.glob("*.wav")))
        for f in bf_files:
            dest = bonafide_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            records.append({"file": f"bonafide/{dest.name}", "label": "bonafide",
                            "source": "librispeech_dev_clean", "tts_system": "human"})
        logger.info("Copied %d existing bonafide samples", len(bf_files))
    
    # 2. Copy existing spoof (Edge-TTS v1 — keeping as baseline)
    existing_sp = Path("data/large_dataset/spoof")
    existing_sp_count = 0
    if include_existing_spoof and existing_sp.exists():
        sp_files = sorted(list(existing_sp.glob("*.wav")))
        for f in sp_files:
            dest = spoof_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            voice_name = f.stem.split("_")[1] if "_" in f.stem else "edge_tts_v1"
            records.append({"file": f"spoof/{dest.name}", "label": "spoof",
                            "source": "edge_tts_v1", "tts_system": voice_name})
            existing_sp_count += 1
        logger.info("Copied %d existing spoof samples", existing_sp_count)
    
    # 3. Generate new diverse samples
    remaining = count - existing_sp_count
    if remaining <= 0:
        logger.info("Already have %d spoof samples, no generation needed", existing_sp_count)
    else:
        logger.info("Generating %d new diverse TTS samples across %d voices...", remaining, len(NEURAL_VOICES))
        
        samples_per_voice = max(1, remaining // len(NEURAL_VOICES))
        gen_count = 0
        
        random.seed(42)
        
        for voice_idx, voice in enumerate(NEURAL_VOICES):
            if gen_count >= remaining:
                break
            
            voice_short = voice.replace("Neural", "").replace("-", "_").lower()
            
            for sample_idx in range(samples_per_voice):
                if gen_count >= remaining:
                    break
                
                text = random.choice(TEXT_CORPUS)
                fname = f"modern_{voice_short}_{sample_idx:03d}.wav"
                fpath = spoof_dir / fname
                
                if fpath.exists():
                    records.append({"file": f"spoof/{fname}", "label": "spoof",
                                    "source": f"edge_tts_{voice_short}", "tts_system": voice})
                    gen_count += 1
                    continue
                
                # Generate with edge-tts (outputs MP3, convert to WAV)
                mp3_path = fpath.with_suffix(".mp3")
                success = await generate_single_sample(voice, text, mp3_path)
                
                if success and mp3_path.exists():
                    try:
                        # Convert MP3 to WAV 16kHz mono
                        data, sr = sf.read(str(mp3_path))
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                        if sr != 16000:
                            gcd = math.gcd(sr, 16000)
                            data = signal.resample_poly(data, 16000 // gcd, sr // gcd).astype(np.float32)
                        sf.write(str(fpath), data, 16000)
                        mp3_path.unlink(missing_ok=True)
                        
                        records.append({"file": f"spoof/{fname}", "label": "spoof",
                                        "source": f"edge_tts_{voice_short}", "tts_system": voice})
                        gen_count += 1
                        
                    except Exception as e:
                        logger.warning("Conversion failed for %s: %s", mp3_path, e)
                        mp3_path.unlink(missing_ok=True)
                else:
                    mp3_path.unlink(missing_ok=True)
            
            if (voice_idx + 1) % 10 == 0:
                logger.info("Voices processed: %d/%d — samples generated: %d/%d",
                           voice_idx + 1, len(NEURAL_VOICES), gen_count, remaining)
        
        logger.info("Generated %d new spoof samples", gen_count)
    
    # 4. Generate codec-augmented versions
    if codec_aug:
        logger.info("Generating codec-augmented versions...")
        spoof_files = sorted(list(spoof_dir.glob("*.wav")))
        aug_count = 0
        
        for f in spoof_files[:count // 3]:  # Augment 1/3 of spoof samples
            try:
                data, sr = sf.read(str(f))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                data = data.astype(np.float32)
                
                # G.711 version
                g711_data = apply_codec_g711(data, sr)
                g711_name = f"codec_g711_{f.stem}.wav"
                sf.write(str(spoof_dir / g711_name), g711_data, sr)
                records.append({"file": f"spoof/{g711_name}", "label": "spoof",
                                "source": "codec_g711", "tts_system": "codec_g711"})
                
                # Low-pass (VoIP) version
                lp_data = apply_codec_lowpass(data, sr, cutoff=4000)
                lp_name = f"codec_voip_{f.stem}.wav"
                sf.write(str(spoof_dir / lp_name), lp_data, sr)
                records.append({"file": f"spoof/{lp_name}", "label": "spoof",
                                "source": "codec_voip", "tts_system": "codec_voip"})
                
                aug_count += 2
            except Exception as e:
                logger.warning("Augmentation failed for %s: %s", f.name, e)
        
        logger.info("Generated %d codec-augmented samples", aug_count)
    
    # 5. Generate replay-augmented versions of some bonafide samples
    # (so the model learns that bonafide+replay != spoof)
    bf_files_for_replay = sorted(list(bonafide_dir.glob("*.flac")) + list(bonafide_dir.glob("*.wav")))
    replay_bf_count = 0
    for f in bf_files_for_replay[:min(50, len(bf_files_for_replay))]:
        try:
            data, sr = sf.read(str(f))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)
            replayed = apply_replay_sim(data, sr)
            replay_name = f"replay_bf_{f.stem}.wav"
            sf.write(str(bonafide_dir / replay_name), replayed, sr)
            records.append({"file": f"bonafide/{replay_name}", "label": "bonafide",
                            "source": "replay_bonafide", "tts_system": "human"})
            replay_bf_count += 1
        except Exception:
            pass
    logger.info("Generated %d replay-augmented bonafide samples", replay_bf_count)
    
    # 6. Write metadata
    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "tts_system"])
        writer.writeheader()
        writer.writerows(records)
    
    bf_count = sum(1 for r in records if r["label"] == "bonafide")
    sp_count = sum(1 for r in records if r["label"] == "spoof")
    sources = set(r["source"] for r in records)
    tts_systems = set(r["tts_system"] for r in records if r["label"] == "spoof")
    
    print("\n" + "=" * 60)
    print("MODERN DATASET READY")
    print("=" * 60)
    print(f"  Output:       {out_dir}")
    print(f"  Bonafide:     {bf_count} samples")
    print(f"  Spoof:        {sp_count} samples")
    print(f"  Sources:      {len(sources)} ({', '.join(sorted(sources)[:10])}...)")
    print(f"  TTS voices:   {len(tts_systems)}")
    print(f"  Metadata:     {meta_path}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate modern diverse spoof dataset")
    parser.add_argument("--out", default="data/modern_dataset", help="Output directory")
    parser.add_argument("--count", type=int, default=500, help="Target spoof sample count")
    parser.add_argument("--codec-aug", action="store_true", default=True, help="Add codec augmentation")
    parser.add_argument("--no-codec-aug", action="store_false", dest="codec_aug")
    args = parser.parse_args()
    
    asyncio.run(generate_dataset(
        out_dir=Path(args.out),
        count=args.count,
        codec_aug=args.codec_aug,
    ))


if __name__ == "__main__":
    main()
