"""
data/expand_dataset.py
Expands the training and benchmark dataset with:
  1. Real human speech clips (LibriSpeech public domain from HuggingFace)
  2. State-of-the-art Neural TTS clips (Edge-TTS neural vocoders: HiFi-GAN/VITS,
     similar to Gemini Live, ElevenLabs, OpenAI Voice)
"""

import asyncio
import csv
import os
from pathlib import Path
import urllib.request
import soundfile as sf
import numpy as np

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = SAMPLES_DIR / "samples_metadata.csv"

# 1. Real human speech clips (LibriSpeech public domain)
HUMAN_URLS = [
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/2.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/3.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/4.flac",
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/5.flac",
]

# 2. Modern Neural TTS voices (multilingual, conversational)
NEURAL_VOICES = [
    ("en-US-JennyNeural", "Hello, I am verifying my account details for the transaction."),
    ("en-US-GuyNeural", "Please transfer seventy thousand dollars to the authorized account immediately."),
    ("en-US-AriaNeural", "I am calling to update the wire transfer beneficiary information on file."),
    ("en-GB-SoniaNeural", "Could you confirm whether the security clearance has been granted for this session?"),
    ("en-GB-RyanNeural", "My authentication code has expired, please issue an override for this request."),
    ("en-IN-NeerjaNeural", "Good afternoon, I need urgent assistance with approving this international payment."),
    ("en-IN-PrabhatNeural", "The executive team approved the wire transfer earlier today, please proceed."),
    ("en-AU-NatashaNeural", "We have verified the customer credentials and need to finalize the transfer."),
]


async def generate_neural_clips():
    import edge_tts
    records = []
    for i, (voice, text) in enumerate(NEURAL_VOICES):
        filename = f"neural_tts_{i:02d}.wav"
        filepath = SAMPLES_DIR / filename
        mp3_path = SAMPLES_DIR / f"temp_{i}.mp3"

        if not filepath.exists():
            print(f"  Synthesizing Neural TTS ({voice}): {filename}...")
            try:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(mp3_path))
                # Convert mp3 to 16kHz wav via soundfile/librosa
                import librosa
                audio, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
                sf.write(str(filepath), audio, 16000)
                if mp3_path.exists():
                    os.remove(mp3_path)
                print(f"  [OK] Saved {filename}")
            except Exception as e:
                print(f"  [WARN] Failed {voice}: {e}")
                continue
        else:
            print(f"  {filename} already exists.")

        records.append({
            "file": filename,
            "label": "spoof",
            "source": f"neural_tts_{voice}",
            "text": text,
        })
    return records


def download_human_clips():
    records = []
    for i, url in enumerate(HUMAN_URLS):
        filename = f"human_librispeech_{i:02d}.wav"
        filepath = SAMPLES_DIR / filename
        flac_temp = SAMPLES_DIR / f"temp_human_{i}.flac"

        if not filepath.exists():
            print(f"  Downloading real human speech clip {i+1}/{len(HUMAN_URLS)}...")
            try:
                urllib.request.urlretrieve(url, str(flac_temp))
                import librosa
                audio, sr = librosa.load(str(flac_temp), sr=16000, mono=True)
                sf.write(str(filepath), audio, 16000)
                if flac_temp.exists():
                    os.remove(flac_temp)
                print(f"  [OK] Saved {filename}")
            except Exception as e:
                print(f"  [WARN] Failed download {url}: {e}")
                continue
        else:
            print(f"  {filename} already exists.")

        records.append({
            "file": filename,
            "label": "bonafide",
            "source": "librispeech_human",
            "text": "human voice recording",
        })
    return records


def main():
    print("=" * 60)
    print("Expanding dataset with real human speech & neural voices...")
    print("=" * 60)

    # 1. Real human speech
    human_records = download_human_clips()

    # 2. Modern Neural TTS
    neural_records = asyncio.run(generate_neural_clips())

    # 3. Read existing records
    existing = []
    if METADATA_CSV.exists():
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    existing_files = {r["file"] for r in existing}
    new_records = [r for r in (human_records + neural_records) if r["file"] not in existing_files]
    all_records = existing + new_records

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "text"])
        writer.writeheader()
        writer.writerows(all_records)

    n_sp = sum(1 for r in all_records if r["label"] == "spoof")
    n_bf = sum(1 for r in all_records if r["label"] == "bonafide")
    print(f"\n[OK] Dataset expanded: {len(all_records)} total clips ({n_bf} bona fide, {n_sp} synthetic/spoof)")


if __name__ == "__main__":
    main()
