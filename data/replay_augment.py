"""
data/replay_augment.py
======================
Presentation Gap Replay-Augmentation Engine.

Simulates and executes physical loudspeaker playback and microphone re-recording
for synthetic speech samples to close the acoustic presentation gap in voice clone
anti-spoofing classifiers.

Modes:
  --mode physical : Plays audio through physical speakers and captures via mic in real-time.
  --mode sim      : High-fidelity room impulse response (RIR) convolution, loudspeaker EQ &
                    non-linear distortion, mic frequency coloration, and room ambient noise.
  --mode hybrid   : Combines physical hardware re-recording with simulated multi-room acoustics.

Usage:
  python data/replay_augment.py --mode hybrid --limit 162
  python data/replay_augment.py --mode sim --in-dir data/large_dataset/spoof --out-dir data/large_dataset/replayed_spoof
"""

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import scipy.signal as signal
import soundfile as sf

TARGET_SR = 16000


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [replay_augment] {msg}", flush=True)


def to_mono_16k_fast(data: np.ndarray, orig_sr: int) -> np.ndarray:
    """Fast pure NumPy/SciPy 16kHz mono converter."""
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)

    if orig_sr != TARGET_SR:
        gcd = math.gcd(orig_sr, TARGET_SR)
        up = TARGET_SR // gcd
        down = orig_sr // gcd
        data = signal.resample_poly(data, up, down).astype(np.float32)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic Simulation Engine (RIR, Loudspeaker EQ, Mic Coloration, Noise)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_rir(sr: int = 16000, rt60: float = 0.35, room_dim: Tuple[float, float, float] = (4.0, 5.0, 2.7)) -> np.ndarray:
    """
    Generates a realistic Room Impulse Response (RIR) combining direct path,
    early specular reflections (image source approximation), and late diffuse reverberation.
    """
    num_samples = int(rt60 * sr)
    t = np.linspace(0, rt60, num_samples, endpoint=False)

    # 1. Direct path
    rir = np.zeros(num_samples, dtype=np.float32)
    direct_idx = int(0.003 * sr)  # ~1 meter distance (3ms delay)
    rir[direct_idx] = 1.0

    # 2. Early reflections (boundary walls)
    c = 343.0  # speed of sound m/s
    boundaries = [
        2 * room_dim[0] / c,
        2 * room_dim[1] / c,
        2 * room_dim[2] / c,
        math.sqrt(room_dim[0]**2 + room_dim[1]**2) / c,
        math.sqrt(room_dim[1]**2 + room_dim[2]**2) / c,
        math.sqrt(room_dim[0]**2 + room_dim[2]**2) / c,
    ]

    for delay_s in boundaries:
        idx = int(delay_s * sr)
        if idx < num_samples:
            attenuation = 0.35 * math.exp(-6.91 * delay_s / max(0.01, rt60))
            phase = random.choice([1.0, -1.0])
            rir[idx] += phase * attenuation

    # 3. Late diffuse reverberation (exponentially decaying Gaussian noise with air absorption)
    decay = np.exp(-6.91 * t / max(0.01, rt60))
    late_noise = np.random.normal(0, 1, num_samples).astype(np.float32)

    alpha = 0.85
    filtered_noise = np.zeros_like(late_noise)
    acc = 0.0
    for i in range(num_samples):
        acc = alpha * acc + (1.0 - alpha) * late_noise[i]
        filtered_noise[i] = acc

    rir += 0.25 * decay * filtered_noise

    norm = np.sqrt(np.sum(rir**2) + 1e-9)
    return (rir / norm).astype(np.float32)


def apply_speaker_eq_and_saturation(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulates physical loudspeaker characteristics:
    - High-pass bass rolloff (typical of laptop / mobile speakers < 140Hz)
    - Resonant presence peak around 2.5 kHz - 4 kHz
    - Gentle non-linear harmonic saturation (soft tanh clipping)
    """
    sos_hp = signal.butter(2, 140, 'hp', fs=sr, output='sos')
    out = signal.sosfilt(sos_hp, wav)

    sos_peak = signal.iirpeak(3200, 1.5, fs=sr)
    b, a = sos_peak
    out = signal.lfilter(b, a, out) * 1.3 + out * 0.7

    drive = 1.25
    out = np.tanh(drive * out) / drive

    return out.astype(np.float32)


def apply_mic_coloration(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulates electret/MEMS microphone transfer characteristics:
    - Slight high-frequency rolloff above 7.5 kHz
    - Proximity effect low-mid resonance
    """
    sos_lp = signal.butter(2, 7500, 'lp', fs=sr, output='sos')
    out = signal.sosfilt(sos_lp, wav)

    sos_notch = signal.iirpeak(300, 1.0, fs=sr)
    b, a = sos_notch
    out = out + 0.15 * signal.lfilter(b, a, out)

    return out.astype(np.float32)


def add_room_ambient_noise(wav: np.ndarray, snr_db: float = 26.0) -> np.ndarray:
    """Adds realistic room ambient noise (pink/brown thermal & HVAC hiss)."""
    signal_power = np.mean(wav**2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    white = np.random.normal(0, 1, len(wav)).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = signal.lfilter(b, a, white)
    pink_norm = pink / (np.std(pink) + 1e-9)

    noise = np.sqrt(noise_power) * pink_norm
    return (wav + noise).astype(np.float32)


def simulate_replayed_audio(wav_16k: np.ndarray, sr: int = 16000, room_idx: int = 0) -> np.ndarray:
    """
    Complete acoustic channel simulation:
    Clean Speech -> Loudspeaker Model -> Room Impulse Response -> Mic Model -> Room Ambient Noise.
    """
    rt60_choices = [0.22, 0.32, 0.45, 0.52]
    snr_choices  = [24.0, 28.0, 32.0, 22.0]
    room_dims    = [(3.5, 4.0, 2.5), (4.5, 5.5, 2.8), (2.8, 3.2, 2.4), (5.0, 6.0, 3.0)]

    idx = room_idx % len(rt60_choices)
    rt60 = rt60_choices[idx]
    snr  = snr_choices[idx]
    rdim = room_dims[idx]

    spk_out = apply_speaker_eq_and_saturation(wav_16k, sr=sr)
    rir = generate_synthetic_rir(sr=sr, rt60=rt60, room_dim=rdim)
    reverb_out = np.convolve(spk_out, rir, mode='full')[:len(wav_16k)]
    mic_out = apply_mic_coloration(reverb_out, sr=sr)
    noisy_out = add_room_ambient_noise(mic_out, snr_db=snr)

    peak = np.max(np.abs(noisy_out)) + 1e-9
    norm_out = (noisy_out / peak) * 0.90
    return norm_out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Physical Hardware Playback & Capture (SoundDevice)
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_devices():
    """Find appropriate speaker and microphone indices."""
    import sounddevice as sd
    devices = sd.query_devices()

    input_dev = None
    output_dev = None

    for i, d in enumerate(devices):
        name = d.get('name', '').lower()
        if d.get('max_input_channels', 0) > 0 and input_dev is None:
            if 'microphone array' in name or 'smart sound' in name or 'realtek' in name:
                input_dev = i

    for i, d in enumerate(devices):
        name = d.get('name', '').lower()
        if d.get('max_output_channels', 0) > 0 and output_dev is None:
            if 'speaker' in name or 'realtek' in name:
                output_dev = i

    if input_dev is None:
        input_dev = sd.default.device[0] if sd.default.device[0] is not None else 1
    if output_dev is None:
        output_dev = sd.default.device[1] if sd.default.device[1] is not None else 3

    return input_dev, output_dev


def physical_replay_sample(wav_16k: np.ndarray, sr: int = 16000,
                           input_device: Optional[int] = None,
                           output_device: Optional[int] = None) -> Optional[np.ndarray]:
    """Plays audio through physical speakers and simultaneously records from microphone."""
    try:
        import sounddevice as sd

        if input_device is None or output_device is None:
            auto_in, auto_out = get_audio_devices()
            input_device = input_device or auto_in
            output_device = output_device or auto_out

        pad_pre = np.zeros(int(0.2 * sr), dtype=np.float32)
        pad_post = np.zeros(int(0.3 * sr), dtype=np.float32)
        play_data = np.concatenate([pad_pre, wav_16k, pad_post])

        recorded = sd.playrec(
            play_data,
            samplerate=sr,
            channels=1,
            dtype='float32',
            device=(input_device, output_device)
        )
        sd.wait()

        rec_flat = recorded.squeeze()
        search_window = rec_flat[int(0.10 * sr):int(0.50 * sr)]
        onset = np.argmax(np.abs(search_window)) + int(0.10 * sr)

        start_idx = max(0, onset - int(0.04 * sr))
        end_idx = start_idx + len(wav_16k)

        trimmed = rec_flat[start_idx:end_idx]
        if len(trimmed) < len(wav_16k):
            trimmed = np.pad(trimmed, (0, len(wav_16k) - len(trimmed)))

        peak = np.max(np.abs(trimmed)) + 1e-9
        trimmed = (trimmed / peak) * 0.90
        return trimmed.astype(np.float32)

    except Exception as e:
        log(f"Physical replay warning: {e} (using simulation)")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Processing & Metadata Management
# ─────────────────────────────────────────────────────────────────────────────

def process_replay_augmentation(
    in_dir: Path,
    out_dir: Path,
    meta_path: Path,
    mode: str = "hybrid",
    limit: Optional[int] = None,
    physical_count: int = 5
):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob("*.wav")) + list(in_dir.glob("*.mp3")) + list(in_dir.glob("*.flac")))
    if limit is not None and limit > 0:
        files = files[:limit]

    log(f"Found {len(files)} synthetic spoof samples in {in_dir} to augment.")
    log(f"Replay mode: {mode} | Output dir: {out_dir}")

    in_dev, out_dev = None, None
    if mode in ("physical", "hybrid"):
        try:
            in_dev, out_dev = get_audio_devices()
            log(f"Selected Audio Devices -> Mic: #{in_dev}, Speaker: #{out_dev}")
        except Exception as e:
            log(f"Could not query sound devices: {e}. Switching to simulation mode.")
            mode = "sim"

    new_metadata_rows = []
    t0 = time.time()

    for idx, fpath in enumerate(files):
        try:
            data, sr = sf.read(str(fpath))
            wav_16k = to_mono_16k_fast(data, sr)

            replayed_wav = None
            source_tag = "replayed_spoof_sim"

            if (mode == "physical") or (mode == "hybrid" and idx < physical_count):
                log(f"[{idx + 1}/{len(files)}] Physical speaker+mic recording: {fpath.name}...")
                replayed_wav = physical_replay_sample(wav_16k, sr=TARGET_SR, input_device=in_dev, output_device=out_dev)
                if replayed_wav is not None:
                    source_tag = "replayed_spoof_physical"

            if replayed_wav is None:
                replayed_wav = simulate_replayed_audio(wav_16k, sr=TARGET_SR, room_idx=idx)

            out_filename = f"replayed_{fpath.stem}.wav"
            out_file = out_dir / out_filename
            sf.write(str(out_file), replayed_wav, TARGET_SR)

            rel_file_path = f"replayed_spoof/{out_filename}"
            new_metadata_rows.append({
                "file": rel_file_path,
                "label": "spoof",
                "source": source_tag,
                "text": ""
            })

            if (idx + 1) % 15 == 0 or (idx + 1) == len(files):
                elapsed = time.time() - t0
                per_file = elapsed / (idx + 1)
                remaining = (len(files) - idx - 1) * per_file
                log(f"Replay progress: {idx + 1}/{len(files)} ({(idx + 1)/len(files)*100:.1f}%) — {per_file:.2f}s/sample, ~{remaining:.0f}s remaining")

        except Exception as err:
            log(f"Failed to augment {fpath.name}: {err}")

    # Update samples_metadata.csv
    if meta_path.exists():
        existing_rows = []
        with open(meta_path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            fieldnames = reader.fieldnames or ["file", "label", "source", "text"]
            for row in reader:
                if not row["file"].startswith("replayed_spoof/"):
                    existing_rows.append(row)

        all_rows = existing_rows + new_metadata_rows
        with open(meta_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        log(f"Updated {meta_path}: total samples now {len(all_rows)} (added {len(new_metadata_rows)} replayed_spoof samples).")
    else:
        with open(meta_path, "w", encoding="utf-8", newline="") as fp:
            fieldnames = ["file", "label", "source", "text"]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_metadata_rows)
        log(f"Created {meta_path} with {len(new_metadata_rows)} replayed_spoof samples.")

    print("\n" + "=" * 60, flush=True)
    print("REPLAY AUGMENTATION COMPLETE", flush=True)
    print(f"Generated samples: {len(new_metadata_rows)} in {out_dir}", flush=True)
    print(f"Metadata updated:  {meta_path}", flush=True)
    print("=" * 60 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Replay Augmentation Pipeline")
    parser.add_argument("--in-dir", default="data/large_dataset/spoof", help="Input synthetic audio directory")
    parser.add_argument("--out-dir", default="data/large_dataset/replayed_spoof", help="Output replayed audio directory")
    parser.add_argument("--meta", default="data/large_dataset/samples_metadata.csv", help="Dataset metadata CSV")
    parser.add_argument("--mode", choices=["physical", "sim", "hybrid"], default="hybrid",
                        help="Replay mode (physical, sim, hybrid)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to augment")
    parser.add_argument("--physical-count", type=int, default=5,
                        help="Number of samples to record physically when in hybrid mode (default: 5)")
    args = parser.parse_args()

    process_replay_augmentation(
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        meta_path=Path(args.meta),
        mode=args.mode,
        limit=args.limit,
        physical_count=args.physical_count
    )


if __name__ == "__main__":
    main()
