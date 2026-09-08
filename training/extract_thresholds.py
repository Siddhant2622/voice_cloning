"""
training/extract_thresholds.py
Extract calibrated acoustic feature statistics and optimal detection thresholds
from the trained CM model and dataset for the browser Edge Engine.
"""

import csv
import json
import logging
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_thresholds")

def compute_acoustic_features(samples: np.ndarray, sample_rate: int = 16000) -> dict:
    """Replicates browser Edge Engine feature extraction in pure Python."""
    samples = samples.astype(np.float32)
    length = len(samples)
    if length == 0:
        return {}

    # 1. RMS & Peak
    rms = float(np.sqrt(np.mean(samples ** 2)))
    max_amp = float(np.max(np.abs(samples)))

    # 2. Zero-crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

    # 3. High-Frequency Power Ratio
    diff = np.diff(samples)
    hf_power = float(np.sum(diff ** 2))
    tot_power = float(np.sum(samples ** 2))
    hf_ratio = float(hf_power / (tot_power * 4)) if tot_power > 0 else 0.0

    # 4. Spectral Centroid & Flatness via FFT
    fft_n = 512
    n_frames = max(1, length // fft_n)
    flatness_list, centroid_list = [], []

    for f in range(min(n_frames, 20)):
        chunk = samples[f * fft_n : (f + 1) * fft_n]
        if len(chunk) < fft_n:
            break
        spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) + 1e-10
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
        
        # Centroid
        cent = np.sum(freqs * spec) / np.sum(spec)
        centroid_list.append(cent / (sample_rate / 2.0))

        # Flatness (geometric mean / arithmetic mean)
        geom = np.exp(np.mean(np.log(spec)))
        arith = np.mean(spec)
        flatness_list.append(geom / arith if arith > 0 else 0.0)

    spectral_centroid = float(np.mean(centroid_list)) if centroid_list else 0.0
    spectral_flatness = float(np.mean(flatness_list)) if flatness_list else 0.0

    # 5. Pitch Periodicity (autocorrelation in 80-400Hz range)
    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 80)
    ac_len = min(length, 4000)
    ac_chunk = samples[:ac_len]
    r0 = np.sum(ac_chunk ** 2)
    best_corr = 0.0
    if r0 > 0:
        for lag in range(min_lag, min(max_lag, ac_len - 1)):
            corr = np.sum(ac_chunk[:ac_len - lag] * ac_chunk[lag:ac_len]) / r0
            if corr > best_corr:
                best_corr = corr
    pitch_periodicity = float(np.clip(best_corr, 0.0, 1.0))

    # 6. Energy Variance across 25ms frames
    frame_len = int(sample_rate * 0.025)
    hop = int(sample_rate * 0.010)
    energies = []
    for i in range(0, length - frame_len, hop):
        e = np.sqrt(np.mean(samples[i : i + frame_len] ** 2))
        energies.append(e)
    e_mean = np.mean(energies) if energies else 0.0
    e_std = np.std(energies) if energies else 0.0
    energy_variance = float(e_std / e_mean) if e_mean > 0 else 0.0

    # 7. Harmonic-to-Noise Ratio (HNR proxy)
    hnr_proxy = float(pitch_periodicity / (spectral_flatness + 1e-4))

    return {
        "rms": round(rms, 4),
        "zcr": round(zcr, 4),
        "hf_ratio": round(hf_ratio, 4),
        "spectral_centroid": round(spectral_centroid, 4),
        "spectral_flatness": round(spectral_flatness, 4),
        "pitch_periodicity": round(pitch_periodicity, 4),
        "energy_variance": round(energy_variance, 4),
        "hnr_proxy": round(hnr_proxy, 4),
    }


def main():
    samples_dir = Path("data/samples")
    meta_csv = samples_dir / "samples_metadata.csv"
    if not meta_csv.exists():
        logger.error("Metadata CSV not found: %s", meta_csv)
        return

    # Load CM model if available
    cm_model_path = Path("models/cm.pt")
    cm_scorer = None
    if cm_model_path.exists():
        try:
            from src.models.cm_classifier import CMScorerWrapper
            cm_scorer = CMScorerWrapper(checkpoint_path=cm_model_path)
            logger.info("Loaded trained CM model from %s", cm_model_path)
        except Exception as e:
            logger.warning("Could not load CM model: %s", e)

    from src.preprocessing import preprocess_file
    from src.features import extract_all

    dataset_stats = []
    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = samples_dir / row["file"]
            if not audio_path.exists():
                continue

            try:
                data, sr = sf.read(str(audio_path))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                # Resample if needed
                if sr != 16000:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                    sr = 16000

                ac_feats = compute_acoustic_features(data, sr)

                cm_score = None
                if cm_scorer:
                    try:
                        wf, _ = preprocess_file(audio_path)
                        feats = extract_all(wf)
                        cm_score = round(cm_scorer.score(feats), 4)
                    except Exception as e:
                        logger.warning("CM scoring failed for %s: %s", row["file"], e)

                dataset_stats.append({
                    "file": row["file"],
                    "label": row["label"],
                    "source": row["source"],
                    "cm_score": cm_score,
                    **ac_feats
                })
            except Exception as e:
                logger.warning("Error analyzing %s: %s", row["file"], e)

    bonafide = [s for s in dataset_stats if s["label"] == "bonafide"]
    spoof = [s for s in dataset_stats if s["label"] == "spoof"]
    neural_tts = [s for s in dataset_stats if "neural" in s.get("source", "")]
    synthetic = [s for s in dataset_stats if "synthetic" in s.get("source", "") or "pyttsx3" in s.get("source", "")]

    logger.info("Dataset breakdown: %d bonafide, %d total spoof (%d neural TTS, %d synthetic)",
                len(bonafide), len(spoof), len(neural_tts), len(synthetic))

    def stat_summary(group, key):
        vals = [g[key] for g in group if g.get(key) is not None]
        if not vals:
            return {}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    keys = ["spectral_flatness", "pitch_periodicity", "energy_variance", "hf_ratio", "spectral_centroid", "zcr", "hnr_proxy"]
    if any(s.get("cm_score") is not None for s in dataset_stats):
        keys.append("cm_score")

    profile_stats = {
        "bonafide": {k: stat_summary(bonafide, k) for k in keys},
        "spoof": {k: stat_summary(spoof, k) for k in keys},
        "neural_tts": {k: stat_summary(neural_tts, k) for k in keys},
        "synthetic": {k: stat_summary(synthetic, k) for k in keys},
    }

    calibrated_thresholds = {
        "bonafide_profile": {
            "spectral_flatness_max": round(profile_stats["bonafide"]["spectral_flatness"].get("max", 0.35) * 1.15, 3),
            "energy_variance_min": round(profile_stats["bonafide"]["energy_variance"].get("min", 0.22) * 0.85, 3),
            "pitch_periodicity_min": round(profile_stats["bonafide"]["pitch_periodicity"].get("min", 0.45) * 0.85, 3),
            "hf_ratio_max": round(profile_stats["bonafide"]["hf_ratio"].get("max", 0.30) * 1.20, 3),
        },
        "spoof_profile": {
            "flatness_threshold": round(
                (profile_stats["bonafide"]["spectral_flatness"].get("mean", 0.2) +
                 profile_stats["spoof"]["spectral_flatness"].get("mean", 0.45)) / 2.0, 3
            ),
            "energy_var_threshold": round(
                (profile_stats["bonafide"]["energy_variance"].get("mean", 0.35) +
                 profile_stats["spoof"]["energy_variance"].get("mean", 0.15)) / 2.0, 3
            ),
            "periodicity_threshold": round(
                (profile_stats["bonafide"]["pitch_periodicity"].get("mean", 0.70) +
                 profile_stats["spoof"]["pitch_periodicity"].get("mean", 0.88)) / 2.0, 3
            ),
        },
        "profiles": profile_stats,
        "sample_evaluations": dataset_stats
    }

    out_file = Path("models/edge_thresholds.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibrated_thresholds, f, indent=2)

    logger.info("Saved edge thresholds to %s", out_file)
    print("\n" + "="*70)
    print("CALIBRATED ACOUSTIC FEATURE PROFILES:")
    print("="*70)
    for group_name in ["bonafide", "spoof", "neural_tts", "synthetic"]:
        print(f"\n--- {group_name.upper()} (n={len(eval(group_name))}) ---")
        for k in ["spectral_flatness", "energy_variance", "pitch_periodicity", "hf_ratio", "cm_score"]:
            if k in profile_stats[group_name] and profile_stats[group_name][k]:
                s = profile_stats[group_name][k]
                print(f"  {k:20s}: mean={s['mean']:.3f} +/- {s['std']:.3f}  [range: {s['min']:.3f} - {s['max']:.3f}]")
    print("="*70)

if __name__ == "__main__":
    main()
