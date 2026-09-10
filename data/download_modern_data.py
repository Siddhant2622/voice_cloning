"""
data/download_modern_data.py
==============================
Downloads and organizes modern audio deepfake datasets into a unified
training directory structure for VoiceGuard CM classifier retraining.

Datasets:
  - MLAAD v3       (deepfake-total.com — 23 TTS systems, 38 languages)
  - WaveFake        (Zenodo — 6 vocoders × LJSpeech)
  - In-the-Wild     (Fraunhofer AISEC — real-world AI audio)

Usage:
    python data/download_modern_data.py --out data/modern_dataset
    python data/download_modern_data.py --out data/modern_dataset --max-per-class 500
    python data/download_modern_data.py --out data/modern_dataset --skip-download
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import random
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
import ssl

# Bypass SSL certificate verification for dataset servers with self-signed certs
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger("download_modern")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------
DATASETS = {
    "mlaad": {
        "url": "https://deepfake-total.com/mlaad/mlaad_v3.zip",
        "out_dir": "mlaad",
        "size_gb": 3.0,
        "desc": "MLAAD v3 — 23 TTS systems, 38 languages",
    },
    "wavefake": {
        "url": "https://zenodo.org/record/5642694/files/WaveFake.zip",
        "out_dir": "wavefake",
        "size_gb": 19.0,
        "desc": "WaveFake — 6 vocoders x LJSpeech",
    },
    "in-the-wild": {
        "url": "https://deepfake-demo.aisec.fraunhofer.de/in_the_wild.zip",
        "out_dir": "in_the_wild",
        "size_gb": 2.0,
        "desc": "In-the-Wild — real-world AI audio from the internet",
    },
}

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()


def download_file(url: str, dest: Path) -> bool:
    logger.info("Downloading: %s", url)
    logger.info("  -> %s", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, str(dest), reporthook=_progress)
        logger.info("Download complete: %.1f MB", dest.stat().st_size / 1e6)
        return True
    except (URLError, OSError) as e:
        logger.error("Download failed: %s", e)
        return False


def extract_archive(archive: Path, out_dir: Path) -> None:
    logger.info("Extracting %s -> %s", archive.name, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    elif archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    elif archive.name.endswith(".tar.bz2"):
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive.suffix}")
    logger.info("Extraction complete.")


def find_audio_files(directory: Path) -> list[Path]:
    files = []
    for ext in AUDIO_EXTS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def organize_mlaad(raw_dir: Path, bonafide_dir: Path, spoof_dir: Path, max_per_class: int) -> list[dict]:
    """MLAAD contains only synthetic (spoof) audio organized by TTS system."""
    records = []
    audio_files = find_audio_files(raw_dir)
    
    if not audio_files:
        logger.warning("No audio files found in MLAAD directory: %s", raw_dir)
        return records
    
    random.shuffle(audio_files)
    count = 0
    
    for f in audio_files:
        if count >= max_per_class:
            break
        rel = f.relative_to(raw_dir)
        tts_system = str(rel.parts[0]) if len(rel.parts) > 1 else "unknown"
        dest = spoof_dir / f"mlaad_{tts_system}_{count:04d}{f.suffix}"
        try:
            shutil.copy2(f, dest)
            records.append({
                "file": f"spoof/{dest.name}",
                "label": "spoof",
                "source": f"mlaad_{tts_system}",
                "tts_system": tts_system,
            })
            count += 1
        except Exception as e:
            logger.warning("Failed to copy %s: %s", f, e)
    
    logger.info("MLAAD: organized %d spoof samples", count)
    return records


def organize_wavefake(raw_dir: Path, bonafide_dir: Path, spoof_dir: Path, max_per_class: int) -> list[dict]:
    """WaveFake: LJSpeech originals (bonafide) + 6 vocoder versions (spoof)."""
    records = []
    
    vocoder_dirs = []
    for d in sorted(raw_dir.rglob("*")):
        if d.is_dir() and d.name.lower() not in ("ljspeech", "original", "__macosx"):
            if any(d.rglob("*.wav")):
                vocoder_dirs.append(d)
    
    bf_count = 0
    for d in sorted(raw_dir.rglob("*")):
        if d.is_dir() and d.name.lower() in ("ljspeech", "original"):
            bf_files = find_audio_files(d)
            random.shuffle(bf_files)
            for f in bf_files[:max_per_class // 4]:
                dest = bonafide_dir / f"wavefake_orig_{bf_count:04d}{f.suffix}"
                try:
                    shutil.copy2(f, dest)
                    records.append({
                        "file": f"bonafide/{dest.name}",
                        "label": "bonafide",
                        "source": "wavefake_ljspeech",
                        "tts_system": "human",
                    })
                    bf_count += 1
                except Exception:
                    pass
    
    sp_count = 0
    samples_per_vocoder = max(1, max_per_class // max(1, len(vocoder_dirs)))
    
    for vdir in vocoder_dirs:
        vocoder_name = vdir.name.lower().replace(" ", "_")
        files = find_audio_files(vdir)
        random.shuffle(files)
        for f in files[:samples_per_vocoder]:
            if sp_count >= max_per_class:
                break
            dest = spoof_dir / f"wavefake_{vocoder_name}_{sp_count:04d}{f.suffix}"
            try:
                shutil.copy2(f, dest)
                records.append({
                    "file": f"spoof/{dest.name}",
                    "label": "spoof",
                    "source": f"wavefake_{vocoder_name}",
                    "tts_system": vocoder_name,
                })
                sp_count += 1
            except Exception:
                pass
    
    logger.info("WaveFake: organized %d bonafide + %d spoof samples", bf_count, sp_count)
    return records


def organize_in_the_wild(raw_dir: Path, bonafide_dir: Path, spoof_dir: Path, max_per_class: int) -> list[dict]:
    """In-the-Wild: real vs. fake with metadata CSV or directory structure."""
    records = []
    bf_count = 0
    sp_count = 0
    
    meta_files = list(raw_dir.rglob("meta.csv")) + list(raw_dir.rglob("metadata.csv"))
    
    if meta_files:
        meta_path = meta_files[0]
        with open(meta_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        random.shuffle(rows)
        for row in rows:
            file_col = next((c for c in row if c.lower() in ("file", "path", "filename")), None)
            label_col = next((c for c in row if c.lower() in ("label", "class", "type")), None)
            if not file_col or not label_col:
                continue
            fpath = meta_path.parent / row[file_col]
            if not fpath.exists():
                continue
            label = row[label_col].lower().strip()
            is_spoof = label in ("fake", "spoof", "synthetic", "1")
            if is_spoof and sp_count < max_per_class:
                dest = spoof_dir / f"itw_spoof_{sp_count:04d}{fpath.suffix}"
                shutil.copy2(fpath, dest)
                records.append({"file": f"spoof/{dest.name}", "label": "spoof",
                                "source": "in_the_wild", "tts_system": "unknown_itw"})
                sp_count += 1
            elif not is_spoof and bf_count < max_per_class:
                dest = bonafide_dir / f"itw_real_{bf_count:04d}{fpath.suffix}"
                shutil.copy2(fpath, dest)
                records.append({"file": f"bonafide/{dest.name}", "label": "bonafide",
                                "source": "in_the_wild", "tts_system": "human"})
                bf_count += 1
    else:
        for label_dir_name in ("real", "bonafide", "genuine", "human"):
            ld = raw_dir / label_dir_name
            if ld.exists():
                files = find_audio_files(ld)
                random.shuffle(files)
                for f in files[:max_per_class]:
                    if bf_count >= max_per_class:
                        break
                    dest = bonafide_dir / f"itw_real_{bf_count:04d}{f.suffix}"
                    shutil.copy2(f, dest)
                    records.append({"file": f"bonafide/{dest.name}", "label": "bonafide",
                                    "source": "in_the_wild", "tts_system": "human"})
                    bf_count += 1
        for label_dir_name in ("fake", "spoof", "synthetic"):
            ld = raw_dir / label_dir_name
            if ld.exists():
                files = find_audio_files(ld)
                random.shuffle(files)
                for f in files[:max_per_class]:
                    if sp_count >= max_per_class:
                        break
                    dest = spoof_dir / f"itw_spoof_{sp_count:04d}{f.suffix}"
                    shutil.copy2(f, dest)
                    records.append({"file": f"spoof/{dest.name}", "label": "spoof",
                                    "source": "in_the_wild", "tts_system": "unknown_itw"})
                    sp_count += 1
    
    logger.info("In-the-Wild: organized %d bonafide + %d spoof samples", bf_count, sp_count)
    return records


def copy_existing_bonafide(src_dir: Path, bonafide_dir: Path, max_count: int) -> list[dict]:
    records = []
    files = find_audio_files(src_dir)
    for i, f in enumerate(files[:max_count]):
        dest = bonafide_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
        records.append({"file": f"bonafide/{dest.name}", "label": "bonafide",
                        "source": "librispeech_dev_clean", "tts_system": "human"})
    logger.info("Existing bonafide: copied %d samples", len(records))
    return records


def main():
    parser = argparse.ArgumentParser(description="Download and organize modern deepfake datasets.")
    parser.add_argument("--out", default="data/modern_dataset", help="Output directory")
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="Max samples per class from each dataset")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, only organize existing data")
    parser.add_argument("--datasets", nargs="+", default=["mlaad", "in-the-wild"],
                        choices=list(DATASETS.keys()),
                        help="Which datasets to download (default: mlaad, in-the-wild)")
    parser.add_argument("--raw-dir", default="data/raw_downloads",
                        help="Where to store raw downloaded archives")
    args = parser.parse_args()

    random.seed(42)
    out_dir = Path(args.out)
    raw_dir = Path(args.raw_dir)
    bonafide_dir = out_dir / "bonafide"
    spoof_dir = out_dir / "spoof"
    bonafide_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    existing_bf = Path("data/large_dataset/bonafide")
    if existing_bf.exists():
        all_records.extend(copy_existing_bonafide(existing_bf, bonafide_dir, args.max_per_class))
    
    for ds_name in args.datasets:
        ds_info = DATASETS[ds_name]
        ds_raw_dir = raw_dir / ds_info["out_dir"]
        
        if not args.skip_download:
            archive_name = ds_info["url"].split("/")[-1]
            archive_path = raw_dir / archive_name
            if ds_raw_dir.exists() and any(ds_raw_dir.iterdir()):
                logger.info("%s already downloaded — skipping.", ds_name)
            else:
                if ds_info["size_gb"] > 10:
                    logger.warning("%s is %.0f GB. Consider --datasets without '%s' for faster start.",
                                   ds_name, ds_info["size_gb"], ds_name)
                if download_file(ds_info["url"], archive_path):
                    extract_archive(archive_path, ds_raw_dir)
                    archive_path.unlink(missing_ok=True)
                else:
                    logger.error("Failed to download %s — skipping.", ds_name)
                    continue
        
        if not ds_raw_dir.exists():
            logger.warning("%s raw directory not found: %s", ds_name, ds_raw_dir)
            continue
        
        if ds_name == "mlaad":
            all_records.extend(organize_mlaad(ds_raw_dir, bonafide_dir, spoof_dir, args.max_per_class))
        elif ds_name == "wavefake":
            all_records.extend(organize_wavefake(ds_raw_dir, bonafide_dir, spoof_dir, args.max_per_class))
        elif ds_name == "in-the-wild":
            all_records.extend(organize_in_the_wild(ds_raw_dir, bonafide_dir, spoof_dir, args.max_per_class))
    
    meta_path = out_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "tts_system"])
        writer.writeheader()
        writer.writerows(all_records)
    
    bf_count = sum(1 for r in all_records if r["label"] == "bonafide")
    sp_count = sum(1 for r in all_records if r["label"] == "spoof")
    sources = set(r["source"] for r in all_records)
    tts_systems = set(r["tts_system"] for r in all_records if r["label"] == "spoof")
    
    print("\n" + "=" * 60)
    print("MODERN DATASET READY")
    print("=" * 60)
    print(f"  Output:       {out_dir}")
    print(f"  Bonafide:     {bf_count} samples")
    print(f"  Spoof:        {sp_count} samples")
    print(f"  Sources:      {len(sources)} ({', '.join(sorted(sources))})")
    print(f"  TTS systems:  {len(tts_systems)} ({', '.join(sorted(tts_systems))})")
    print(f"  Metadata:     {meta_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
