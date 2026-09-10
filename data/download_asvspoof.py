"""
data/download_asvspoof.py
=========================
Download ASVspoof datasets:
  - ASVspoof 2017 V2 (Physical Access / Replay & Microphone Transmission)
    Source: http://dx.doi.org/10.7488/ds/2332 (Edinburgh DataShare handle 10283/3055)
  - ASVspoof 2019 LA (Logical Access / Digital Synthesis)
    Source: https://datashare.ed.ac.uk/handle/10283/3336

Usage:
    python data/download_asvspoof.py --version 2017 --partition train dev
    python data/download_asvspoof.py --version 2017 --partition eval
    python data/download_asvspoof.py --version 2019 --partition eval
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR_2019 = Path(__file__).parent / "ASVspoof2019_LA"
BASE_DIR_2017 = Path(__file__).parent / "asvspoof2017"

# Edinburgh DataShare direct bitstream endpoints for ASVspoof 2017 Version 2 (Handle 10283/3055)
ASVSPOOF_2017_URLS = {
    "protocol": {
        "url": "https://datashare.ed.ac.uk/bitstreams/dbf267c4-517e-4193-9cdf-c8eadc82ec78/download",
        "filename": "protocol_V2.zip",
        "size_mb": 0.1,
        "desc": "Protocol files & ground truth trial keys"
    },
    "train": {
        "url": "https://datashare.ed.ac.uk/bitstreams/4c7e2262-fe78-497e-839f-0ceabbbf2d1e/download",
        "filename": "ASVspoof2017_V2_train.zip",
        "size_mb": 200.8,
        "desc": "Training partition (3,014 files: 1,507 genuine + 1,507 replay)"
    },
    "dev": {
        "url": "https://datashare.ed.ac.uk/bitstreams/4daef0d3-f9e8-49e4-9ffc-a7362842a8f2/download",
        "filename": "ASVspoof2017_V2_dev.zip",
        "size_mb": 133.8,
        "desc": "Development partition (1,710 files across 7 microphones)"
    },
    "eval": {
        "url": "https://datashare.ed.ac.uk/bitstreams/77c52086-76ed-4517-a72a-cc94e54a2c0a/download",
        "filename": "ASVspoof2017_V2_eval.zip",
        "size_mb": 1070.0,
        "desc": "Evaluation partition (13,306 files across 24 microphones)"
    }
}


def _progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        mb_done = count * block_size / 1_048_576
        mb_total = total_size / 1_048_576
        bar = "#" * (pct // 4) + "." * (25 - (pct // 4))
        print(f"\r  [{bar}] {pct:3d}%  {mb_done:6.1f} / {mb_total:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path, max_retries: int = 8) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Target: {url}")
    print(f"  Destination: {dest}")

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = dest.stat().st_size if dest.exists() else 0
            # Query server for total file size
            head_req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(head_req, timeout=30) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))

            if total_size > 0 and downloaded >= total_size:
                print(f"  [OK] Already fully downloaded ({downloaded / 1e6:.1f} MB).")
                return True

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                mode = "ab"
                print(f"  Resuming from {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB (attempt {attempt}/{max_retries})...")
            else:
                mode = "wb"
                print(f"  Starting download (attempt {attempt}/{max_retries})...")

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp, open(dest, mode) as out_fp:
                t0 = time.time()
                bytes_this_session = 0
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    out_fp.write(chunk)
                    downloaded += len(chunk)
                    bytes_this_session += len(chunk)
                    if total_size > 0:
                        pct = min(100, downloaded * 100 // total_size)
                        elapsed = max(0.1, time.time() - t0)
                        speed_mb = (bytes_this_session / 1_048_576) / elapsed
                        bar = "#" * (pct // 4) + "." * (25 - (pct // 4))
                        print(f"\r  [{bar}] {pct:3d}%  {downloaded/1e6:6.1f} / {total_size/1e6:.1f} MB ({speed_mb:.1f} MB/s)", end="", flush=True)
            print()

            if total_size > 0 and downloaded < total_size:
                print(f"  Connection closed prematurely ({downloaded}/{total_size} bytes). Retrying in 2s...")
                time.sleep(2)
                continue
            return True
        except Exception as exc:
            print(f"\n  [WARN] Attempt {attempt} failed: {exc}")
            time.sleep(3)

    return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    print(f"  Extracting: {archive_path.name} -> {dest_dir}")
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir)
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        print("  [OK] Extracted.")
        return True
    except Exception as exc:
        print(f"  [FAIL] Extraction failed: {exc}")
        return False


def download_asvspoof_2017(partitions: list[str] = None, out_dir: Path = BASE_DIR_2017) -> Path:
    """
    Download ASVspoof 2017 Version 2 (Physical Access / Replay & Microphones).
    DOI: http://dx.doi.org/10.7488/ds/2332
    """
    if partitions is None:
        partitions = ["train", "dev"]

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("ASVspoof 2017 Version 2.0 Database Downloader")
    print("Physical Access (PA) & Microphone Replay Spoofing Challenge")
    print("DOI: http://dx.doi.org/10.7488/ds/2332 (Handle 10283/3055)")
    print("Target directory:", out_dir)
    print("=" * 70)

    # 1. Download protocol_V2.zip first
    proto_info = ASVSPOOF_2017_URLS["protocol"]
    proto_zip = out_dir / proto_info["filename"]
    proto_dir = out_dir / "protocol_V2"

    if not proto_dir.exists():
        if not proto_zip.exists():
            print(f"\n[1/4] Downloading {proto_info['desc']}...")
            download_file(proto_info["url"], proto_zip)
        if proto_zip.exists():
            extract_archive(proto_zip, out_dir)
    else:
        print("\n[OK] protocol_V2 already extracted.")

    # 2. Download requested partitions
    for p in partitions:
        if p not in ASVSPOOF_2017_URLS:
            continue
        info = ASVSPOOF_2017_URLS[p]
        zip_path = out_dir / info["filename"]
        target_audio_dir = out_dir / f"ASVspoof2017_V2_{p}"

        print(f"\n=== Partition: {p.upper()} ({info['desc']}) ===")
        if not target_audio_dir.exists() or not any(target_audio_dir.glob("*.wav")):
            is_valid_zip = zip_path.exists() and zipfile.is_zipfile(zip_path)
            if not is_valid_zip:
                print(f"Downloading {info['filename']} (~{info['size_mb']:.1f} MB)...")
                ok = download_file(info["url"], zip_path)
                if not ok:
                    print(f"[FAIL] Could not download {info['filename']}.")
                    continue
            if zip_path.exists() and zipfile.is_zipfile(zip_path):
                extract_archive(zip_path, out_dir)
            else:
                print(f"[FAIL] Archive {zip_path.name} is corrupt or incomplete.")
        else:
            print(f"[OK] Partition audio already present in {target_audio_dir}")

    # 3. Automatically run prepare_asvspoof2017
    print("\nRunning metadata preparation (prepare_asvspoof2017.py)...")
    try:
        from data.prepare_asvspoof2017 import prepare_split
        for p in partitions:
            prepare_split(out_dir, p, out_dir)
    except Exception as e:
        print(f"Preparation notice: {e}")

    print("\n[OK] ASVspoof 2017 V2 ready at:", out_dir)
    return out_dir


def download_asvspoof_2019_la(partition: str = "eval") -> Path:
    """Download ASVspoof 2019 LA partition."""
    print(f"\nASVspoof 2019 LA — downloading partition: {partition}")
    print("License: https://datashare.ed.ac.uk/handle/10283/3336")
    print("=" * 60)

    BASE_DIR_2019.mkdir(parents=True, exist_ok=True)
    archive_name = f"ASVspoof2019_LA_{partition}.tar.gz"
    archive_path = BASE_DIR_2019 / archive_name

    candidate_urls = [
        f"https://www.openslr.org/resources/105/{archive_name}",
        f"https://datashare.ed.ac.uk/download/{archive_name}",
    ]

    downloaded = False
    if not archive_path.exists():
        for url in candidate_urls:
            print(f"\nTrying: {url}")
            if download_file(url, archive_path):
                downloaded = True
                break
    if archive_path.exists():
        extract_archive(archive_path, BASE_DIR_2019)
    return BASE_DIR_2019


def main():
    parser = argparse.ArgumentParser(description="Download ASVspoof deepfake detection datasets.")
    parser.add_argument(
        "--version", choices=["2017", "2019"], default="2017",
        help="ASVspoof dataset version: 2017 (Physical Replay & Mic) or 2019 (Logical Access). [default: 2017]"
    )
    parser.add_argument(
        "--partition", nargs="+", default=["train", "dev"],
        choices=["train", "dev", "eval", "all"],
        help="Partitions to download. [default: train dev]"
    )
    parser.add_argument("--out", default=None, help="Output destination directory.")
    args = parser.parse_args()

    partitions = ["train", "dev", "eval"] if "all" in args.partition else args.partition

    if args.version == "2017":
        out = Path(args.out) if args.out else BASE_DIR_2017
        download_asvspoof_2017(partitions=partitions, out_dir=out)
    else:
        for p in partitions:
            download_asvspoof_2019_la(p)


if __name__ == "__main__":
    main()
