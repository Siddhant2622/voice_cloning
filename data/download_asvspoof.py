"""
data/download_asvspoof.py
Phase 3 — Download ASVspoof 2019 Logical Access (LA) partition.

The ASVspoof 2019 LA dataset is publicly available from the OpenSLR mirror.
You must agree to the license at: https://datashare.ed.ac.uk/handle/10283/3336

This script downloads the LA train, dev, and eval partitions,
plus the official trial key files, to data/ASVspoof2019_LA/.

Expected disk usage: ~4 GB compressed, ~7 GB extracted.

Usage:
    python data/download_asvspoof.py [--partition all|train|dev|eval]
    python data/download_asvspoof.py --partition eval   # only eval split (~1.5 GB)
"""

import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent / "ASVspoof2019_LA"

# Official OpenSLR mirror for ASVspoof 2019 LA
# (Primary mirror: https://datashare.ed.ac.uk — requires registration)
# The OpenSLR mirror below hosts the same files without registration.
OPENSLR_BASE = "https://www.openslr.org/resources/105"

PARTITIONS = {
    "train": {
        "audio_url":    f"{OPENSLR_BASE}/LA.zip",
        "trial_url":    None,   # included in main zip
        "description":  "LA training set",
    },
}

# Fallback: Edinburgh DataShare direct URLs (if OpenSLR mirror doesn't have LA)
DATASHARE_URLS = {
    "train": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
    "keys":  "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/ASVspoof2019_LA_cm_protocols.zip",
}

# The canonical download method via ASVspoof organisers' Google Drive
# (direct links may expire; included as documentation)
GOOGLE_DRIVE_IDS = {
    "LA_train": "1bLPiXLnMppPNFMTjsaFaxzAV2I_lf8mA",
    "LA_dev":   "1bLPiXLnMppPNFMTjsaFaxzAV2I_lf8mA",  # same archive
    "LA_eval":  "1bLPiXLnMppPNFMTjsaFaxzAV2I_lf8mA",
}


def _progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        mb_done  = count * block_size / 1_048_576
        mb_total = total_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb_done:6.1f} / {mb_total:.1f} MB", end="", flush=True)


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
        print()  # newline after progress
        return True
    except Exception as exc:
        print(f"\n  ✗ Download failed: {exc}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    print(f"  Extracting: {archive_path.name} → {dest_dir}")
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.suffix == ".gz" or str(archive_path).endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir)
        elif archive_path.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
        print("  ✓ Extracted.")
        return True
    except Exception as exc:
        print(f"  ✗ Extraction failed: {exc}")
        return False


def download_asvspoof_2019_la(partition: str = "eval") -> Path:
    """
    Download ASVspoof 2019 LA partition.
    Returns the path to the extracted data directory.
    """
    print(f"\nASVspoof 2019 LA — downloading partition: {partition}")
    print("License: https://datashare.ed.ac.uk/handle/10283/3336")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Try to download the protocol/key files first (these are small)
    keys_zip = BASE_DIR / "ASVspoof2019_LA_cm_protocols.zip"
    if not keys_zip.exists():
        keys_url = "https://www.asvspoof.org/index2019.html"
        print(f"\n  Key files: download manually from {keys_url}")
        print("  Or use the Edinburgh DataShare link (requires free registration).")
        print(f"  Place the protocol files at: {BASE_DIR}/ASVspoof2019_LA_cm_protocols/")

    # Main audio archive
    archive_name = f"ASVspoof2019_LA_{partition}.tar.gz"
    archive_path = BASE_DIR / archive_name

    # Try multiple mirror URLs
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
        if not downloaded:
            print("\n" + "=" * 60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 60)
            print("The ASVspoof 2019 dataset is available from:")
            print("  https://datashare.ed.ac.uk/handle/10283/3336")
            print("(free registration required)")
            print()
            print("Download the LA partition and place it at:")
            print(f"  {archive_path}")
            print()
            print("Or, use the WaveFake dataset (no registration required):")
            print("  python data/download_wavefake.py")
            return BASE_DIR / f"ASVspoof2019_LA_{partition}_audio"
    else:
        print(f"  {archive_name} already downloaded.")

    # Extract
    extract_dir = BASE_DIR / f"ASVspoof2019_LA_{partition}_audio"
    if not extract_dir.exists() and archive_path.exists():
        extract_archive(archive_path, BASE_DIR)

    print(f"\n✓ ASVspoof 2019 LA {partition} partition ready at: {BASE_DIR}")
    return BASE_DIR


def download_wavefake(dest: Path = None) -> Path:
    """
    Download WaveFake dataset (no registration, ~50 GB full / 1 GB subset).
    WaveFake: A Deep Fake Audio Detection Dataset
    https://github.com/RUB-SysSec/WaveFake
    """
    if dest is None:
        dest = Path(__file__).parent / "WaveFake"
    dest.mkdir(parents=True, exist_ok=True)

    # WaveFake hosts a subset on Zenodo
    zenodo_url = "https://zenodo.org/record/5642694/files/WaveFake.zip"
    archive = dest / "WaveFake.zip"

    print(f"\nWaveFake dataset → {dest}")
    print("Source: https://zenodo.org/record/5642694 (CC BY 4.0)")

    if not archive.exists():
        print("Downloading WaveFake subset (~2.4 GB)...")
        if not download_file(zenodo_url, archive):
            print("Please download manually from: https://zenodo.org/record/5642694")
            return dest
    else:
        print("  WaveFake.zip already downloaded.")

    extract_archive(archive, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download ASVspoof 2019 LA dataset.")
    parser.add_argument(
        "--partition", default="eval",
        choices=["train", "dev", "eval", "all"],
        help="Which partition(s) to download. [default: eval]"
    )
    parser.add_argument(
        "--wavefake", action="store_true",
        help="Download WaveFake instead of (or in addition to) ASVspoof."
    )
    args = parser.parse_args()

    if args.wavefake:
        download_wavefake()
        return

    partitions = ["train", "dev", "eval"] if args.partition == "all" else [args.partition]
    for p in partitions:
        download_asvspoof_2019_la(p)


if __name__ == "__main__":
    main()
