"""
data/download_from_drive.py
===========================
Fetches VoiceGuard datasets, physical microphone data (ASVspoof 2017 V2),
and the hardened DETECT-2B v3 model checkpoint from Google Drive.

Supports:
  1. PyDrive2 authentication (using settings.yaml / gdrive_credentials.json).
  2. Unauthenticated fallback via gdown / direct Google Drive download IDs.

Folder: https://drive.google.com/drive/folders/1B5xNpwBbyI8a_RyK86wAFCl82K-ExjNe

Usage:
    python data/download_from_drive.py --target asvspoof2017
    python data/download_from_drive.py --target models
    python data/download_from_drive.py --target all
    python data/download_from_drive.py --extract
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drive_downloader")

GDRIVE_ROOT_FOLDER_ID = "1B5xNpwBbyI8a_RyK86wAFCl82K-ExjNe"
GDRIVE_ASVSPOOF_FOLDER_ID = "1MPUMacw1TEhrPaPPLgeghCApJ879sYe1"
GDRIVE_MODELS_FOLDER_ID = "13RT_FZUU0c3WJK1pmh9GZl0QsxFym0ho"

# Known direct file IDs on Google Drive
KNOWN_DRIVE_FILES = {
    # ASVspoof 2017 V2 Physical Microphones & Replay Data
    "protocol_V2.zip": {
        "id": "1qU4X3CBBdpJzZHI-jQGKv_sy-fIxTddP",
        "dest": Path("data/asvspoof2017/protocol_V2.zip"),
        "extract_to": Path("data/asvspoof2017"),
    },
    "ASVspoof2017_V2_dev.zip": {
        "id": "10gH9VXKvWj0RYmDH56tdF7S48o4pW6RL",
        "dest": Path("data/asvspoof2017/ASVspoof2017_V2_dev.zip"),
        "extract_to": Path("data/asvspoof2017"),
    },
    "ASVspoof2017_V2_train.zip": {
        "id": "1TV_9eKZeDPCVEudT2ChCGgE5B9OeqaVt",
        "dest": Path("data/asvspoof2017/ASVspoof2017_V2_train.zip"),
        "extract_to": Path("data/asvspoof2017"),
    },
    "ASVspoof2017_train_samples_metadata.csv": {
        "id": "1pznp0Fstv0ZN844ZjKYiHlit2biSKm3U",
        "dest": Path("data/asvspoof2017/train/samples_metadata.csv"),
    },
    "ASVspoof2017_dev_samples_metadata.csv": {
        "id": "1NyZeOWZK9lVS5DnBxGuQhSvEQSnhk37K",
        "dest": Path("data/asvspoof2017/dev/samples_metadata.csv"),
    },
    "asvspoof2017_benchmark_results.json": {
        "id": "1-9iTyg85zimz89hPnWyuKuCqBH0aQM9O",
        "dest": Path("results/asvspoof2017_eval_dev.json"),
    },
    "modern_features_cache_350_asv17.pt": {
        "id": "1lPHwvQx7mS2sqsTP9hKnhtV6Mq8B8Jcb",
        "dest": Path("data/modern_features_cache_350_asv17.pt"),
    },
    # Hardened Model Checkpoint (cm_detect2b_v3.pt)
    "cm_detect2b_v3.pt": {
        "id": "15z-8vfrTvu9UrKX8YrQQRoRbV_PYRkEd",
        "dest": Path("models/cm_detect2b_v3.pt"),
    },
    "cm_detect2b_v3.json": {
        "id": "1SupFzBDRniGaPPEDJ1OrOBRQrKxqzb5Z",
        "dest": Path("models/cm_detect2b_v3.json"),
    },
}


def get_authenticated_drive():
    """Try to initialize authenticated PyDrive2 client."""
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive

        settings_file = "settings.yaml" if Path("settings.yaml").exists() else "data/gdrive_settings.yaml"
        if not Path(settings_file).exists() and not Path("gdrive_credentials.json").exists():
            return None

        gauth = GoogleAuth(settings_file=settings_file)
        if Path("gdrive_credentials.json").exists():
            gauth.LoadCredentialsFile("gdrive_credentials.json")
            if gauth.credentials is None or gauth.access_token_expired:
                try:
                    gauth.Refresh()
                    gauth.SaveCredentialsFile("gdrive_credentials.json")
                except Exception:
                    return None
            return GoogleDrive(gauth)
    except Exception as e:
        logger.debug("PyDrive2 initialization skipped: %s", e)
    return None


def download_with_gdown(file_id: str, dest_path: Path) -> bool:
    """Download file using gdown."""
    try:
        import gdown
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        logger.info("Downloading %s via gdown...", dest_path.name)
        gdown.download(url, str(dest_path), quiet=False)
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("gdown download failed for %s: %s", dest_path.name, exc)
        return False


def download_file_by_id(file_id: str, dest_path: Path, drive=None) -> bool:
    """Download a file by Drive ID using PyDrive2 or gdown fallback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if drive is not None:
        try:
            logger.info("Downloading %s via PyDrive2...", dest_path.name)
            gfile = drive.CreateFile({"id": file_id})
            gfile.GetContentFile(str(dest_path))
            logger.info("Successfully downloaded %s (%.2f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))
            return True
        except Exception as e:
            logger.warning("PyDrive2 fetch failed for %s (%s) — falling back to gdown", dest_path.name, e)

    return download_with_gdown(file_id, dest_path)


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip archive if destination is not yet populated."""
    if not zip_path.exists():
        return
    logger.info("Extracting %s -> %s...", zip_path.name, extract_dir)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("[OK] Extracted %s", zip_path.name)
    except Exception as e:
        logger.error("Failed to extract %s: %s", zip_path.name, e)


def fetch_from_drive(target: str = "asvspoof2017", extract: bool = True):
    """Fetch requested dataset or model components from Google Drive."""
    drive = get_authenticated_drive()
    if drive:
        logger.info("Authenticated with Google Drive via PyDrive2.")
    else:
        logger.info("Operating in standalone mode using Google Drive direct bitstreams.")

    items_to_download = []
    if target in ("all", "asvspoof2017"):
        items_to_download.extend([
            "protocol_V2.zip",
            "ASVspoof2017_V2_dev.zip",
            "ASVspoof2017_V2_train.zip",
            "ASVspoof2017_train_samples_metadata.csv",
            "ASVspoof2017_dev_samples_metadata.csv",
            "asvspoof2017_benchmark_results.json",
            "modern_features_cache_350_asv17.pt",
        ])

    if target in ("all", "models"):
        items_to_download.extend([
            "cm_detect2b_v3.pt",
            "cm_detect2b_v3.json",
        ])

    for item_key in items_to_download:
        if item_key not in KNOWN_DRIVE_FILES:
            continue
        info = KNOWN_DRIVE_FILES[item_key]
        dest = info["dest"]
        file_id = info["id"]

        if dest.exists() and dest.stat().st_size > 0:
            logger.info("[EXISTS] %s is already present locally at %s", item_key, dest)
        else:
            success = download_file_by_id(file_id, dest, drive=drive)
            if not success:
                logger.error("Could not fetch %s from Drive.", item_key)

        if extract and "extract_to" in info and dest.exists():
            extract_zip(dest, info["extract_to"])

    # If ASVspoof was fetched, ensure prepare_asvspoof2017 is run if needed
    if target in ("all", "asvspoof2017"):
        asv_root = Path("data/asvspoof2017")
        try:
            from data.prepare_asvspoof2017 import prepare_split
            for split in ["train", "dev"]:
                meta = asv_root / split / "samples_metadata.csv"
                if not meta.exists() and (asv_root / f"ASVspoof2017_V2_{split}").exists():
                    prepare_split(asv_root, split, asv_root)
        except Exception as e:
            logger.debug("prepare_asvspoof notice: %s", e)

    logger.info("Google Drive synchronization completed for target '%s'.", target)


def main():
    parser = argparse.ArgumentParser(description="Fetch VoiceGuard datasets and models from Google Drive.")
    parser.add_argument("--target", choices=["asvspoof2017", "models", "all"], default="asvspoof2017",
                        help="Components to fetch from Google Drive.")
    parser.add_argument("--no-extract", action="store_true", help="Do not extract downloaded zip archives.")
    args = parser.parse_args()

    fetch_from_drive(target=args.target, extract=not args.no_extract)


if __name__ == "__main__":
    main()
