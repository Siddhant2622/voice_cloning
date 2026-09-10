"""
data/upload_asvspoof_to_drive.py
================================
Uploads ASVspoof 2017 V2 dataset assets, metadata, representative sample clips,
and the hardened cm_detect2b_v3.pt model to Google Drive under 'VoiceGuard-Dataset'.
"""

import sys
import os
import time
import logging
import shutil
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_asvspoof_drive")

ROOT_FOLDER_NAME = "VoiceGuard-Dataset"


def get_drive():
    settings_file = "settings.yaml" if Path("settings.yaml").exists() else "data/gdrive_settings.yaml"
    gauth = GoogleAuth(settings_file=settings_file)
    gauth.LoadCredentialsFile("gdrive_credentials.json")
    if gauth.credentials is None or gauth.access_token_expired:
        logger.info("Refreshing credentials...")
        gauth.Refresh()
        gauth.SaveCredentialsFile("gdrive_credentials.json")
    else:
        gauth.Authorize()
    return GoogleDrive(gauth)


def get_or_create_folder(drive, name: str, parent_id: str = "root") -> str:
    query = (
        f"title='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    file_list = drive.ListFile({"q": query}).GetList()
    if file_list:
        logger.info("Found existing Drive folder '%s' (id=%s)", name, file_list[0]["id"])
        return file_list[0]["id"]

    folder = drive.CreateFile({
        "title": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"id": parent_id}],
    })
    folder.Upload()
    logger.info("Created Drive folder '%s' (id=%s)", name, folder["id"])
    return folder["id"]


def upload_file(drive, local_path: Path, parent_id: str, title: str = None, overwrite: bool = False) -> str:
    fname = title or local_path.name
    query = f"title='{fname}' and '{parent_id}' in parents and trashed=false"
    existing = drive.ListFile({"q": query}).GetList()

    if existing:
        if not overwrite:
            logger.info("  [EXISTS] %s already on Drive (id=%s)", fname, existing[0]["id"])
            return existing[0]["id"]
        else:
            gfile = existing[0]
            logger.info("  [UPDATING] %s (id=%s)...", fname, gfile["id"])
    else:
        gfile = drive.CreateFile({
            "title": fname,
            "parents": [{"id": parent_id}],
        })
        logger.info("  [UPLOADING] %s (%.2f MB)...", fname, local_path.stat().st_size / (1024 * 1024))

    gfile.SetContentFile(str(local_path))
    t0 = time.time()
    gfile.Upload()
    elapsed = time.time() - t0
    logger.info("  [DONE] Uploaded %s in %.1fs (id=%s)", fname, elapsed, gfile["id"])
    return gfile["id"]


def main():
    drive = get_drive()
    root_id = get_or_create_folder(drive, ROOT_FOLDER_NAME)
    drive_link = f"https://drive.google.com/drive/folders/{root_id}"
    logger.info("VoiceGuard-Dataset Root ID: %s", root_id)

    # 1. Create or get 'asvspoof2017' subfolder
    asv_folder_id = get_or_create_folder(drive, "asvspoof2017", parent_id=root_id)
    asv_dir = Path("data/asvspoof2017")

    # Upload protocol and dataset zip archives
    for zip_name in ["protocol_V2.zip", "ASVspoof2017_V2_dev.zip", "ASVspoof2017_V2_train.zip"]:
        p = asv_dir / zip_name
        if p.exists():
            upload_file(drive, p, asv_folder_id, overwrite=False)

    # Upload metadata CSVs
    if (asv_dir / "train" / "samples_metadata.csv").exists():
        upload_file(drive, asv_dir / "train" / "samples_metadata.csv", asv_folder_id,
                    title="ASVspoof2017_train_samples_metadata.csv", overwrite=True)

    if (asv_dir / "dev" / "samples_metadata.csv").exists():
        upload_file(drive, asv_dir / "dev" / "samples_metadata.csv", asv_folder_id,
                    title="ASVspoof2017_dev_samples_metadata.csv", overwrite=True)

    # Upload evaluation report
    eval_json = Path("results/asvspoof2017_eval_dev.json")
    if eval_json.exists():
        upload_file(drive, eval_json, asv_folder_id, title="asvspoof2017_benchmark_results.json", overwrite=True)

    # 2. Upload representative listenable audio clips into asvspoof2017/preview_samples
    preview_folder_id = get_or_create_folder(drive, "preview_samples", parent_id=asv_folder_id)
    dev_audio_dir = asv_dir / "ASVspoof2017_V2_dev"
    if dev_audio_dir.exists():
        import csv
        meta_dev = asv_dir / "dev" / "samples_metadata.csv"
        if meta_dev.exists():
            with open(meta_dev, "r", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
            # Pick 2 bonafide and 2 per mic R01-R07
            selected_files = []
            bf_count = 0
            mic_counts = {}
            for r in rows:
                if r["label"] == "bonafide" and bf_count < 4:
                    selected_files.append(Path(r["file"]))
                    bf_count += 1
                elif r["label"] == "spoof":
                    mic = r.get("recording_mic", "unknown")
                    if mic_counts.get(mic, 0) < 2:
                        selected_files.append(Path(r["file"]))
                        mic_counts[mic] = mic_counts.get(mic, 0) + 1

            logger.info("Uploading %d listenable audio preview samples to Drive...", len(selected_files))
            for f in selected_files:
                if f.exists():
                    upload_file(drive, f, preview_folder_id, overwrite=False)

    # 3. Upload hardened models to models/ folder
    model_folder_id = get_or_create_folder(drive, "models", parent_id=root_id)
    for model_file in [
        Path("models/cm_detect2b_v4.pt"),      # NEW: mobile-replay-hardened v4
        Path("models/cm_detect2b_v4.json"),
        Path("models/cm_detect2b_v3.pt"),
        Path("models/cm_detect2b_v3.json"),
        Path("models/cm_detect2b_v3_pre_asv17_backup.pt"),
    ]:
        if model_file.exists():
            upload_file(drive, model_file, model_folder_id, overwrite=True)

    # 4. Upload feature cache (v4 cache uses mobile_replay_chain augmentation)
    for cache_file in [
        Path("data/modern_features_cache_500_asv17_v4.pt"),
        Path("data/modern_features_cache_350_asv17_v4.pt"),
        Path("data/modern_features_cache_350_asv17.pt"),  # keep v3 cache too
    ]:
        if cache_file.exists():
            upload_file(drive, cache_file, asv_folder_id, overwrite=True)

    print("\n" + "=" * 60)
    print("[OK] DATASET & HARDENED MODEL SUCCESSFULLY PUSHED TO DRIVE!")
    print(f"Google Drive URL: {drive_link}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
