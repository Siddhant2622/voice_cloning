"""
data/upload_extras.py
=====================
Uploads remaining assets to the Google Drive VoiceGuard-Dataset folder:
1. bonafide/ audio samples (50 diverse LibriSpeech clips)
2. dev-clean.tar.gz (complete LibriSpeech bonafide archive, 322 MB)
3. models/cm_detect2b_v2.pt + json (the trained DETECT-2B v2 model checkpoint)
4. README.md (dataset and model documentation)
"""

import sys
import time
import logging
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_extras")

ROOT_FOLDER_NAME = "VoiceGuard-Dataset"


def get_drive():
    gauth = GoogleAuth("data/gdrive_settings.yaml")
    gauth.LoadCredentialsFile("gdrive_credentials.json")
    if gauth.credentials is None or gauth.access_token_expired:
        gauth.Authorize()
        gauth.SaveCredentialsFile("gdrive_credentials.json")
    return GoogleDrive(gauth)


def get_or_create_folder(drive, name: str, parent_id: str = "root") -> str:
    query = (
        f"title='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    file_list = drive.ListFile({"q": query}).GetList()
    if file_list:
        return file_list[0]["id"]

    folder = drive.CreateFile({
        "title": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"id": parent_id}],
    })
    folder.Upload()
    logger.info("Created folder: %s (id=%s)", name, folder["id"])
    return folder["id"]


def upload_file(drive, local_path: Path, parent_id: str, overwrite: bool = False) -> str:
    fname = local_path.name
    if not overwrite:
        query = f"title='{fname}' and '{parent_id}' in parents and trashed=false"
        existing = drive.ListFile({"q": query}).GetList()
        if existing:
            return existing[0]["id"]

    gfile = drive.CreateFile({
        "title": fname,
        "parents": [{"id": parent_id}],
    })
    gfile.SetContentFile(str(local_path))
    gfile.Upload()
    logger.info("Uploaded %s (%.2f MB)", fname, local_path.stat().st_size / 1_048_576)
    return gfile["id"]


def main():
    drive = get_drive()
    root_id = get_or_create_folder(drive, ROOT_FOLDER_NAME)
    logger.info("Root folder ID: %s", root_id)

    # 1. Upload README.md
    readme_path = Path("data/large_dataset/README.md")
    if readme_path.exists():
        logger.info("Uploading README.md...")
        upload_file(drive, readme_path, root_id, overwrite=True)

    # 2. Upload bonafide/ samples
    bf_dir = Path("data/large_dataset/bonafide")
    if bf_dir.exists():
        bf_files = list(bf_dir.glob("*.flac")) + list(bf_dir.glob("*.wav"))
        logger.info("Uploading %d bonafide audio clips...", len(bf_files))
        bf_folder_id = get_or_create_folder(drive, "bonafide", parent_id=root_id)
        for i, f in enumerate(bf_files):
            upload_file(drive, f, bf_folder_id, overwrite=False)
            if (i + 1) % 10 == 0 or (i + 1) == len(bf_files):
                logger.info("  bonafide: %d/%d", i + 1, len(bf_files))

    # 3. Upload models/
    model_dir = Path("models")
    model_folder_id = get_or_create_folder(drive, "models", parent_id=root_id)
    for mfile in [model_dir / "cm_detect2b_v2.pt", model_dir / "cm_detect2b_v2.json"]:
        if mfile.exists():
            logger.info("Uploading model file: %s...", mfile.name)
            upload_file(drive, mfile, model_folder_id, overwrite=True)

    # 4. Upload dev-clean.tar.gz (LibriSpeech full dataset archive)
    tar_path = Path("data/large_dataset/dev-clean.tar.gz")
    if tar_path.exists():
        logger.info("Uploading LibriSpeech full archive (%s, %.1f MB)...", tar_path.name, tar_path.stat().st_size / 1_048_576)
        upload_file(drive, tar_path, root_id, overwrite=False)

    drive_url = f"https://drive.google.com/drive/folders/{root_id}"
    logger.info("All extras uploaded successfully!")
    logger.info("Google Drive URL: %s", drive_url)
    return drive_url


if __name__ == "__main__":
    main()
