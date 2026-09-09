"""
data/upload_to_drive.py
========================
Uploads the VoiceGuard training dataset to Google Drive.

Steps:
  1. First run: opens browser for Google OAuth consent → saves credentials.json
  2. Creates folder structure on Drive:
       VoiceGuard-Dataset/
         bonafide/      ← real human speech (LibriSpeech)
         spoof/         ← synthetic AI speech (edge-tts, gTTS, pyttsx3)
         samples_metadata.csv
  3. Uploads all audio files with progress reporting.

Usage:
    # Step 1: Set up OAuth credentials (one-time)
    #   → Go to: https://console.cloud.google.com/
    #   → Create project → Enable Drive API → Create OAuth 2.0 Desktop credentials
    #   → Download as client_secrets.json and place in this folder
    #
    # Step 2: Run upload
    python data/upload_to_drive.py --data data/large_dataset

Requirements:
    pip install pydrive2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger("drive_upload")


# ---------------------------------------------------------------------------
# OAuth & Drive client setup
# ---------------------------------------------------------------------------
def get_drive_client(secrets_file: str = "client_secrets.json"):
    """
    Authenticate with Google Drive using OAuth2.
    Opens browser on first run to grant access.
    Caches credentials in 'gdrive_credentials.json' for future runs.
    """
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive

    # Write settings.yaml for pydrive2 if not present
    settings_path = Path("settings.yaml")
    if not settings_path.exists():
        settings_path.write_text(f"""
client_config_backend: file
client_config_file: {secrets_file}

save_credentials: True
save_credentials_backend: file
save_credentials_file: gdrive_credentials.json

get_refresh_token: True

oauth_scope:
  - https://www.googleapis.com/auth/drive
""")

    gauth = GoogleAuth(settings_file=str(settings_path))

    # Try to load saved credentials
    creds_file = "gdrive_credentials.json"
    gauth.LoadCredentialsFile(creds_file)

    if gauth.credentials is None:
        # Use CommandLineAuth: prints a URL, you paste back the code.
        # More reliable than LocalWebserverAuth in scripted environments.
        logger.info("="*60)
        logger.info("GOOGLE DRIVE AUTHENTICATION")
        logger.info("="*60)
        logger.info("1. Open this URL in your browser:")
        logger.info("2. Log in with your Google account")
        logger.info("3. Click Allow, then PASTE the code shown back here")
        logger.info("="*60)
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        logger.info("Refreshing expired Google OAuth token...")
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile(creds_file)
    drive = GoogleDrive(gauth)
    logger.info("[OK] Google Drive authenticated.")
    return drive


# ---------------------------------------------------------------------------
# Drive folder helpers
# ---------------------------------------------------------------------------
def get_or_create_folder(drive, name: str, parent_id: str = "root") -> str:
    """Get or create a folder on Drive. Returns folder ID."""
    query = (
        f"title='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    file_list = drive.ListFile({"q": query}).GetList()
    if file_list:
        folder_id = file_list[0]["id"]
        logger.info("Found existing Drive folder: %s (id=%s)", name, folder_id)
        return folder_id

    folder = drive.CreateFile({
        "title": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"id": parent_id}],
    })
    folder.Upload()
    logger.info("Created Drive folder: %s (id=%s)", name, folder["id"])
    return folder["id"]


def upload_file(drive, local_path: Path, parent_id: str, overwrite: bool = False) -> str:
    """Upload a single file to Drive folder. Returns file ID."""
    fname = local_path.name

    if not overwrite:
        # Check if file already exists
        query = f"title='{fname}' and '{parent_id}' in parents and trashed=false"
        existing = drive.ListFile({"q": query}).GetList()
        if existing:
            return existing[0]["id"]  # Skip upload

    gfile = drive.CreateFile({
        "title": fname,
        "parents": [{"id": parent_id}],
    })
    gfile.SetContentFile(str(local_path))
    gfile.Upload()
    return gfile["id"]


# ---------------------------------------------------------------------------
# Main upload logic
# ---------------------------------------------------------------------------
def upload_dataset(drive, data_dir: Path, root_folder_name: str = "VoiceGuard-Dataset"):
    """Upload the complete dataset to Google Drive."""
    logger.info("=== Uploading dataset to Google Drive ===")
    logger.info("Source: %s", data_dir)

    # Create root folder
    root_id = get_or_create_folder(drive, root_folder_name)
    logger.info("Drive root folder: %s (id=%s)", root_folder_name, root_id)

    # Get Drive folder link
    drive_link = f"https://drive.google.com/drive/folders/{root_id}"
    logger.info("Drive URL: %s", drive_link)

    total_uploaded = 0
    total_skipped  = 0
    total_bytes    = 0

    # Upload each subdirectory
    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    subdirs = ["bonafide", "spoof"]

    for subdir_name in subdirs:
        subdir = data_dir / subdir_name
        if not subdir.exists():
            logger.warning("Subdir not found, skipping: %s", subdir)
            continue

        files = [f for f in subdir.iterdir() if f.suffix.lower() in audio_exts]
        if not files:
            continue

        logger.info("Uploading %s/ (%d files)...", subdir_name, len(files))
        folder_id = get_or_create_folder(drive, subdir_name, parent_id=root_id)

        for i, f in enumerate(files):
            try:
                fid = upload_file(drive, f, folder_id, overwrite=False)
                size = f.stat().st_size
                total_bytes += size
                total_uploaded += 1
                if (i + 1) % 20 == 0 or (i + 1) == len(files):
                    logger.info(
                        "  %s: %d/%d  (%.1f MB total)",
                        subdir_name, i + 1, len(files), total_bytes / 1_048_576
                    )
            except Exception as exc:
                logger.warning("Failed to upload %s: %s", f.name, exc)
                total_skipped += 1
                time.sleep(1)  # Back-off on error

    # Upload metadata CSV
    csv_file = data_dir / "samples_metadata.csv"
    if csv_file.exists():
        logger.info("Uploading samples_metadata.csv...")
        upload_file(drive, csv_file, root_id, overwrite=True)

    logger.info("=== Upload complete ===")
    logger.info("  Uploaded:   %d files (%.1f MB)", total_uploaded, total_bytes / 1_048_576)
    logger.info("  Skipped:    %d files (already exist)", total_skipped)
    logger.info("  Drive URL:  %s", drive_link)

    return drive_link


def main():
    parser = argparse.ArgumentParser(description="Upload training dataset to Google Drive.")
    parser.add_argument("--data",     default="data/large_dataset",  help="Local dataset directory.")
    parser.add_argument("--folder",   default="VoiceGuard-Dataset",  help="Google Drive folder name.")
    parser.add_argument("--secrets",  default="client_secrets.json", help="OAuth client secrets file.")
    parser.add_argument("--overwrite",action="store_true",           help="Overwrite existing files on Drive.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Dataset directory not found: %s", data_dir)
        logger.error("Run: python data/build_large_dataset.py first")
        sys.exit(1)

    secrets = Path(args.secrets)
    if not secrets.exists():
        print("\n" + "="*60)
        print("GOOGLE DRIVE SETUP — ONE TIME ONLY")
        print("="*60)
        print()
        print("To upload to Google Drive, you need OAuth 2.0 credentials.")
        print()
        print("Steps:")
        print("  1. Go to: https://console.cloud.google.com/apis/credentials")
        print("  2. Click 'Create Credentials' → 'OAuth 2.0 Client ID'")
        print("  3. Application type: 'Desktop app'")
        print("  4. Download the JSON → rename to 'client_secrets.json'")
        print("  5. Place in:", Path.cwd())
        print("  6. Run this script again")
        print()
        print("  Alternatively: use the service account JSON file if")
        print("  you already have one configured for your project.")
        print("="*60 + "\n")
        sys.exit(1)

    drive = get_drive_client(secrets_file=str(secrets))
    drive_link = upload_dataset(drive, data_dir, root_folder_name=args.folder)

    # Save the Drive link for reference
    link_file = data_dir / "gdrive_link.txt"
    link_file.write_text(drive_link)
    logger.info("Drive link saved to: %s", link_file)

    print(f"\n[OK] Dataset uploaded to Google Drive!")
    print(f"     URL: {drive_link}")


if __name__ == "__main__":
    main()
