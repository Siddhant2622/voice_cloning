"""
data/download_datasets.py
=========================
Automated downloader for public Audio Deepfake Detection datasets.

Supports:
  - ASVspoof 2019 LA  (Edinburgh DataShare — requires free registration, provides direct link)
  - WaveFake          (Zenodo — fully open)
  - In-the-Wild       (Fraunhofer AISEC — fully open)
  - MLAAD             (deepfake-total.com — fully open)
  - Hugging Face Hub  (push / pull model checkpoints)

Usage:
    python data/download_datasets.py --dataset asvspoof2019 --out data/
    python data/download_datasets.py --dataset wavefake --out data/
    python data/download_datasets.py --dataset in-the-wild --out data/
    python data/download_datasets.py --dataset mlaad --out data/
    python data/download_datasets.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger("download_datasets")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "asvspoof2019": {
        "description": "ASVspoof 2019 Logical Access (LA) — 118K utterances, 19 TTS/VC attacks",
        "note": (
            "ASVspoof 2019 requires a free DataShare account.\n"
            "1. Register at: https://datashare.ed.ac.uk/handle/10283/3336\n"
            "2. Download 'LA.zip' (~2.6 GB) manually.\n"
            "3. Pass the path: python data/prepare_asvspoof.py --zip path/to/LA.zip --out data/asvspoof2019\n"
            "Alternatively, run on Google Colab — see notebooks/colab_train.py for the full pipeline."
        ),
        "colab_cell": "asvspoof2019",
        "size_gb": 2.6,
    },
    "wavefake": {
        "description": "WaveFake — 6 vocoders × LJSpeech (GriffinLim, MelGAN, MB-MelGAN, FullBand-MelGAN, WaveGlow, ParallelWaveGAN)",
        "url": "https://zenodo.org/record/5642694/files/WaveFake.zip",
        "sha256": None,  # Large file — skip checksum
        "out_dir": "wavefake",
        "size_gb": 19.0,
    },
    "in-the-wild": {
        "description": "In-the-Wild — real-world AI audio collected from the internet by Fraunhofer AISEC",
        "url": "https://deepfake-demo.aisec.fraunhofer.de/in_the_wild.zip",
        "sha256": None,
        "out_dir": "in_the_wild",
        "size_gb": 2.0,
    },
    "mlaad": {
        "description": "MLAAD — Multi-Language Audio Anti-Spoofing Dataset (23 TTS systems, 38 languages)",
        "url": "https://deepfake-total.com/mlaad/mlaad_v3.zip",
        "sha256": None,
        "out_dir": "mlaad",
        "size_gb": 3.0,
    },
    "fakeavceleb": {
        "description": "FakeAVCeleb — ~20K celebrity audio deepfakes",
        "note": (
            "FakeAVCeleb requires a request form.\n"
            "Submit at: https://github.com/DASH-Lab/FakeAVCeleb\n"
            "Once approved you'll receive a Google Drive link."
        ),
        "colab_cell": "fakeavceleb",
        "size_gb": 5.0,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading: %s", url)
    logger.info("  → %s", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(dest), reporthook=_progress_hook)
    logger.info("Download complete: %.1f MB", dest.stat().st_size / 1e6)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _extract(archive: Path, out_dir: Path) -> None:
    logger.info("Extracting %s → %s", archive.name, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip" or (archive.suffix == ".zip"):
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


# ---------------------------------------------------------------------------
# Dataset-specific downloaders
# ---------------------------------------------------------------------------

def download_wavefake(out_root: Path) -> Path:
    info = DATASETS["wavefake"]
    zip_path = out_root / "wavefake.zip"
    out_dir  = out_root / info["out_dir"]

    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info("WaveFake already exists at %s — skipping download.", out_dir)
        return out_dir

    _download(info["url"], zip_path)
    _extract(zip_path, out_dir)
    zip_path.unlink()  # free disk space
    return out_dir


def download_in_the_wild(out_root: Path) -> Path:
    info = DATASETS["in-the-wild"]
    zip_path = out_root / "in_the_wild.zip"
    out_dir  = out_root / info["out_dir"]

    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info("In-the-Wild already exists — skipping download.")
        return out_dir

    _download(info["url"], zip_path)
    _extract(zip_path, out_dir)
    zip_path.unlink()
    return out_dir


def download_mlaad(out_root: Path) -> Path:
    info = DATASETS["mlaad"]
    zip_path = out_root / "mlaad.zip"
    out_dir  = out_root / info["out_dir"]

    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info("MLAAD already exists — skipping download.")
        return out_dir

    _download(info["url"], zip_path)
    _extract(zip_path, out_dir)
    zip_path.unlink()
    return out_dir


# ---------------------------------------------------------------------------
# Hugging Face Hub helpers
# ---------------------------------------------------------------------------

def push_checkpoint_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    token: str | None = None,
) -> None:
    """
    Push a trained model checkpoint to Hugging Face Hub.

    Args:
        checkpoint_path:  Local path to .pt file
        repo_id:          HF repo, e.g. "your-username/voiceguard-cm"
        token:            HF write token (or set HF_TOKEN env var)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("pip install huggingface_hub")
        return

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("Set HF_TOKEN env var or pass --hf-token")
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="cm.pt",
        repo_id=repo_id,
        token=token,
    )
    logger.info("Checkpoint pushed to https://huggingface.co/%s", repo_id)


def pull_checkpoint_from_hub(
    repo_id: str,
    local_path: str | Path = "models/cm.pt",
    token: str | None = None,
) -> Path:
    """
    Download a trained checkpoint from Hugging Face Hub.

    Args:
        repo_id:     HF repo, e.g. "your-username/voiceguard-cm"
        local_path:  Where to save the checkpoint
        token:       HF read token (optional for public repos)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    token = token or os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id=repo_id,
        filename="cm.pt",
        local_dir=str(Path(local_path).parent),
        token=token,
    )
    dest = Path(local_path)
    shutil.copy(path, dest)
    logger.info("Checkpoint downloaded: %s", dest)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download audio deepfake detection datasets.")
    parser.add_argument("--dataset",   choices=list(DATASETS.keys()), help="Dataset to download.")
    parser.add_argument("--out",       default="data", help="Root output directory.")
    parser.add_argument("--list",      action="store_true", help="List available datasets.")
    parser.add_argument("--hf-push",   metavar="REPO_ID", help="Push checkpoint to HF Hub.")
    parser.add_argument("--hf-pull",   metavar="REPO_ID", help="Pull checkpoint from HF Hub.")
    parser.add_argument("--checkpoint",default="models/cm.pt", help="Local checkpoint path for HF ops.")
    parser.add_argument("--hf-token",  help="Hugging Face token (or set HF_TOKEN env var).")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name:20s}  ~{info['size_gb']:.0f} GB  —  {info['description']}")
        print()
        return

    if args.hf_push:
        push_checkpoint_to_hub(args.checkpoint, args.hf_push, args.hf_token)
        return

    if args.hf_pull:
        pull_checkpoint_from_hub(args.hf_pull, args.checkpoint, args.hf_token)
        return

    if not args.dataset:
        parser.print_help()
        return

    out_root = Path(args.out)
    info = DATASETS[args.dataset]

    # Manual-only datasets
    if "note" in info and "url" not in info:
        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"{'='*60}")
        print(info["note"])
        print(f"{'='*60}\n")
        return

    # Downloadable datasets
    if args.dataset == "wavefake":
        download_wavefake(out_root)
    elif args.dataset == "in-the-wild":
        download_in_the_wild(out_root)
    elif args.dataset == "mlaad":
        download_mlaad(out_root)
    else:
        logger.error("Dataset '%s' requires manual download. Run with --list for instructions.", args.dataset)


if __name__ == "__main__":
    main()
