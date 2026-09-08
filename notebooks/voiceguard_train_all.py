"""
VoiceGuard — Complete Colab Training (Single File)
===================================================
UPLOAD THIS FILE to Google Colab and click Runtime > Run all

HOW TO USE:
  1. Go to https://colab.research.google.com
  2. Click File > Upload notebook > Upload
  3. Select THIS file (voiceguard_train_all.py)
  4. OR: File > New notebook, then click + Code and paste everything below

OR just copy-paste the entire content of this file into ONE Colab cell.
"""

import os
import sys

# ── Your credentials (configure via environment or edit here) ────────────────
GITHUB_REPO = os.environ.get("GITHUB_REPO", "https://github.com/Siddhant2622/Voice-Cloning")
HF_REPO_ID  = os.environ.get("HF_REPO_ID", "C00EX2622/voiceguard-cm")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
# ─────────────────────────────────────────────────────────────────────────────
import json
import zipfile
import subprocess
import urllib.request

print("=" * 60)
print("  VoiceGuard Cloud Training Pipeline")
print("=" * 60)

# ── Step 0: Install dependencies ─────────────────────────────────────────────
print("\n[1/7] Installing dependencies...")
os.system("pip install -q transformers librosa soundfile scipy scikit-learn huggingface_hub rich tqdm 2>&1")
print("✓ Dependencies ready")

# ── Step 1: Mount Google Drive ────────────────────────────────────────────────
print("\n[2/7] Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = "/content/drive/MyDrive/voiceguard"
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    print(f"✓ Drive mounted → {DRIVE_ROOT}")
except ImportError:
    DRIVE_ROOT = "/content/voiceguard_data"
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    print(f"⚠ Not in Colab, using local path: {DRIVE_ROOT}")

# ── Step 2: Clone repo ────────────────────────────────────────────────────────
print("\n[3/7] Cloning repository...")
REPO_DIR = "/content/Voice-Cloning"
if not os.path.exists(REPO_DIR):
    ret = os.system(f"git clone {GITHUB_REPO} {REPO_DIR}")
    if ret != 0:
        print("ERROR: git clone failed. Check your GITHUB_REPO URL.")
        sys.exit(1)
else:
    print(f"✓ Repo already exists at {REPO_DIR}")

os.chdir(REPO_DIR)
os.system("pip install -q -r requirements.txt 2>&1")
print(f"✓ Repo ready at {REPO_DIR}")

# ── Step 3: Download ASVspoof 2019 LA ─────────────────────────────────────────
print("\n[4/7] Downloading ASVspoof 2019 LA dataset (~2.6 GB)...")
DATA_DIR = f"{DRIVE_ROOT}/asvspoof2019_la"
os.makedirs(DATA_DIR, exist_ok=True)

ZENODO = "https://zenodo.org/record/5105682/files/"
FILES = [
    "ASVspoof2019_LA_cm_protocols.zip",
    "ASVspoof2019_LA_train.zip",
    "ASVspoof2019_LA_dev.zip",
    "ASVspoof2019_LA_eval.zip",
]

def dl_progress(block, bsize, total):
    done = min(block * bsize, total)
    pct = done / total * 100 if total > 0 else 0
    print(f"  {pct:5.1f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)", end="\r")

for fname in FILES:
    dest = f"{DATA_DIR}/{fname}"
    if os.path.exists(dest):
        print(f"  ✓ Already downloaded: {fname}")
        continue
    print(f"  Downloading {fname}...")
    urllib.request.urlretrieve(f"{ZENODO}{fname}", dest, reporthook=dl_progress)
    print(f"\n  ✓ {fname} done")

print("  Extracting archives...")
for fname in FILES:
    path = f"{DATA_DIR}/{fname}"
    if os.path.exists(path):
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(DATA_DIR)
print("✓ Dataset ready!")

# ── Step 4: Prepare dataset (protocol → CSV) ──────────────────────────────────
print("\n[5/7] Preparing dataset...")
result = subprocess.run([
    "python", "data/prepare_asvspoof.py",
    "--dir",   f"{DATA_DIR}",
    "--out",   f"{DRIVE_ROOT}/asvspoof2019_prepared",
    "--split", "all",
])
if result.returncode != 0:
    print("ERROR: Data preparation failed.")
    sys.exit(1)
print("✓ Dataset prepared!")

# ── Step 5: Train CM classifier ───────────────────────────────────────────────
print("\n[6/7] Training CM classifier (this takes ~3 hours on T4 GPU)...")
print("      Checkpoint saves to Google Drive so you can disconnect safely.")
OUT_MODEL = f"{DRIVE_ROOT}/models/cm_asvspoof19.pt"
os.makedirs(f"{DRIVE_ROOT}/models", exist_ok=True)

result = subprocess.run([
    "python", "training/train_cm.py",
    "--data",      f"{DRIVE_ROOT}/asvspoof2019_prepared/train",
    "--val-split", "0.1",
    "--epochs",    "50",
    "--lr",        "0.0003",
    "--batch",     "32",
    "--out",       OUT_MODEL,
])
if result.returncode != 0:
    print("ERROR: Training failed. Check output above.")
    sys.exit(1)

size_mb = os.path.getsize(OUT_MODEL) / 1e6
print(f"✓ Training done! Checkpoint: {OUT_MODEL} ({size_mb:.1f} MB)")

# ── Step 6: Evaluate EER ──────────────────────────────────────────────────────
print("\n  Evaluating EER on eval partition...")
os.makedirs("results", exist_ok=True)
subprocess.run([
    "python", "training/eval_eer.py",
    "--data",       f"{DRIVE_ROOT}/asvspoof2019_prepared/eval",
    "--checkpoint", OUT_MODEL,
    "--out",        "results/benchmark_asvspoof19.json",
    "--dataset",    "asvspoof2019_la_eval",
])
if os.path.exists("results/benchmark_asvspoof19.json"):
    with open("results/benchmark_asvspoof19.json") as f:
        res = json.load(f)
    print(f"  EER: {res.get('eer')}%  |  min-tDCF: {res.get('min_tDCF')}")

# ── Step 7: Push to Hugging Face Hub ─────────────────────────────────────────
print(f"\n[7/7] Pushing checkpoint to Hugging Face Hub...")
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj=OUT_MODEL,
    path_in_repo="cm.pt",
    repo_id=HF_REPO_ID,
    token=HF_TOKEN,
    commit_message="VoiceGuard CM checkpoint — ASVspoof 2019 LA, 50 epochs",
)

# Also upload benchmark results
if os.path.exists("results/benchmark_asvspoof19.json"):
    api.upload_file(
        path_or_fileobj="results/benchmark_asvspoof19.json",
        path_in_repo="benchmark_asvspoof19.json",
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        commit_message="Add EER benchmark results",
    )

print(f"\n{'='*60}")
print(f"  ALL DONE!")
print(f"  Checkpoint: https://huggingface.co/{HF_REPO_ID}")
print(f"{'='*60}")
print()
print("To use locally:")
print(f"  python data/download_datasets.py \\")
print(f"    --hf-pull {HF_REPO_ID} \\")
print(f"    --checkpoint models/cm.pt")
