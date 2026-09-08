#!/usr/bin/env python3
"""
VoiceGuard — Google Colab Training Script
==========================================
USAGE:
  Option A: Upload this file to Colab → Runtime → Run all
  Option B: Copy each SECTION into a separate Colab cell

BEFORE RUNNING: Edit the 3 config lines in SECTION 2 below.

PREREQUISITES:
  1. Google Colab with T4 GPU: Runtime → Change runtime type → T4 GPU
  2. Hugging Face account + Write token: huggingface.co → Settings → Access Tokens
  3. Your GitHub repo URL (push your code first with: git push)
"""

# ─── SECTION 1: Install dependencies ─────────────────────────────────────────
# Paste this into Colab Cell 1 and run it

CELL_1 = """
!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers librosa soundfile scipy scikit-learn
!pip install -q huggingface_hub rich tqdm
print("✓ Dependencies installed")
"""

# ─── SECTION 2: Configuration — EDIT THESE 3 LINES ──────────────────────────
# Paste this into Colab Cell 2 and run it

CELL_2 = """
# ╔══════════════════════════════════════════════════════════╗
# ║   EDIT THESE 3 LINES BEFORE RUNNING ANYTHING ELSE       ║
# ╚══════════════════════════════════════════════════════════╝
GITHUB_REPO  = "https://github.com/Siddhant2622/Voice-Cloning"
HF_REPO_ID   = "C00EX2622/voiceguard-cm"
HF_TOKEN     = "hf_ENTER_IN_COLAB_NOT_HERE"  # ← Type your token ONLY in Colab, never save to file
# ─────────────────────────────────────────────────────────────

import os
from google.colab import drive

drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/voiceguard"
os.makedirs(DRIVE_ROOT, exist_ok=True)

os.chdir("/content")
if not os.path.exists("voiceguard"):
    os.system(f"git clone {GITHUB_REPO} voiceguard")

os.chdir("voiceguard")
os.system("pip install -q -r requirements.txt")

# Save config for later cells
import json
REPO_DIR = "/content/voiceguard"
cfg = {"GITHUB_REPO": GITHUB_REPO, "HF_REPO_ID": HF_REPO_ID,
       "HF_TOKEN": HF_TOKEN, "DRIVE_ROOT": DRIVE_ROOT,
       "REPO_DIR": REPO_DIR}
with open("/content/cfg.json", "w") as f:
    json.dump(cfg, f)

print(f"✓ Repo cloned and config saved")
print(f"✓ Drive root: {DRIVE_ROOT}")
"""

# ─── SECTION 3: Download ASVspoof 2019 LA ─────────────────────────────────────
# Paste this into Colab Cell 3 and run it
# This downloads ~2.6 GB — saved to Drive so it survives reconnects

CELL_3 = """
import os, zipfile, urllib.request, json

with open("/content/cfg.json") as f:
    cfg = json.load(f)

DRIVE_ROOT = cfg["DRIVE_ROOT"]
DATA_DIR   = f"{DRIVE_ROOT}/asvspoof2019_la"
os.makedirs(DATA_DIR, exist_ok=True)

ZENODO = "https://zenodo.org/record/5105682/files/"
FILES  = [
    "ASVspoof2019_LA_cm_protocols.zip",
    "ASVspoof2019_LA_train.zip",
    "ASVspoof2019_LA_dev.zip",
    "ASVspoof2019_LA_eval.zip",
]

def progress(block, bsize, total):
    done = block * bsize
    if total > 0:
        pct = min(done / total * 100, 100)
        print(f"  {pct:5.1f}%  {done/1e6:.0f}/{total/1e6:.0f} MB", end="\\r")

for fname in FILES:
    dest = f"{DATA_DIR}/{fname}"
    if os.path.exists(dest):
        print(f"✓ {fname} (already downloaded)")
        continue
    print(f"Downloading {fname}...")
    urllib.request.urlretrieve(f"{ZENODO}{fname}", dest, reporthook=progress)
    print(f"\\n✓ {fname} downloaded")

print("\\nExtracting archives...")
for fname in FILES:
    path = f"{DATA_DIR}/{fname}"
    if os.path.exists(path):
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(DATA_DIR)
        print(f"✓ Extracted {fname}")

print("\\n✓ ASVspoof 2019 LA dataset ready!")
"""

# ─── SECTION 4: Prepare dataset ───────────────────────────────────────────────
# Paste this into Colab Cell 4 and run it

CELL_4 = """
import subprocess, json

with open("/content/cfg.json") as f:
    cfg = json.load(f)

DRIVE_ROOT = cfg["DRIVE_ROOT"]
os.chdir(cfg["REPO_DIR"])

result = subprocess.run([
    "python", "data/prepare_asvspoof.py",
    "--dir",   f"{DRIVE_ROOT}/asvspoof2019_la",
    "--out",   f"{DRIVE_ROOT}/asvspoof2019_prepared",
    "--split", "all",
])

if result.returncode == 0:
    print("\\n✓ Dataset prepared! Files in:")
    print(f"  {DRIVE_ROOT}/asvspoof2019_prepared/train/samples_metadata.csv")
    print(f"  {DRIVE_ROOT}/asvspoof2019_prepared/dev/samples_metadata.csv")
    print(f"  {DRIVE_ROOT}/asvspoof2019_prepared/eval/samples_metadata.csv")
else:
    print("ERROR during preparation — check output above")
"""

# ─── SECTION 5: Train the CM classifier ───────────────────────────────────────
# Paste this into Colab Cell 5 and run it
# Takes ~2-3 hours on T4 GPU

CELL_5 = """
import subprocess, json, os

with open("/content/cfg.json") as f:
    cfg = json.load(f)

DRIVE_ROOT = cfg["DRIVE_ROOT"]
OUT_MODEL  = f"{DRIVE_ROOT}/models/cm_asvspoof19.pt"
os.makedirs(f"{DRIVE_ROOT}/models", exist_ok=True)
os.chdir(cfg["REPO_DIR"])

print("Starting training... (this takes ~2-3 hours on T4 GPU)")
print("You can close this tab — Drive saves progress automatically")
print()

result = subprocess.run([
    "python", "training/train_cm.py",
    "--data",      f"{DRIVE_ROOT}/asvspoof2019_prepared/train",
    "--val-split", "0.1",
    "--epochs",    "50",
    "--lr",        "0.0003",
    "--batch",     "32",
    "--out",       OUT_MODEL,
])

if result.returncode == 0:
    size_mb = os.path.getsize(OUT_MODEL) / 1e6
    print(f"\\n✓ Training complete!")
    print(f"  Checkpoint: {OUT_MODEL}  ({size_mb:.1f} MB)")
else:
    print("ERROR during training — check output above")
"""

# ─── SECTION 6: Push to Hugging Face Hub ─────────────────────────────────────
# Paste this into Colab Cell 6 and run it

CELL_6 = """
import json, os
from huggingface_hub import HfApi

with open("/content/cfg.json") as f:
    cfg = json.load(f)

DRIVE_ROOT = cfg["DRIVE_ROOT"]
HF_REPO_ID = cfg["HF_REPO_ID"]
HF_TOKEN   = cfg["HF_TOKEN"]
OUT_MODEL  = f"{DRIVE_ROOT}/models/cm_asvspoof19.pt"

print(f"Pushing to https://huggingface.co/{HF_REPO_ID} ...")

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True,
                repo_type="model")
api.upload_file(
    path_or_fileobj=OUT_MODEL,
    path_in_repo="cm.pt",
    repo_id=HF_REPO_ID,
    token=HF_TOKEN,
    commit_message="Add trained VoiceGuard CM checkpoint (ASVspoof 2019 LA, 50 epochs)",
)

print(f"\\n✓ Checkpoint live at:")
print(f"  https://huggingface.co/{HF_REPO_ID}")
print()
print("To use it locally:")
print(f"  python data/download_datasets.py --hf-pull {HF_REPO_ID} --checkpoint models/cm.pt")
"""

# ─── SECTION 7: Evaluate EER ──────────────────────────────────────────────────
# Paste this into Colab Cell 7 and run it

CELL_7 = """
import subprocess, json, os

with open("/content/cfg.json") as f:
    cfg = json.load(f)

DRIVE_ROOT = cfg["DRIVE_ROOT"]
os.chdir("/content/voiceguard")
os.makedirs("results", exist_ok=True)

print("Running EER evaluation on ASVspoof 2019 LA eval partition...")
print("(This takes 20-40 minutes)")

result = subprocess.run([
    "python", "training/eval_eer.py",
    "--data",       f"{DRIVE_ROOT}/asvspoof2019_prepared/eval",
    "--checkpoint", f"{DRIVE_ROOT}/models/cm_asvspoof19.pt",
    "--out",        "results/benchmark_asvspoof19.json",
    "--dataset",    "asvspoof2019_la_eval",
])

if result.returncode == 0:
    with open("results/benchmark_asvspoof19.json") as f:
        results = json.load(f)
    print(f"\\n{'='*50}")
    print(f"  EER:      {results['eer']}%")
    print(f"  min-tDCF: {results['min_tDCF']}")
    print(f"  Dataset:  {results['n_bonafide']} bonafide | {results['n_spoof']} spoof")
    print(f"{'='*50}")

    # Commit results to GitHub
    os.system('git config user.email "colab@voiceguard.ai"')
    os.system('git config user.name "VoiceGuard Colab"')
    os.system("git add results/benchmark_asvspoof19.json")
    os.system('git commit -m "ci: add ASVspoof2019 LA eval results from Colab"')
    os.system("git push")
    print("\\n✓ Results committed and pushed to GitHub!")
    print("✓ Open the Benchmark Explorer tab to see your results live!")
"""


# ─── Standalone runner ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           VoiceGuard — Google Colab Training Instructions               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  STEP 1: Open https://colab.research.google.com → New Notebook          ║
║  STEP 2: Runtime → Change runtime type → T4 GPU → Save                 ║
║  STEP 3: Create 7 cells and paste each CELL_N section from this file   ║
║  STEP 4: In CELL 2, replace:                                            ║
║          YOUR_USERNAME  → your GitHub username                          ║
║          YOUR_HF_USERNAME → your HuggingFace username                  ║
║          hf_PASTE_YOUR_TOKEN_HERE → your HF write token                ║
║  STEP 5: Run cells one by one (Ctrl+Enter)                              ║
║                                                                          ║
║  Total time: ~3-4 hours on T4 GPU (mostly Cell 5 training)             ║
║  Result: checkpoint at https://huggingface.co/YOUR_HF_USERNAME/...     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Each CELL variable above contains the code to paste into that Colab cell.
""")

    for i, cell_code in enumerate([CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7], 1):
        print(f"{'─'*70}")
        print(f"CELL {i}:")
        print(cell_code.strip())
        print()
