"""
data/prepare_asvspoof2017.py
============================
Convert ASVspoof 2017 V2 Replay/Microphone dataset into normalized
samples_metadata.csv format for VoiceGuard CM training and evaluation.

ASVspoof 2017 V2 Protocol Format (7 columns):
    1: File ID (e.g. T_1000001.wav)
    2: Speech Type: 'genuine' (bonafide) or 'spoof' (replay)
    3: Speaker ID (e.g. M0002)
    4: Phrase ID (e.g. S05)
    5: Recording Environment ID (E01-E26, or '-' for genuine)
    6: Playback Device ID (P01-P26, or '-' for genuine)
    7: Recording Microphone ID (R01-R25, or '-' for genuine)

Usage:
    python data/prepare_asvspoof2017.py --split train
    python data/prepare_asvspoof2017.py --split dev
    python data/prepare_asvspoof2017.py --split all
    python data/prepare_asvspoof2017.py --root data/asvspoof2017 --out data/asvspoof2017
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("prepare_asvspoof2017")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Protocol files relative to root
PROTOCOLS = {
    "train": "protocol_V2/ASVspoof2017_V2_train.trn.txt",
    "dev":   "protocol_V2/ASVspoof2017_V2_dev.trl.txt",
    "eval":  "protocol_V2/ASVspoof2017_V2_eval.trl.txt",
}

AUDIO_DIRS = {
    "train": ["ASVspoof2017_V2_train", "train", "ASVspoof2017_train"],
    "dev":   ["ASVspoof2017_V2_dev", "dev", "ASVspoof2017_dev"],
    "eval":  ["ASVspoof2017_V2_eval", "eval", "ASVspoof2017_eval"],
}

# Environment descriptions from Instructions_V2.txt
ENV_DESCRIPTIONS = {
    "E01": "Anechoic room", "E02": "Balcony 01", "E03": "Balcony 02",
    "E04": "Home 07", "E05": "Home 08", "E06": "Cantine",
    "E07": "Home 01", "E08": "Home 02", "E09": "Home 03",
    "E10": "Home 04", "E11": "Home 05", "E12": "Home 06",
    "E13": "Office 01", "E14": "Office 02", "E15": "Office 03",
    "E16": "Office 04", "E17": "Office 05", "E18": "Office 06",
    "E19": "Office 07", "E20": "Office 08", "E21": "Office 09",
    "E22": "Office 10", "E23": "Studio", "E24": "Analog wire 01",
    "E25": "Analog wire 02", "E26": "Analog wire 03"
}

# Microphone descriptions from Instructions_V2.txt
MIC_DESCRIPTIONS = {
    "R01": "Zoom H6 handy recorder",
    "R02": "BQ Aquaris M5 smartphone",
    "R03": "Low-quality headset",
    "R04": "Nokia Lumia 635 smartphone",
    "R05": "Rode NT2 microphone",
    "R06": "Rode smartLav+ microphone",
    "R07": "Samsung Galaxy S7 smartphone",
    "R08": "Desktop PC microphone input",
    "R09": "Zoom H6 with Behringer ECM8000 mic",
    "R10": "Zoom H6 with MSH-6 mic",
    "R11": "Zoom H6 with XY mic",
    "R12": "iPhone 5c smartphone",
    "R13": "iPhone 7 plus smartphone",
    "R14": "iPhone 4 smartphone",
    "R15": "Logitech C920 webcam mic",
    "R16": "miniDSP UMIK-1 microphone",
    "R17": "Samsung Galaxy Trend 2 smartphone",
    "R18": "Samsung GT-I9100 smartphone",
    "R19": "Samsung GT-P6200 tablet",
    "R20": "Samsung Trend 2 smartphone",
    "R21": "AKG C3000 microphone",
    "R22": "SE electronic 2200a microphone",
    "R23": "Focusrite Scarlett 2i2 interface line input",
    "R24": "Focusrite Scarlett 2i4 interface line input",
    "R25": "Zoom HD1 handy recorder"
}

# Playback device descriptions
PLAYBACK_DESCRIPTIONS = {
    "P01": "All-in-one PC speakers", "P02": "Creative A60 speakers",
    "P03": "Genelec 8020C studio monitor", "P04": "Genelec 8020C monitor (2 spk)",
    "P05": "Beyerdynamic DT 770 PRO headphones", "P06": "Dell laptop internal speakers",
    "P07": "Dynaudio BM5A speaker", "P08": "HP Laptop internal speakers",
    "P09": "VIFA M10MD-39-08 speaker", "P10": "ACER netbook internal speakers",
    "P11": "BQ Aquaris M5 smartphone", "P12": "Logitech low quality speakers",
    "P13": "Desktop PC line output", "P14": "Labtec LCS-1050 speakers",
    "P15": "Edirol MA-15D studio monitor", "P16": "Lenovo Ideatab S6000-H tablet",
    "P17": "Logitech S120 multimedia speakers", "P18": "MacBook pro internal speakers",
    "P19": "Altec Lansing Orbit USB portable speaker", "P20": "Samsung GT-I9100 smartphone",
    "P21": "Samsung GT-P6200 tablet", "P22": "Behringer Truth B2030A studio monitor",
    "P23": "Focusrite Scarlett 2i2 line output", "P24": "Focusrite Scarlett 2i4 line output",
    "P25": "Genelec 6010A studio monitor", "P26": "AKG K242HD Headset"
}


def parse_asvspoof2017_protocol(proto_path: Path) -> List[Dict[str, str]]:
    """Parse ASVspoof 2017 V2 protocol file."""
    records = []
    if not proto_path.exists():
        logger.error("Protocol file not found: %s", proto_path)
        return records

    with open(proto_path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            file_id = parts[0]
            if not file_id.endswith(".wav"):
                file_id = f"{file_id}.wav"

            raw_label = parts[1].lower()
            label = "bonafide" if raw_label in ("genuine", "bonafide") else "spoof"

            speaker = parts[2] if len(parts) > 2 else ""
            phrase = parts[3] if len(parts) > 3 else ""
            env = parts[4] if len(parts) > 4 else "-"
            playback = parts[5] if len(parts) > 5 else "-"
            mic = parts[6] if len(parts) > 6 else "-"

            records.append({
                "file_id": file_id,
                "label": label,
                "speaker": speaker,
                "phrase": phrase,
                "environment": env,
                "playback_device": playback,
                "recording_mic": mic,
                "source": f"asvspoof2017_{label}" if label == "bonafide" else f"asvspoof2017_replay_{mic}_{playback}"
            })

    logger.info("Parsed %d records from %s", len(records), proto_path.name)
    return records


def find_audio_dir(root: Path, split: str) -> Optional[Path]:
    """Find the directory containing audio for this split."""
    candidates = AUDIO_DIRS.get(split, [])
    for c in candidates:
        d = root / c
        if d.exists() and d.is_dir():
            # Verify it has audio files
            if any(d.glob("*.wav")):
                return d
    return None


def prepare_split(root: Path, split: str, out_dir: Path) -> Optional[Path]:
    """
    Prepare metadata CSV and organize split for ASVspoof 2017.
    """
    proto_rel = PROTOCOLS[split]
    proto_path = root / proto_rel

    if not proto_path.exists():
        # Check alternative locations
        alt_proto = root / f"ASVspoof2017_V2_{split}.trn.txt" if split == "train" else root / f"ASVspoof2017_V2_{split}.trl.txt"
        if alt_proto.exists():
            proto_path = alt_proto
        else:
            logger.error("Protocol for split '%s' not found at %s", split, proto_path)
            return None

    audio_dir = find_audio_dir(root, split)
    if not audio_dir:
        logger.error("Audio directory for split '%s' not found in %s", split, root)
        return None

    records = parse_asvspoof2017_protocol(proto_path)
    if not records:
        return None

    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    csv_path = split_out / "samples_metadata.csv"
    matched_rows = []
    missing_count = 0

    for rec in records:
        f_name = rec["file_id"]
        src_path = audio_dir / f_name
        if src_path.exists():
            matched_rows.append({
                "file": str(src_path.resolve()),
                "filename": f_name,
                "label": rec["label"],
                "speaker": rec["speaker"],
                "phrase": rec["phrase"],
                "environment": rec["environment"],
                "playback_device": rec["playback_device"],
                "recording_mic": rec["recording_mic"],
                "source": rec["source"],
            })
        else:
            missing_count += 1

    if missing_count > 0:
        logger.warning("Split '%s': %d files missing out of %d in audio dir %s",
                       split, missing_count, len(records), audio_dir)

    fieldnames = ["file", "filename", "label", "speaker", "phrase",
                  "environment", "playback_device", "recording_mic", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_rows)

    n_bf = sum(1 for r in matched_rows if r["label"] == "bonafide")
    n_sp = sum(1 for r in matched_rows if r["label"] == "spoof")
    logger.info("Wrote %s: %d total (%d bonafide, %d replayed spoof)",
                csv_path, len(matched_rows), n_bf, n_sp)

    # Breakdown by recording microphone
    mics = {}
    for r in matched_rows:
        m = r["recording_mic"]
        mics[m] = mics.get(m, 0) + 1
    logger.info("Microphone distribution (%d distinct): %s", len(mics), dict(sorted(mics.items())[:8]))

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Prepare ASVspoof 2017 V2 dataset")
    parser.add_argument("--root", default="data/asvspoof2017", help="ASVspoof 2017 root directory")
    parser.add_argument("--out", default="data/asvspoof2017", help="Output metadata directory")
    parser.add_argument("--split", choices=["train", "dev", "eval", "all"], default="all")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    splits = ["train", "dev", "eval"] if args.split == "all" else [args.split]
    for s in splits:
        logger.info("=== Preparing ASVspoof 2017 V2 split: %s ===", s)
        prepare_split(root, s, out)


if __name__ == "__main__":
    main()
