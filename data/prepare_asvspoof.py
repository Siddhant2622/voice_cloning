"""
data/prepare_asvspoof.py
========================
Convert ASVspoof 2019/2021 LA dataset into the samples_metadata.csv format
that training/train_cm.py consumes.

Usage (after downloading LA.zip):
    python data/prepare_asvspoof.py --zip path/to/LA.zip --out data/asvspoof2019_la
    python data/prepare_asvspoof.py --dir path/to/LA     --out data/asvspoof2019_la
    python data/prepare_asvspoof.py --dir path/to/LA     --out data/asvspoof2019_la --split train
    python data/prepare_asvspoof.py --dir path/to/LA     --out data/asvspoof2019_la --split eval

Output structure:
    data/asvspoof2019_la/
        train/
            samples_metadata.csv   (file, label)
            flac/                  (symlinked or copied)
        dev/
            samples_metadata.csv
            flac/
        eval/
            samples_metadata.csv
            flac/

Protocol file format (ASVspoof 2019 LA):
    SPEAKER_ID  FILE_ID  -  ATTACK_ID  LABEL
    LA_0069     LA_E_5932896  -  -  bonafide
    LA_0069     LA_E_5932898  -  A09  spoof
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger("prepare_asvspoof")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Protocol file locations within the LA archive
PROTOCOL_PATHS = {
    "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev":   "ASVspoof2019_LA_dev/flac",
    "eval":  "ASVspoof2019_LA_eval/flac",
}


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    logger.info("Extracting %s → %s", zip_path.name, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Show progress
        names = zf.namelist()
        for i, name in enumerate(names):
            zf.extract(name, out_dir)
            if (i + 1) % 1000 == 0:
                logger.info("  Extracted %d / %d files", i + 1, len(names))
    logger.info("Extraction complete.")
    return out_dir


def parse_protocol(protocol_file: Path) -> list[tuple[str, str]]:
    """
    Parse an ASVspoof protocol file.
    Returns list of (filename_stem, label) where label = 'bonafide' | 'spoof'
    """
    records = []
    with open(protocol_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Format: SPEAKER FILE - ATTACK LABEL
            # parts[1] = file ID, parts[4] = label
            if len(parts) >= 5:
                file_id = parts[1]
                label   = parts[4]  # 'bonafide' or 'spoof'
                records.append((file_id, label))
    logger.info("Parsed %d records from %s", len(records), protocol_file.name)
    return records


def build_metadata_csv(
    records: list[tuple[str, str]],
    audio_dir: Path,
    out_dir: Path,
    copy_audio: bool = False,
) -> Path:
    """
    Build samples_metadata.csv in out_dir, optionally copying audio files.
    Returns path to the CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "samples_metadata.csv"

    found, missing = 0, 0
    rows = []
    for file_id, label in records:
        # ASVspoof 2019 LA uses .flac
        src = audio_dir / f"{file_id}.flac"
        if not src.exists():
            missing += 1
            continue
        found += 1

        if copy_audio:
            dst = out_dir / f"{file_id}.flac"
            if not dst.exists():
                shutil.copy2(src, dst)
            rows.append((f"{file_id}.flac", label))
        else:
            # Use absolute path so train_cm.py can find it without copying
            rows.append((str(src.resolve()), label))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label"])
        writer.writerows(rows)

    logger.info(
        "CSV written: %s  (%d found, %d missing)",
        csv_path, found, missing
    )
    if missing > 0:
        logger.warning("%d audio files not found — check the archive extraction.", missing)
    return csv_path


def prepare_split(la_root: Path, split: str, out_root: Path, copy_audio: bool = False) -> None:
    proto_rel  = PROTOCOL_PATHS[split]
    audio_rel  = AUDIO_DIRS[split]
    proto_file = la_root / proto_rel
    audio_dir  = la_root / audio_rel
    out_dir    = out_root / split

    if not proto_file.exists():
        logger.error("Protocol file not found: %s", proto_file)
        logger.error("Expected ASVspoof 2019 LA archive structure. Check extraction.")
        return

    if not audio_dir.exists():
        logger.error("Audio directory not found: %s", audio_dir)
        return

    records = parse_protocol(proto_file)
    build_metadata_csv(records, audio_dir, out_dir, copy_audio=copy_audio)

    # Summary
    bonafide = sum(1 for _, l in records if l == "bonafide")
    spoof    = sum(1 for _, l in records if l == "spoof")
    logger.info(
        "Split '%s': %d bonafide | %d spoof | %d total",
        split, bonafide, spoof, len(records)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASVspoof 2019 LA dataset.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", help="Path to downloaded LA.zip archive.")
    group.add_argument("--dir", help="Path to already-extracted LA directory.")
    parser.add_argument("--out",   default="data/asvspoof2019_la", help="Output directory.")
    parser.add_argument("--split", choices=["train", "dev", "eval", "all"], default="all",
                        help="Which split(s) to prepare.")
    parser.add_argument("--copy-audio", action="store_true",
                        help="Copy audio files into --out (uses more disk but self-contained).")
    args = parser.parse_args()

    out_root = Path(args.out)

    # Extract if zip given
    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            logger.error("ZIP not found: %s", zip_path)
            sys.exit(1)
        la_root = out_root / "_archive"
        extract_zip(zip_path, la_root)
    else:
        la_root = Path(args.dir)
        if not la_root.exists():
            logger.error("Directory not found: %s", la_root)
            sys.exit(1)

    splits = ["train", "dev", "eval"] if args.split == "all" else [args.split]
    for split in splits:
        logger.info("=== Preparing split: %s ===", split)
        prepare_split(la_root, split, out_root, copy_audio=args.copy_audio)

    logger.info("\nAll done. Train with:")
    logger.info("  python training/train_cm.py --data %s/train --out models/cm.pt", out_root)


if __name__ == "__main__":
    main()
