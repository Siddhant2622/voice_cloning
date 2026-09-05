"""Rebuild samples_metadata.csv from all .wav files present in data/samples/."""
import csv
from pathlib import Path

samples_dir = Path("data/samples")
rows = []
for f in sorted(samples_dir.glob("*.wav")):
    if f.name.startswith("bonafide"):
        rows.append({"file": f.name, "label": "bonafide", "source": "pyttsx3_fast", "text": ""})
    elif f.name.startswith("synthetic"):
        rows.append({"file": f.name, "label": "spoof", "source": "pyttsx3", "text": ""})

with open(samples_dir / "samples_metadata.csv", "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["file", "label", "source", "text"])
    writer.writeheader()
    writer.writerows(rows)

n_bf = sum(1 for r in rows if r["label"] == "bonafide")
n_sp = sum(1 for r in rows if r["label"] == "spoof")
print(f"CSV updated: {len(rows)} total  ({n_sp} spoof, {n_bf} bonafide)")
for r in rows:
    print(f"  {r['file']:30s}  [{r['label']}]")
