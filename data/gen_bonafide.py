"""Generate 3 bona-fide TTS clips at faster speech rate to distinguish from synthetic."""
import csv
import pyttsx3
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

BF_SENTENCES = [
    ("The quick brown fox jumps over the lazy dog near the riverbank today.", 220),
    ("She sells sea shells by the seashore on a warm summer afternoon in July.", 200),
    ("How much wood would a woodchuck chuck if a woodchuck could chuck wood here.", 210),
]

new_records = []
for i, (sentence, rate) in enumerate(BF_SENTENCES):
    out = SAMPLES_DIR / f"bonafide_{i:02d}.wav"
    if not out.exists():
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(sentence, str(out))
        engine.runAndWait()
        print(f"  Generated {out.name}")
    else:
        print(f"  {out.name} already exists")
    new_records.append({"file": out.name, "label": "bonafide", "source": "pyttsx3_fast", "text": sentence})

# Update CSV
meta_path = SAMPLES_DIR / "samples_metadata.csv"
existing = []
if meta_path.exists():
    with open(meta_path, newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))

seen = {r["file"] for r in existing}
combined = existing + [r for r in new_records if r["file"] not in seen]

with open(meta_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "label", "source", "text"])
    writer.writeheader()
    writer.writerows(combined)

n_spoof   = sum(1 for r in combined if r["label"] == "spoof")
n_bonafide = sum(1 for r in combined if r["label"] == "bonafide")
print(f"[OK] Total records: {len(combined)}  ({n_spoof} spoof, {n_bonafide} bonafide)")
