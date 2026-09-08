import csv
from pathlib import Path
import soundfile as sf

csv_path = Path("data/samples/samples_metadata.csv")
with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total samples in metadata: {len(rows)}")
for r in rows:
    fpath = Path("data/samples") / r["file"]
    if fpath.exists():
        info = sf.info(str(fpath))
        print(f"{r['file']:26s} | label={r['label']:8s} | src={r['source'][:20]:20s} | {info.duration:.2f}s | {info.samplerate}Hz")
    else:
        print(f"{r['file']:26s} | MISSING")
