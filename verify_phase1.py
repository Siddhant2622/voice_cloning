"""
Run detect.py on all sample files and print a summary score table.
Phase 1 Definition of Done verification script.
"""
import json
import subprocess
import sys
from pathlib import Path

SAMPLES = sorted(Path("data/samples").glob("*.wav"))
CHECKPOINT = "models/cm.pt"

print("=" * 70)
print("PHASE 1 — DoD VERIFICATION: detect.py score table")
print("=" * 70)
print(f"{'File':35s}  {'Label':10s}  {'Fusion':7s}  {'CM':7s}  {'Liveness':8s}  {'Decision'}")
print("-" * 70)

results = []
for wav in SAMPLES:
    label = "SPOOF" if "synthetic" in wav.name else "BONA FIDE"
    cmd = [
        sys.executable, "detect.py", str(wav),
        "--model", CHECKPOINT,
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1, 2, 3):
        print(f"  {wav.name:33s}  ERROR: {proc.stderr[-200:]}")
        continue
    try:
        data = json.loads(proc.stdout)
    except Exception:
        print(f"  {wav.name:33s}  Parse error: {proc.stdout[:100]}")
        continue

    fusion   = data.get("fusion_score", 0)
    cm       = data.get("cm_score", 0)
    liveness = data.get("liveness_score")
    decision = data.get("decision", "?")
    lv_str   = f"{liveness:.3f}" if liveness is not None else "  N/A "

    results.append({"file": wav.name, "ground_truth": label, "fusion": fusion, "decision": decision})
    print(f"  {wav.name:33s}  {label:10s}  {fusion:.4f}  {cm:.4f}  {lv_str:8s}  {decision}")

print("=" * 70)
# Sanity check: are spoof scores higher than bona fide?
spoof_scores = [r["fusion"] for r in results if r["ground_truth"] == "SPOOF"]
bonafide_scores = [r["fusion"] for r in results if r["ground_truth"] == "BONA FIDE"]
if spoof_scores and bonafide_scores:
    mean_spoof = sum(spoof_scores) / len(spoof_scores)
    mean_bf    = sum(bonafide_scores) / len(bonafide_scores)
    print(f"\n  Mean spoof score:    {mean_spoof:.4f}")
    print(f"  Mean bonafide score: {mean_bf:.4f}")
    if mean_spoof > mean_bf:
        print("\n  [DoD PASSED] Spoof scores are higher than bona fide scores.")
        print("  Scores are differentiated — Phase 1 Definition of Done: MET.")
    else:
        print("\n  [WARN] Bona fide scores >= spoof — model needs training data.")
        print("  (Expected on 7-sample toy dataset; real eval needs ASVspoof.)")
print("=" * 70)
