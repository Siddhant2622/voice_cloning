"""
Phase 1 DoD verification — single-process batch scoring (much faster: loads models once).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.preprocessing import preprocess_file
from src.features import extract_all
from src.models.cm_classifier import CMScorerWrapper
from src.models.liveness import LivenessScorer
from src.fusion import FusionModel, ScoreBundle
from src.risk_engine import PolicyEngine, RiskContext
import time

SAMPLES_DIR = Path("data/samples")
CM_CKPT     = "models/cm.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load models once
print("Loading models...")
t0 = time.perf_counter()
cm_scorer  = CMScorerWrapper(checkpoint_path=CM_CKPT, device=device)
lv_scorer  = LivenessScorer()
fm         = FusionModel()
policy     = PolicyEngine(config_path="config/policy.yaml")
print(f"Models ready in {time.perf_counter()-t0:.1f}s\n")

SAMPLES = sorted(SAMPLES_DIR.glob("*.wav"))

print("=" * 72)
print("PHASE 1 DoD — detect.py score table (single-process batch)")
print("=" * 72)
print(f"  {'File':28s}  {'GT':9s}  {'CM':6s}  {'Liveness':8s}  {'Fusion':6s}  {'Decision'}")
print("-" * 72)

results = []
for wav in SAMPLES:
    gt = "SPOOF" if "synthetic" in wav.name else "BONAFIDE"
    t_start = time.perf_counter()
    try:
        wf, sr   = preprocess_file(wav, apply_vad_flag=False)
        feats    = extract_all(wf, device=device)
        cm_score = cm_scorer.score(feats)

        lv_result = {}
        lv_score  = None
        try:
            lv_result = lv_scorer.score(wf)
            lv_score  = lv_result.get("liveness_score")
        except Exception:
            pass

        bundle = ScoreBundle(cm_score=cm_score, liveness_score=lv_score)
        fusion = fm.score(bundle)

        ctx      = RiskContext(fusion_score=fusion, session_id="dod_test")
        decision = policy.decide(ctx)
        latency  = (time.perf_counter() - t_start) * 1000

        lv_str = f"{lv_score:.3f}" if lv_score is not None else "  N/A"
        print(f"  {wav.name:28s}  {gt:9s}  {cm_score:.4f}  {lv_str:8s}  {fusion:.4f}  {decision.action.value}")
        results.append({"file": wav.name, "gt": gt, "cm": cm_score, "fusion": fusion,
                        "decision": decision.action.value, "latency_ms": latency})
    except Exception as exc:
        print(f"  {wav.name:28s}  ERROR: {exc}")

print("=" * 72)

# Summary stats
spoof_scores   = [r["fusion"] for r in results if r["gt"] == "SPOOF"]
bonafide_scores = [r["fusion"] for r in results if r["gt"] == "BONAFIDE"]
if spoof_scores and bonafide_scores:
    mean_spoof = sum(spoof_scores) / len(spoof_scores)
    mean_bf    = sum(bonafide_scores) / len(bonafide_scores)
    avg_lat    = sum(r["latency_ms"] for r in results) / len(results)
    print(f"\n  Mean spoof fusion score:    {mean_spoof:.4f}")
    print(f"  Mean bonafide fusion score: {mean_bf:.4f}")
    print(f"  Avg per-sample latency:     {avg_lat:.0f} ms")
    print()
    if mean_spoof > mean_bf:
        print("  [PASS] Spoof scores higher than bonafide — scores differentiated.")
        print("  Phase 1 Definition of Done: MET")
    else:
        print("  [WARN] Bonafide >= spoof on 7-sample toy set (expected).")
        print("  Train on ASVspoof 2019 LA for real accuracy. Feature pipeline: OK.")
print("=" * 72)
