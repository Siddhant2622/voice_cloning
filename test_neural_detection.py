"""
test_neural_detection.py
Verify detection performance on modern Neural TTS (Gemini Live / ElevenLabs / Azure Neural)
vs Real Human Speech recordings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.features import extract_all
from src.models.cm_classifier import CMScorerWrapper
from src.models.liveness import LivenessScorer
from src.fusion import FusionModel, ScoreBundle
from src.risk_engine import PolicyEngine, RiskContext
from src.preprocessing import preprocess_file

device = "cuda" if torch.cuda.is_available() else "cpu"
cm = CMScorerWrapper("models/cm.pt", device=device)
lv = LivenessScorer()
fm = FusionModel()
policy = PolicyEngine("config/policy.yaml")

test_files = [
    ("data/samples/neural_tts_00.wav", "SPOOF (Neural Female)"),
    ("data/samples/neural_tts_01.wav", "SPOOF (Neural Male)"),
    ("data/samples/neural_tts_06.wav", "SPOOF (Neural Indian Male)"),
    ("data/samples/human_librispeech_00.wav", "BONAFIDE (Human Reader 1)"),
    ("data/samples/human_librispeech_01.wav", "BONAFIDE (Human Reader 2)"),
    ("data/samples/human_librispeech_02.wav", "BONAFIDE (Human Reader 3)"),
    ("data/samples/synthetic_00.wav", "SPOOF (Legacy TTS)"),
]

print("=" * 78)
print(f"{'File':32s}  {'Type':24s}  {'CM':6s}  {'Live':6s}  {'Fusion':6s}  {'Decision'}")
print("-" * 78)

for path, desc in test_files:
    wf, _ = preprocess_file(path, apply_vad_flag=False)
    feats = extract_all(wf, device=device)
    cm_score = cm.score(feats)
    lv_res = lv.score(wf)
    lv_score = lv_res.get("liveness_score", 0.5)

    bundle = ScoreBundle(cm_score=cm_score, liveness_score=lv_score)
    fusion = fm.score(bundle)
    ctx = RiskContext(fusion_score=fusion, session_id="test")
    dec = policy.decide(ctx)

    print(f"{Path(path).name:32s}  {desc:24s}  {cm_score:.3f}  {lv_score:.3f}  {fusion:.3f}  {dec.action.value}")

print("=" * 78)
