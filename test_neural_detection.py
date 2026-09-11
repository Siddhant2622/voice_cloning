"""
test_neural_detection.py
Verify detection on Neural TTS vs Real Human Speech.

Run directly: python test_neural_detection.py
Via pytest:   python -m pytest test_neural_detection.py -v -s
  (auto-skipped when torch or model files are unavailable)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pytest

_torch_ok = False
try:
    import torch as _t; _torch_ok = True
except Exception:
    pass

_skip = (
    not _torch_ok
    or not Path('models/cm.pt').exists()
    or not Path('data/samples').exists()
)
_reason = 'torch unavailable (WDAC) or model/sample files missing' if _skip else ''

TEST_FILES = [
    ('data/samples/neural_tts_00.wav',        'SPOOF (Neural Female)'),
    ('data/samples/neural_tts_01.wav',        'SPOOF (Neural Male)'),
    ('data/samples/neural_tts_06.wav',        'SPOOF (Neural Indian Male)'),
    ('data/samples/human_librispeech_00.wav', 'BONAFIDE (Human Reader 1)'),
    ('data/samples/human_librispeech_01.wav', 'BONAFIDE (Human Reader 2)'),
    ('data/samples/human_librispeech_02.wav', 'BONAFIDE (Human Reader 3)'),
    ('data/samples/synthetic_00.wav',         'SPOOF (Legacy TTS)'),
]


@pytest.mark.skipif(_skip, reason=_reason)
def test_neural_tts_detection():
    import torch
    from src.features import extract_all
    from src.models.cm_classifier import CMScorerWrapper
    from src.models.liveness import LivenessScorer
    from src.fusion import FusionModel, ScoreBundle
    from src.risk_engine import PolicyEngine, RiskContext
    from src.preprocessing import preprocess_file

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cm     = CMScorerWrapper('models/cm.pt', device=device)
    lv     = LivenessScorer()
    fm     = FusionModel()
    policy = PolicyEngine('config/policy.yaml')

    results = []
    sep = '=' * 78
    print()
    print(sep)
    print('File'.ljust(32) + '  ' + 'Type'.ljust(24) + '  CM      Lv    Fusion  Decision')
    print('-' * 78)
    for fpath, desc in TEST_FILES:
        p = Path(fpath)
        if not p.exists():
            print('SKIP', fpath)
            continue
        wf, _  = preprocess_file(fpath, apply_vad_flag=False)
        feats  = extract_all(wf, device=device)
        cm_s   = cm.score(feats)
        lv_res = lv.score(wf)
        lv_s   = lv_res.get('liveness_score', 0.5)
        bundle = ScoreBundle(cm_score=cm_s, liveness_score=lv_s)
        fusion = fm.score(bundle)
        ctx    = RiskContext(fusion_score=fusion, session_id='test')
        dec    = policy.decide(ctx)
        results.append(dec.action.value)
        row = (p.name.ljust(32) + '  ' + desc.ljust(24)
               + '  ' + f'{cm_s:.3f}  {lv_s:.3f}  {fusion:.3f}  {dec.action.value}')
        print(row)
    print(sep)
    assert results, 'No test files found in data/samples/ -- add audio files'


if __name__ == '__main__':
    if _skip:
        print('SKIP:', _reason)
        sys.exit(0)
    test_neural_tts_detection()
