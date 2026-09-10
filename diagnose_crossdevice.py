"""diagnose_crossdevice.py — run from repo root: python diagnose_crossdevice.py"""
import sys, random, logging
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as signal
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(Path(".").resolve()))

def apply_mobile_replay_chain(wav, sr=16000):
    nyq = sr / 2.0
    b_hp, a_hp = signal.butter(2, random.uniform(350,500)/nyq, btype="high")
    b_lp, a_lp = signal.butter(4, random.uniform(3200,3900)/nyq, btype="low")
    out = signal.filtfilt(b_hp, a_hp, wav).astype(np.float32)
    out = signal.filtfilt(b_lp, a_lp, out).astype(np.float32)
    d = random.uniform(0.01,0.06)
    out = np.clip(out + d*(out**2) - d*0.5*(out**3), -1, 1).astype(np.float32)
    # simple room RIR
    n_rir = max(int(0.15*sr), 16)
    rir = np.zeros(n_rir, dtype=np.float32)
    rir[int(0.002*sr)] = 1.0
    rir /= (np.abs(rir).max()+1e-8)
    out = signal.fftconvolve(out, rir, mode="full")[:len(wav)].astype(np.float32)
    # MEMS mic
    b2, a2 = signal.butter(2, random.uniform(180,280)/nyq, btype="high")
    b3, a3 = signal.butter(3, random.uniform(4000,4800)/nyq, btype="low")
    out = signal.filtfilt(b2,a2,out).astype(np.float32)
    out = signal.filtfilt(b3,a3,out).astype(np.float32)
    # AGC
    frame_len = max(int(0.02*sr),1); agc = np.zeros_like(out); g=1.0
    for i in range(0,len(out),frame_len):
        f=out[i:i+frame_len]; rms=float(np.sqrt(np.mean(f**2)+1e-10))
        dg=float(np.clip(0.08/rms if rms>1e-6 else g,0.1,8.0))
        g = (0.1 if dg<g else 0.5)*dg + (0.9 if dg<g else 0.5)*g
        agc[i:i+frame_len]=f*g
    out=agc.astype(np.float32)
    # noise
    snr_db=random.uniform(14,28); sp=float(np.mean(out**2))+1e-10
    np_=float(sp/(10**(snr_db/10))); white=np.random.randn(len(out)).astype(np.float32)
    bn,an=signal.butter(1, min(2000/nyq,0.99), btype="low")
    tinted=signal.lfilter(bn,an,white).astype(np.float32)
    out += tinted*(float(np_)**0.5/(float(np.mean(tinted**2))**0.5+1e-10))
    out=np.nan_to_num(out,nan=0.,posinf=0.,neginf=0.).astype(np.float32)
    mx=np.abs(out).max()
    return (out/mx*0.90) if mx>1e-6 else out

import torch
from src.preprocessing import to_mono_16k
from src.features import extract_all
from src.models.cm_classifier import CMScorerWrapper
from src.models.liveness import LivenessScorer
from src.fusion import FusionModel, ScoreBundle

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a TTS spoof sample
spoof_dir = Path("data/modern_dataset/spoof")
candidates = list(spoof_dir.glob("*.wav"))[:5] if spoof_dir.exists() else []
if candidates:
    src = random.choice(candidates)
    print(f"[TTS source] {src.name}")
    data, sr = sf.read(str(src))
    wav_np = data.astype(np.float32)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
else:
    print("[fallback] 200Hz sine (no dataset spoof samples found)")
    sr = 16000
    t = np.linspace(0, 3.0, 48000)
    wav_np = (0.5*np.sin(2*np.pi*200*t)).astype(np.float32)

wf_clean = to_mono_16k(torch.from_numpy(wav_np).unsqueeze(0), sr)
np_clean = wf_clean.numpy().astype(np.float32)

random.seed(99); np.random.seed(99)
np_mobile = apply_mobile_replay_chain(np_clean.copy())
wf_mobile = torch.from_numpy(np_mobile)

cm_ckpt = next((c for c in ["models/cm_detect2b_v4.pt","models/cm_detect2b_v3.pt","models/cm.pt"] if Path(c).exists()), None)
print(f"[CM model] {cm_ckpt}\n")

cm = CMScorerWrapper(checkpoint_path=cm_ckpt, device=device)
lv = LivenessScorer()
fm = FusionModel()

SEP = "="*60
for label, wf, np_wav in [("CLEAN TTS (should=SPOOF)", wf_clean, np_clean),
                            ("MOBILE REPLAY TTS (should=SPOOF)", wf_mobile, np_mobile)]:
    print(SEP)
    print(f"  {label}")
    print(SEP)
    feats = extract_all(wf, device=device)
    cm_s = cm.score(feats)
    lv_r = lv.score(wf, sr=16000)
    lv_s = lv_r["liveness_score"]
    bundle = ScoreBundle(cm_score=cm_s, liveness_score=lv_s)
    fus = fm._heuristic_score(bundle)
    if fus >= 0.88: dec = "BLOCK"
    elif fus >= 0.72: dec = "ALERT"
    elif fus >= 0.50: dec = "STEP_UP"
    else: dec = "ALLOW  <-- MISSED!"

    n_fft = 512
    S = np.abs(np.fft.rfft(np_wav, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/16000)
    hf_r = (float(np.mean(S[freqs>4000]**2))+1e-10) / (float(np.mean(S[freqs<=4000]**2))+1e-10)

    print(f"  cm_score       = {cm_s:.4f}  {'OK' if cm_s>0.65 else '<<< LOW - CM not detecting'}")
    print(f"  liveness_score = {lv_s:.4f}  {'OK' if lv_s>0.25 else '<<< LOW - liveness not suspicious'}")
    print(f"    flatness     = {lv_r.get('flatness_score',0):.4f}")
    print(f"    jitter       = {lv_r.get('jitter_score',0):.4f}")
    print(f"    contrast     = {lv_r.get('contrast_score',0):.4f}")
    print(f"    bandwidth    = {lv_r.get('bandwidth_score',0):.4f}  (HF/LF ratio = {hf_r:.4f})")
    print(f"  fusion_score   = {fus:.4f}")
    print(f"  DECISION       = {dec}")
    print()
