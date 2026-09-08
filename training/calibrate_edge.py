import json

with open("models/edge_thresholds.json") as f:
    data = json.load(f)

samples = data["sample_evaluations"]

def score_edge(s):
    flat  = s["spectral_flatness"]
    eVar  = s["energy_variance"]
    pitch = s["pitch_periodicity"]
    cent  = s["spectral_centroid"]
    zcr   = s["zcr"]
    hnr   = s["hnr_proxy"]
    hf    = s["hf_ratio"]

    # Natural biological voice safeguards:
    # 1. Very dynamic speech energy envelope across syllables
    if eVar >= 1.10:
        return 0.08
    # 2. Natural human pitch contour with low hiss and dynamic envelope
    if eVar >= 0.98 and pitch <= 0.40 and zcr <= 0.125:
        return 0.05

    score = 0.05

    # 1. Classical vocoder (pyttsx3): robotic periodic pitch + low energy dynamic range + high HNR
    if pitch > 0.70:
        score += 0.55 + (pitch - 0.70) * 1.8
    if hnr > 2.8:
        score += 0.25
    if eVar < 0.92:
        score += 0.15

    # 2. Modern Neural TTS (Jenny, Guy, Aria, etc.):
    # Characterized by elevated zero-crossings (neural vocoder noise), flatter higher-frequency content
    if zcr >= 0.125:
        score += 0.25 + (zcr - 0.125) * 5.0
    if cent > 0.23:
        score += (cent - 0.23) * 3.5
    if flat > 0.35 and zcr > 0.125:
        score += 0.18
    if pitch > 0.50 and zcr > 0.120:
        score += 0.20
    if hf > 0.032:
        score += (hf - 0.032) * 7.0

    # 3. Biological human voice safeguard
    if pitch < 0.55 and cent < 0.23 and zcr < 0.120:
        score = min(score, 0.12)
    if eVar > 1.05 and zcr < 0.130:
        score = min(score, 0.15)

    return min(1.0, max(0.02, score))

correct = 0
for s in samples:
    pred_score = score_edge(s)
    label = 1 if s["label"] == "spoof" else 0
    pred = 1 if pred_score >= 0.50 else 0
    if pred == label:
        correct += 1
    status = "OK" if pred == label else "FAIL"
    print(f"{s['file'][:25]:25s} | {s['label']:8s} | score={pred_score:.3f} | {status}")

print(f"\nAccuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
