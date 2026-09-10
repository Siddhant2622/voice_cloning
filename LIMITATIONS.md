# LIMITATIONS.md — What Would Break This in Production

This document is a required artifact per the build specification. Being
specific and honest about failure modes is more useful than a headline
accuracy number.

---

## 1. Generalization to Unseen Synthesis Methods

**The core unsolved problem.** Every published anti-spoofing system, including
the state-of-the-art entries in ASVspoof 5 (2024), shows significant accuracy
degradation on synthesis methods not seen during training.

- A model trained on vocoder A (e.g. HiFi-GAN) does not automatically
  generalise to vocoder B (e.g. a new diffusion-based TTS).
- In this prototype the CM classifier is trained on `pyttsx3`-generated
  clips — the easiest possible attack. Against a modern voice-cloning system
  (Vall-E, Bark, ElevenLabs, Tortoise), expect accuracy to collapse until
  retrained on representative examples.

**Mitigation plan:**
- Add new synthesis methods to training data quarterly.
- Track score-distribution drift (Layer 8) as an early warning.
- Run periodic red-team exercises with current open/commercial cloners.
- The SSL backbone (WavLM-base) generalises better than hand-crafted features —
  its frozen features are the most transfer-stable part of the system.

---

## 2. Cross-Lingual Audio

The current feature stack is primarily validated on English:
- WavLM-base is English-dominant.
- The liveness heuristics (F0 jitter, spectral flatness) are calibrated on
  English phonetic distributions.
- For Mandarin, Hindi, Arabic, or tonal languages, expect degraded liveness
  scores and possibly biased CM outputs.

**Mitigation:** Replace or supplement WavLM-base with XLS-R (multilingual SSL
backbone). Re-calibrate liveness thresholds on per-language data. ASVspoof 5
includes multilingual tracks that serve as a benchmark.

---

## 3. Heavy Codec Compression (G.711 / AMR / Opus at Low Bitrate)

Real PSTN calls arrive at 8 kHz / G.711 μ-law or AMR-NB (narrow-band codec).
These codecs:
- Remove frequencies above 4 kHz — masking vocoder artifacts that live in
  the 4–8 kHz band.
- Introduce their own compression artifacts that can mimic, mask, or be
  confused with synthesis artifacts by a classifier.

The prototype operates on clean 16 kHz audio. When codec effects are present:
- Liveness scores (especially spectral contrast) will be unreliable.
- The CM classifier, trained on clean audio, may systematically mis-score
  codec-degraded genuine speech as synthetic.

**Mitigation:**
- Include codec-augmented versions of training data (G.711/AMR simulation
  via `torchaudio` or `sox`).
- The RADAR/ADD-style challenge datasets in §6 of the architecture are
  designed to test this — include them in evaluation.

---

## 4. Adversarial Audio

An attacker who has query access to the detector can, in principle:
- Use gradient-based adversarial examples (if they can access gradients).
- Use transfer-based attacks (craft adversarial perturbations on a local copy
  of the detector).
- Use score-probing to estimate decision boundaries.

**Current exposure in this prototype:** unbounded, because there is no rate
limiting or score quantisation in the demo.

**Mitigations implemented (in code comments):**
- Rate-limit exposed scores (architectural note in `src/risk_engine.py`).
- Round/quantise scores before returning them to the caller (architectural stub).
- Ensemble diversity is the main structural defence — an attacker who evades
  the CM classifier still has to beat the liveness heuristic and SV pillar.

**Not implemented:** adversarial training, randomised smoothing, or model
ensembling across architecturally distinct families (AASIST + RawNet2 would
be more robust than two WavLM heads).

---

## 5. The Presentation Gap: Replay and Room Acoustic Degradation

**The Presentation Gap is the primary cause of live voice clone false-negatives.**

When an anti-spoofing model (such as WavLM or Wav2Vec2) is trained solely on direct digital or clean studio audio, it learns representations sensitive to phase artifacts, high-frequency vocoder spectra, and pristine glottal pulses. In live testing or attack scenarios, synthetic speech is transmitted across a physical acoustic air path (e.g. played through a loudspeaker or smartphone and picked up by a laptop microphone).

### The Physical Failure Mechanism:
1. **Loudspeaker Frequency Shaping & Distortion:**
   - Physical transducers have steep high-pass bass rolloff (<140 Hz), resonant peak colorations (2.5–4.0 kHz), and driver harmonic distortion.
   - Non-linear compression masks the micro-level synthetic vocoder artifacts that the CM classifier relies on.
2. **Room Impulse Response (RIR) & Reverberation:**
   - Multi-path reflections ($RT_{60} \approx 0.20$ to $0.55$s) smear phoneme boundaries and blend vocoder phase discrepancies into diffuse room energy.
3. **Microphone Coloration & Ambient Room Noise:**
   - MEMS / Electret microphone frequency response curves, directional polar characteristics, and ambient background noise (HVAC, fans, thermal hiss at SNR 20–30 dB) degrade high-frequency signals (>7.5 kHz).
4. **The False-Bonafide Trap:**
   - Because genuine human speech spoken into a live microphone also exhibits room reverberation and mic coloration, a classifier unaccustomed to degraded synthetic audio misattributes this room coloration as "human acoustic naturalness," erroneously classifying the replayed spoof as bonafide.

### Implemented Mitigation:
- **Hybrid Replay Augmentation Pipeline (`data/replay_augment.py`):**
  - **Physical Replay:** Samples are played through physical loudspeakers and re-recorded via the microphone array in real time to capture true hardware transducer non-linearities.
  - **Acoustic Simulation:** Samples are convolved with multi-room impulse responses, speaker soft-clipping saturation, mic transfer curves, and room ambient noise profiles.
  - **Balanced Mixed Retraining:** The CM classifier (`cm_detect2b_v2.pt`) and Fusion model are retrained on a configurable mixture (e.g. 50% clean spoof, 50% replayed spoof vs bonafide speech), teaching the SSL backbone invariant spoof cues across both pristine and degraded acoustic channels.
- **Direct Buffer "Stream File" Mode:**
  - Added a dedicated direct stream path in the UI that feeds pristine audio buffers directly into the analyzer worklet/WebSocket, bypassing physical mic and speaker degradation to enable fair, uncolored side-by-side comparison against live voice.

---

## 6. Replay Attacks via Pre-Recorded Audio Injection

A replay attack (playing a pre-recorded genuine human clip through a virtual audio cable or speaker) differs from speech synthesis:
- Natural biological vocal tract characteristics are present in the recording.
- Liveness heuristics check F0 pitch jitter and dynamic spectral variance (low variance across repeating sessions indicates a pre-recorded loop).
- The active challenge-response step (random verification digits) guarantees temporal freshness and defeats static replay loops.

---

## 7. Live Real-Time Voice Conversion

The hardest attack case. The attacker speaks live; a voice-conversion model
reshapes their voice to the target's in ~200–400ms. As of 2025–2026, state-of-
the-art real-time conversion (FreeVC, RVC, OpenVoice) can be convincing enough
to defeat simple threshold-based detectors.

**Why this is hard to defend against:**
- The audio originates from a real human speaking in real time, so liveness
  signals (breath, jitter, room acoustics) can be nearly genuine.
- Conversion artifacts are present but subtle.
- The challenge-response only partially helps: the attacker can respond in
  real time (unlike pre-recorded injection) — the converted voice just has to
  match closely enough.

**Current prototype performance against real-time conversion:** unknown — we
have not tested it. Honest assessment: probably insufficient without targeted
fine-tuning on conversion examples.

---

## 8. Speaker Enrollment Quality

The SASV pillar (Layer 4b) only activates when a reference voiceprint is
enrolled. In IVR systems, enrollment quality varies:
- Enrollment recorded over a 2G/3G connection: quality degraded.
- Enrollment recorded in a noisy environment: higher false positive rate.
- Short enrollment audio (< 5s): insufficient for a reliable ECAPA-TDNN
  embedding.

In this prototype there is no minimum enrollment quality check.

---

## 9. Small Training Set in Prototype Mode

When trained on the demo `data/samples/` directory (pyttsx3 + fallback clips),
the CM classifier has at most ~10 examples of each class. Logistic fusion has
the same problem. Both models will overfit severely at this scale.

**Real accuracy numbers are only meaningful when evaluated on ASVspoof 2019
LA eval partition** (or similar benchmark). The `evaluate.py` script is the
correct verification path.

---

## Summary

| Risk | Severity | Current mitigation |
|---|---|---|
| Presentation Gap (Speaker/Room Air Path) | **Critical** | Physical replay + RIR simulation pipeline (`replay_augment.py`) + Stream File Mode |
| Unseen synthesis methods | **Critical** | SSL backbone (partial); retrain pipeline |
| Cross-lingual audio | **High** | None in prototype |
| Codec compression (G.711/AMR) | **High** | None in prototype |
| Adversarial audio | **Medium** | Ensemble diversity (structural) |
| Replay / injection | **Medium** | Liveness heuristic + challenge stub |
| Live voice conversion | **High** | Unknown; untested |
| Enrollment quality | **Medium** | None in prototype |
| Small training set | **High** | Use ASVspoof 2019 for real eval |

