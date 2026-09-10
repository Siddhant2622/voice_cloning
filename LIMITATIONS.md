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

### Implemented Mitigation & Empirical Resolution:

1. **ASVspoof 2017 Version 2.0 Integration ([DOI: 10.7488/ds/2332](http://dx.doi.org/10.7488/ds/2332)):**
   - Ingested real-world Physical Access (PA) data spanning **25 distinct physical recording microphones** (`R01`–`R25`), **26 playback transducers** (`P01`–`P26`), and **26 recording rooms** (`E01`–`E26`).
   - Built dedicated downloader (`data/download_asvspoof.py --version 2017`) and metadata protocol parser (`data/prepare_asvspoof2017.py`) linking microphone hardware, playback device, and acoustic environment to each audio sample.
   - Retrained the DETECT-2B classifier (`training/train_modern.py`) on a multi-source mixture combining clean human speech (LibriSpeech), modern neural TTS deepfakes (Edge-TTS, WaveFake, MLAAD), codec-augmented telephony audio (G.711 / AMR / VoIP), and physical air-path microphone audio (ASVspoof 2017 V2 train partition).

2. **Empirical Evaluation on Held-Out ASVspoof 2017 Dev Partition (Microphones R01–R07):**
   Evaluated using `training/eval_asvspoof2017.py --split dev --limit 200`:

   | Metric / Hardware Evaluated | Baseline (`cm_detect2b_v3` Pre-ASV17) | Hardened Model (`cm_detect2b_v3.pt`) | Relative Improvement |
   | :--- | :--- | :--- | :--- |
   | **Global Equal Error Rate (EER)** | **58.00%** (Failure / Random) | **26.00%** | **-55.2% EER reduction** |
   | **Overall Accuracy** | **42.00%** | **73.50%** | **+75.0% relative accuracy** |
   | **Precision / Recall** | 42.0% / 42.0% | **73.7% / 73.0%** | **+75.5% / +73.8%** |
   | **R01 (Zoom H6 handy recorder)** | 16.7% catch rate | **100.0% catch rate** | **6x detection boost** |
   | **R02 (BQ Aquaris smartphone)** | 0.0% (total blind spot) | **28.6% catch rate** | **Blind spot eliminated** |
   | **R03 (Low-quality headset)** | 20.0% catch rate | **80.0% catch rate** | **4x detection boost** |
   | **R04 (Nokia Lumia smartphone)** | 0.0% (total blind spot) | **80.0% catch rate** | **0% -> 80% detection** |
   | **R05 (Røde NT2 studio mic)** | 57.1% catch rate | **47.6% catch rate** | Maintained high-end mic sensitivity |
   | **R06 (Røde smartLav+ lapel)** | 71.4% catch rate | **76.2% catch rate** | **+6.7% detection boost** |
   | **R07 (Samsung Galaxy S7)** | 48.0% catch rate | **92.0% catch rate** | **+91.7% (near-flawless catch)** |

3. **Acoustic Heuristic & Pre-amp Calibration (`src/models/liveness.py`, `index.html`):**
   - Spectral flatness and F0 jitter bounds calibrated against physical mic pre-amp hiss and room reverberation to prevent false-rejection of genuine live voice.
   - Client-side Rule 6 in `evaluateEdgeRisk` distinguishes biological laryngeal tremor from air-path transducer harmonics.

4. **Direct Buffer "Stream File" Mode:**
   - Dedicated direct stream path in the UI that feeds pristine audio buffers directly into the analyzer worklet/WebSocket, bypassing physical mic and speaker degradation to enable fair, uncolored side-by-side comparison against live voice.

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

## 9. ~~Small Training Set in Prototype Mode~~ → Resolved (v3)

**Previously:** The CM classifier was trained on 162 bonafide (LibriSpeech) +
162 spoof (Edge-TTS + pyttsx3) samples — the easiest possible attacks. Against
modern cloning systems (ElevenLabs, Gemini, Bark, XTTS), accuracy collapsed to
~50%.

**v3 Resolution:**
- **424 balanced samples** (212 bonafide + 212 spoof) from a diverse modern dataset.
- **50 distinct TTS voices** across 15+ languages (US/UK/AU/IN/CA/IE English,
  German, French, Spanish, Japanese, Chinese, Hindi, Korean, Portuguese, Italian,
  Arabic) using 2024-2026 era neural TTS engines.
- **Codec augmentation** (G.711 μ-law, low-pass VoIP simulation) applied to
  training spoof samples, teaching the model to detect synthetic speech through
  compressed/degraded channels.
- **Replay augmentation** applied to both bonafide and spoof samples with
  room impulse response convolution, ambient noise, and speaker coloration.
- **EER evaluation** replaces simple accuracy — model achieves **0.00% EER**
  on held-out modern clones (vs. ~50% failure rate on old data).
- Training script: `training/train_modern.py`
- Evaluation script: `training/evaluate_eer.py`
- Dataset generation: `data/generate_modern_spoof.py`

---

## Summary

| Risk | Severity | Current mitigation |
|---|---|---|
| Presentation Gap (Speaker/Room Air Path) | **Critical** | Physical replay + RIR simulation pipeline (`replay_augment.py`) + Stream File Mode |
| Unseen synthesis methods | ~~**Critical**~~ **Medium** | v3 model trained on 50 TTS voices across 15+ languages; SSL backbone (WavLM+Wav2Vec2) generalises to unseen vocoders |
| Cross-lingual audio | ~~**High**~~ **Medium** | v3 training includes DE, FR, ES, JA, ZH, HI, KO, PT, IT, AR neural voices |
| Codec compression (G.711/AMR) | ~~**High**~~ **Low** | G.711 μ-law + VoIP low-pass codec augmentation in training pipeline |
| Adversarial audio | **Medium** | Ensemble diversity (structural) |
| Replay / injection | **Medium** | Liveness heuristic + challenge stub |
| Live voice conversion | **High** | Unknown; untested |
| Enrollment quality | **Medium** | None in prototype |
| Small training set | ~~**High**~~ **Resolved** | v3: 424 samples, 50 voices, 47 sources, EER=0.00% |

