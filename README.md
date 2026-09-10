# VoiceGuard — Real-Time Voice Clone Detection Prototype

A **defensive anti-fraud research prototype** that detects AI-synthesized or
voice-converted speech in near real time and demonstrates prevention actions.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FSiddhant2622%2FVoice-Cloning)

> **Ground rules:** This system is a *detector only*. No voice cloning, TTS,
> or voice conversion capability is included. All synthetic audio used for
> testing comes from public benchmark datasets or non-identifying scripted
> sentences via the system TTS engine (`pyttsx3`).

---

## 🌐 Deploy to Vercel

You can deploy the VoiceGuard detection dashboard to **Vercel** with a single click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FSiddhant2622%2FVoice-Cloning)

### Vercel Deployment Features:
- **Zero-Config Instant Deployment**: Static dashboard and Serverless API functions ready out of the box.
- **Edge Acoustic Analysis Engine**: Uses Web Audio API directly in the browser to analyze microphone audio and test audio files in real time (spectral flatness, zero-crossing rate, RMS gating).
- **One-Click Audio Presets**: Test bona fide human speech, synthetic AI voice clones, and neural TTS with pre-bundled sample audio.
- **Hybrid GPU Connectivity**: Connect the Vercel frontend to your local or hosted GPU PyTorch backend (`ws://localhost:8000/ws/stream` or custom `wss://`) via the built-in Settings modal.
- **Serverless API Routes**:
  - `GET /health` or `GET /api/health` — System status
  - `POST /challenge` or `POST /api/challenge` — Step-up challenge generator
  - `GET /recent-logs` or `GET /api/recent-logs` — Audit telemetry
  - `POST /detect` or `POST /api/detect` — Serverless risk policy scoring

---

## Quick Start (Local Full-Stack with GPU)

### 1. Install dependencies

```bash
# PyTorch (CUDA 12.8 — adjust for your GPU or remove +cu128 for CPU)
python -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# All other dependencies
python -m pip install -r requirements.txt
```

### 2. Generate demo samples

```bash
python data/generate_samples.py
```

### 3. Train the CM classifier

```bash
python training/train_cm.py --data data/samples --epochs 30 --out models/cm.pt
```

### 4. Run the CLI detector (Phase 1)

```bash
python detect.py data/samples/synthetic_00.wav
python detect.py data/samples/bonafide_00.flac --transaction-value 50000 --action-type wire_transfer
```

### 5. Start the live demo (Phase 2)

```bash
python demo/server.py
# Open http://localhost:8000 in your browser
# Click "Start Mic" or "Stream File" to begin live detection
```

### 6. Stream a file to the demo

```bash
python demo/stream_file.py data/samples/synthetic_00.wav
```

### 7. Fetch Datasets & Hardened Models from Google Drive
```bash
# Fetch ASVspoof 2017 V2 physical microphone dataset
python data/download_from_drive.py --target asvspoof2017

# Fetch hardened DETECT-2B v3 model checkpoint
python data/download_from_drive.py --target models

# Or fetch everything
python data/download_from_drive.py --target all
```

### 8. Evaluate on ASVspoof 2017 V2 (Physical Microphones & Replay)
```bash
# Benchmark on real physical microphones (R01–R07)
python training/eval_asvspoof2017.py --split dev --checkpoint models/cm_detect2b_v3.pt
```

### 9. Evaluate on ASVspoof 2019 (Phase 3)

```bash
python data/download_asvspoof.py --partition eval
python training/train_fusion.py --data data/ASVspoof2019_LA --cm-model models/cm_detect2b_v3.pt
python evaluate.py --data data/ASVspoof2019_LA --partition eval
```

### 10. Run tests

```bash
python -m pytest tests/ -v
```

---

## Architecture

Nine-layer architecture described in detail in [`ARCHITECTURE.md`](ARCHITECTURE.md).

```
Input audio
    │
    ▼
[L1] Preprocessing (VAD, resample, normalise)
    │
    ▼
[L3] Feature extraction (log-mel + WavLM-base SSL)
    │
    ├──▶ [L4a] CM Classifier          (bona fide vs spoof score)
    ├──▶ [L4b] Speaker Verify         (ECAPA-TDNN cosine similarity)
    ├──▶ [L4c] Liveness Heuristic     (F0 jitter, spectral flatness)
    │
    ▼
[L5] Score Fusion                     (calibrated logistic regression)
    │
    ▼
[L6] Contextual Risk Engine           (metadata + transaction context)
    │
    ▼
[L7] Decision & Prevention            (ALLOW / STEP_UP / ALERT / BLOCK)
    │
    ▼
[L9] Decision Log                     (structured JSON-lines audit trail)
```

---

## Project Structure

```
Voice Cloning/
├── detect.py               CLI: score a single file
├── evaluate.py             Evaluation harness (EER, APCER/BPCER, DET curve)
├── requirements.txt
├── setup.py
├── src/
│   ├── preprocessing.py    Layer 1 — VAD, resample, normalize
│   ├── features.py         Layer 3 — log-mel + WavLM SSL embedding
│   ├── fusion.py           Layer 5 — calibrated score fusion
│   ├── risk_engine.py      Layers 6/7 — context + policy engine
│   ├── challenge.py        Layer 4c — active challenge-response (stub)
│   ├── decision_log.py     Layer 9 — structured audit log
│   └── models/
│       ├── cm_classifier.py   Layer 4a — CM countermeasure classifier
│       ├── speaker_verify.py  Layer 4b — ECAPA-TDNN speaker verification
│       └── liveness.py        Layer 4c — passive liveness heuristic
├── demo/
│   ├── server.py           FastAPI + WebSocket streaming server
│   ├── stream_file.py      File → WebSocket streamer
│   ├── benchmark_latency.py   p50/p95/p99 latency benchmark
│   └── static/index.html   Live detection UI (dark mode, glassmorphism)
├── training/
│   ├── train_cm.py         Train CM classifier
│   └── train_fusion.py     Train fusion layer
├── data/
│   ├── generate_samples.py Demo sample generator
│   └── download_asvspoof.py ASVspoof 2019 LA downloader
├── tests/                  Unit tests (pytest)
├── config/policy.yaml      Decision threshold configuration
├── Dockerfile
├── docker-compose.yml
├── ARCHITECTURE.md         Built vs. production design comparison
└── LIMITATIONS.md          Honest failure-mode analysis
```

---

## Key CLI Options

```bash
# With speaker enrollment (activates SASV pillar)
python detect.py test.wav --enroll reference.wav

# High-value transaction context (raises risk threshold adjustment)
python detect.py test.wav --transaction-value 100000 --action-type wire_transfer

# JSON output for programmatic use
python detect.py test.wav --json | python -m json.tool

# Evaluation on a custom dataset dir with bonafide/ and spoof/ subdirs
python evaluate.py --data /path/to/dataset --threshold 0.45
```

---

## Docker

```bash
docker compose up --build
# Demo available at http://localhost:8000
```

---

## Evaluation Metrics

Per ISO/IEC 30107-3:2023:

| Metric | Description |
|---|---|
| **EER** | Equal Error Rate — headline anti-spoofing metric |
| **APCER** | Attack Presentation Classification Error Rate — missed attacks |
| **BPCER** | Bona fide Presentation Classification Error Rate — false alarms |
| **Generalization gap** | EER on held-out attack families vs. trained ones |

See [`LIMITATIONS.md`](LIMITATIONS.md) for why headline accuracy numbers can be misleading.

---

## Telephony Integration

See [`docs/telephony_integration.md`](docs/telephony_integration.md) for
interface documentation showing where SIP/Asterisk AGI, FreeSWITCH ESL, and
Twilio webhook integration would connect.
