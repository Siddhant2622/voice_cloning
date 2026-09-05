# ARCHITECTURE.md — What Was Built vs. the Full Production Design

## What This Is

A **research prototype** of the core detection pipeline described in
`voice-clone-detection-architecture.md`. It implements the signal-processing and
ML heart of the system, plus minimal-viable prevention stubs, so the design can
be exercised end-to-end on real audio.

---

## Layer-by-Layer: Built vs. Production

| Layer | Architecture description | What's in this prototype |
|---|---|---|
| **L0 — Audio ingestion** | PSTN/SIP, WebRTC, Twilio, IVR | WebSocket (binary PCM) from browser mic or file streamer. Telephony integration: stub only — see `docs/telephony_integration.md` |
| **L1 — Preprocessing** | VAD, denoise, echo-cancel, resample, diarization | `src/preprocessing.py`: Silero VAD (torch.hub) with energy fallback, resample to 16 kHz mono, RMS normalisation. **Not built:** denoising, echo cancellation, multi-speaker diarization |
| **L2 — Watermark check** | AudioSeal / Perth / C2PA | Stub field in `ScoreBundle.watermark_detected`. A hard override in the fusion heuristic (score → 0.97 if watermark found). **Not built:** actual watermark detector |
| **L3 — Feature extraction** | Log-mel, CQT, SSL (wav2vec2/WavLM/XLS-R), speaker embeddings, prosodic | `src/features.py`: 80-band log-mel + frozen WavLM-base SSL embedding (768-dim). **Not built:** CQT, XLS-R multilingual, raw-waveform features |
| **L4a — CM classifier** | AASIST, RawNet2, SSL + attentive pooling | `src/models/cm_classifier.py`: 1-D CNN log-mel branch + SSL projection, attentive statistics pooling, 2-layer MLP head. Inspired by AASIST-lite / SSL-pooling approaches |
| **L4b — SASV** | ECAPA-TDNN + ASV fusion | `src/models/speaker_verify.py`: SpeechBrain ECAPA-TDNN wrapper, cosine similarity → spoof probability. Active only when enrollment audio is provided |
| **L4c — Liveness** | Passive room-acoustics, active challenge | `src/models/liveness.py`: spectral flatness + F0 jitter + spectral contrast heuristic. `src/challenge.py`: digit/phrase challenge generator + edit-distance verifier stub |
| **L5 — Score fusion** | Calibrated logistic regression / MLP | `src/fusion.py`: scikit-learn LogisticRegression + Platt scaling. Falls back to weighted heuristic before training |
| **L6 — Contextual risk** | Caller-ID, geolocation, SIM-swap, transaction value | `src/risk_engine.py`: `RiskContext` dataclass, additive penalty model. All fields are stubs — `INTEGRATION_POINT` comments mark where real APIs go |
| **L7 — Decision & prevention** | Allow / Step-up / Alert / Block | `src/risk_engine.py`: `PolicyEngine` with configurable YAML thresholds. Active challenge response stub in `src/challenge.py` |
| **L8 — Continuous learning** | Retraining loop, drift monitoring | **Not built.** `training/train_cm.py` + `training/train_fusion.py` are the retraining hooks. Drift monitoring: documented, not implemented |
| **L9 — Observability** | Structured logs, dashboards, GDPR hooks | `src/decision_log.py`: JSON-lines audit log with all signal scores. Prometheus / Grafana: not built; `INTEGRATION_POINT` comments show where metrics emit |

---

## File → Architecture Layer Map

```
src/preprocessing.py          Layer 1
src/features.py               Layer 3
src/models/cm_classifier.py   Layer 4a
src/models/speaker_verify.py  Layer 4b
src/models/liveness.py        Layer 4c
src/challenge.py              Layer 4c (active)
src/fusion.py                 Layer 5
src/risk_engine.py            Layers 6 + 7
src/decision_log.py           Layer 9
demo/server.py                Layer 0 (WebSocket ingestion) + orchestration
demo/static/index.html        Real-time visualization UI
detect.py                     CLI orchestrator (all layers, file input)
evaluate.py                   Evaluation harness (§7 metrics)
training/train_cm.py          Layer 4a training
training/train_fusion.py      Layer 5 training
data/download_asvspoof.py     Benchmark dataset acquisition
config/policy.yaml            Layer 7 threshold configuration
```

---

## Key Simplifications and Trade-offs

1. **SSL backbone is frozen.** Production deployment would fine-tune on ASVspoof data for 2–5 epochs. The prototype uses mean-pooled hidden states without fine-tuning, which sacrifices some accuracy for zero training-data-at-boot usability.

2. **Fusion is logistic regression.** Sufficient at prototype scale. For production with diverse attack families, a small attention-based fusion head (as in recent SASV papers) would generalise better.

3. **No streaming-friendly causal attention.** The demo uses a sliding window (2s / 1s hop) which recomputes features on every window. Production should use causal streaming inference with incremental computation.

4. **WebSocket, not Kafka/Redis Streams.** The prototype uses a direct WebSocket for simplicity. The architecture §5 tech stack calls for Kafka/Redis at scale; the `INTEGRATION_POINT` comments in `demo/server.py` show where to plug them in.

5. **No real watermark detector.** AudioSeal and Perth are available as open-source tools. Integrating one is a 1-day addition; it's not included here to keep dependencies minimal.

---

## What to Build Next (Priority Order)

1. Fine-tune the SSL backbone on ASVspoof 2019 LA train split — largest single accuracy gain
2. Add AudioSeal watermark detector to Layer 2
3. Implement causal/streaming attention for lower latency
4. Wire SIP/Asterisk AGI or Twilio webhook to Layer 0 for real telephony
5. Add Prometheus metrics and Grafana dashboard to Layer 9
6. Implement the continuous-learning loop (Layer 8)
