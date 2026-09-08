"""
tests/test_api.py
Integration tests for api/server.py endpoints and real-time audio scoring.
"""

import io
import json
import numpy as np
import pytest
import torch
import soundfile as sf
from fastapi.testclient import TestClient

from api.server import app, _load_models, _score_waveform, get_settings


@pytest.fixture(scope="module", autouse=True)
def init_models():
    settings = get_settings()
    _load_models(settings)


def test_score_waveform_tensor():
    waveform = torch.zeros(16000, dtype=torch.float32)
    scores = _score_waveform(waveform)
    assert "cm_score" in scores
    assert "liveness_score" in scores
    assert "fusion_score" in scores
    assert "decision" in scores
    assert isinstance(scores["fusion_score"], float)
    assert scores["decision"] in {"ALLOW", "STEP_UP", "ALERT", "BLOCK"}


def test_score_waveform_numpy():
    waveform_np = np.zeros(16000, dtype=np.float32)
    scores = _score_waveform(waveform_np)
    assert "cm_score" in scores
    assert "liveness_score" in scores
    assert "fusion_score" in scores
    assert "decision" in scores
    assert isinstance(scores["fusion_score"], float)


def test_health_endpoint():
    with TestClient(app) as client:
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in {"ready", "loading"}
        assert "uptime_s" in data
        assert "version" in data


def test_analyze_endpoint():
    with TestClient(app) as client:
        # Create a synthetic WAV in memory
        sr = 16000
        duration = 1.0
        audio = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)

        headers = {"x-api-key": "dev-key-voiceguard"}
        res = client.post(
            "/api/v1/analyze",
            files={"file": ("test.wav", buf.read(), "audio/wav")},
            headers=headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert "cm_score" in data
        assert "liveness_score" in data
        assert "fusion_score" in data
        assert "decision" in data
        assert data["duration_s"] > 0


def test_ws_stream():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            # Send silence chunk (16000 samples = 1 sec at 16kHz)
            silence = np.zeros(16000, dtype=np.int16).tobytes()
            ws.send_bytes(silence)

            # Receive score update
            msg = ws.receive_json()
            assert msg.get("type") == "score"
            assert "fusion_score" in msg
            assert "decision" in msg
            assert msg["decision"] == "ALLOW"
            assert msg.get("reason") is not None
