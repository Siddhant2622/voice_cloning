"""
tests/test_features.py
Unit tests for src/features.py
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def torch():
    import torch
    return torch


@pytest.fixture(scope="module")
def sample_waveform(torch):
    """A 2-second sine wave at 16 kHz."""
    t = torch.linspace(0, 2.0, 32000)
    return torch.sin(2 * 3.14159 * 440 * t)


class TestLogMel:
    def test_output_shape(self, torch, sample_waveform):
        from src.features import extract_logmel
        logmel = extract_logmel(sample_waveform)
        assert logmel.ndim == 2
        n_mels, t_frames = logmel.shape
        assert n_mels == 80, f"Expected 80 mel bins, got {n_mels}"
        assert t_frames > 0

    def test_output_dtype(self, torch, sample_waveform):
        from src.features import extract_logmel
        logmel = extract_logmel(sample_waveform)
        assert logmel.dtype == torch.float32

    def test_no_nan_inf(self, torch, sample_waveform):
        from src.features import extract_logmel
        logmel = extract_logmel(sample_waveform)
        assert not torch.isnan(logmel).any(), "Log-mel contains NaN"
        assert not torch.isinf(logmel).any(), "Log-mel contains Inf"

    def test_silence_gives_low_values(self, torch):
        from src.features import extract_logmel
        silence = torch.zeros(32000)
        logmel  = extract_logmel(silence)
        # log(0 + 1e-9) ≈ -20.7 — should be strongly negative
        assert float(logmel.mean()) < -10.0


class TestSSLEmbedding:
    def test_output_shape(self, torch, sample_waveform):
        from src.features import extract_ssl_embedding
        emb = extract_ssl_embedding(sample_waveform)
        assert emb.ndim == 1
        assert emb.shape[0] == 768, f"Expected 768-dim WavLM-base embedding, got {emb.shape[0]}"

    def test_output_dtype(self, torch, sample_waveform):
        from src.features import extract_ssl_embedding
        emb = extract_ssl_embedding(sample_waveform)
        assert emb.dtype == torch.float32

    def test_no_nan(self, torch, sample_waveform):
        from src.features import extract_ssl_embedding
        emb = extract_ssl_embedding(sample_waveform)
        assert not torch.isnan(emb).any()

    def test_different_inputs_give_different_embeddings(self, torch):
        from src.features import extract_ssl_embedding
        t    = torch.linspace(0, 2, 32000)
        wav1 = torch.sin(2 * 3.14159 * 440 * t)
        wav2 = torch.sin(2 * 3.14159 * 880 * t)
        e1 = extract_ssl_embedding(wav1)
        e2 = extract_ssl_embedding(wav2)
        assert not torch.allclose(e1, e2, atol=1e-3), "Different inputs should give different embeddings"


class TestExtractAll:
    def test_returns_expected_keys(self, torch, sample_waveform):
        from src.features import extract_all
        feats = extract_all(sample_waveform)
        assert "logmel" in feats
        assert "ssl_embedding" in feats

    def test_shapes_are_consistent(self, torch, sample_waveform):
        from src.features import extract_all
        feats = extract_all(sample_waveform)
        assert feats["logmel"].shape[0] == 80
        assert feats["ssl_embedding"].shape[0] == 768
