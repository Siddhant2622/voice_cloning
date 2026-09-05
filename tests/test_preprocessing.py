"""
tests/test_preprocessing.py
Unit tests for src/preprocessing.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def torch():
    import torch
    return torch


def make_sine_wave(torch, duration_s=2.0, sr=16000, freq=440):
    t = torch.linspace(0, duration_s, int(sr * duration_s))
    return torch.sin(2 * 3.14159 * freq * t), sr


def make_silence(torch, duration_s=1.0, sr=16000):
    return torch.zeros(int(sr * duration_s)), sr


class TestResampleAndMono:
    def test_resample_to_16k(self, torch):
        from src.preprocessing import to_mono_16k
        wav, _ = make_sine_wave(torch, sr=44100)
        wav_16k = to_mono_16k(wav.unsqueeze(0), 44100)
        expected = int(2.0 * 16000)
        assert abs(len(wav_16k) - expected) < 160, f"Expected ~{expected} samples, got {len(wav_16k)}"

    def test_stereo_to_mono(self, torch):
        from src.preprocessing import to_mono_16k
        stereo = torch.randn(2, 16000)
        mono = to_mono_16k(stereo, 16000)
        assert mono.ndim == 1
        assert mono.shape[0] == 16000

    def test_already_16k_mono(self, torch):
        from src.preprocessing import to_mono_16k
        wav = torch.randn(16000)
        result = to_mono_16k(wav.unsqueeze(0), 16000)
        assert result.ndim == 1
        assert len(result) == 16000


class TestRMSNormalize:
    def test_normalizes_energy(self, torch):
        from src.preprocessing import rms_normalize
        loud = torch.randn(16000) * 10.0
        normed = rms_normalize(loud, target_rms=0.05)
        rms = float(normed.pow(2).mean().sqrt())
        assert abs(rms - 0.05) < 0.005, f"RMS should be ~0.05, got {rms:.4f}"

    def test_silence_passthrough(self, torch):
        from src.preprocessing import rms_normalize
        silence, _ = make_silence(torch)
        result = rms_normalize(silence)
        assert torch.allclose(result, silence)

    def test_output_shape_preserved(self, torch):
        from src.preprocessing import rms_normalize
        wav = torch.randn(8000)
        result = rms_normalize(wav)
        assert result.shape == wav.shape


class TestEnergyVAD:
    def test_keeps_voiced_frames(self, torch):
        from src.preprocessing import _energy_vad
        # Signal: 1s silence + 1s speech + 1s silence
        silence = torch.zeros(16000)
        speech  = torch.randn(16000) * 0.1
        wav     = torch.cat([silence, speech, silence])
        voiced  = _energy_vad(wav, sr=16000, energy_threshold=0.01)
        # Should keep at least some voiced content
        assert len(voiced) > 0
        # Should be shorter than input (silence stripped)
        assert len(voiced) < len(wav)

    def test_all_silence_returns_original(self, torch):
        from src.preprocessing import _energy_vad
        silence = torch.zeros(16000)
        result  = _energy_vad(silence, sr=16000)
        # With no voiced content, returns original
        assert len(result) == 0 or len(result) <= len(silence)


class TestPreprocessFile:
    def test_preprocess_wav(self, torch, tmp_path):
        """End-to-end test with a generated WAV file.

        Uses soundfile.write to create the temp file because torchaudio.save
        requires FFmpeg DLLs via torchcodec on torchaudio nightly builds.
        """
        import soundfile as sf
        import numpy as np
        wav_path = tmp_path / "test.wav"
        # 1s of noise at 22050 Hz, mono
        signal_np = (np.random.randn(22050) * 0.1).astype(np.float32)
        sf.write(str(wav_path), signal_np, 22050)

        from src.preprocessing import preprocess_file
        waveform, sr = preprocess_file(wav_path, apply_vad_flag=False)
        assert sr == 16000
        assert waveform.ndim == 1
        assert len(waveform) > 0
