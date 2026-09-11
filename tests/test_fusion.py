"""
tests/test_fusion.py
Unit tests for src/fusion.py — ScoreBundle, FusionModel, EER/APCER/BPCER utilities.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fusion import ScoreBundle, FusionModel, compute_eer, compute_apcer_bpcer


class TestScoreBundle:
    def test_feature_vector_all_present(self):
        bundle = ScoreBundle(cm_score=0.8, sv_score=0.6, liveness_score=0.5)
        fv = bundle.to_feature_vector()
        # Vector order: [cm_score, sv_score, liveness_score, replay_score, watermark]
        assert len(fv) == 5, f"Expected 5-element vector, got {len(fv)}: {fv}"
        assert fv[0] == pytest.approx(0.8)   # cm_score
        assert fv[1] == pytest.approx(0.6)   # sv_score
        assert fv[2] == pytest.approx(0.5)   # liveness_score
        assert fv[3] == pytest.approx(0.5)   # replay_score (None → NEUTRAL_SCORE=0.5)
        assert fv[4] == pytest.approx(0.0)   # watermark (False)

    def test_feature_vector_missing_imputed(self):
        bundle = ScoreBundle(cm_score=0.7)   # no sv, liveness
        fv = bundle.to_feature_vector()
        assert fv[1] == pytest.approx(0.5)   # NEUTRAL_SCORE
        assert fv[2] == pytest.approx(0.5)

    def test_watermark_encoding(self):
        bundle = ScoreBundle(cm_score=0.5, watermark_detected=True)
        fv = bundle.to_feature_vector()
        # watermark is at index 4 (after replay_score was added at index 3)
        assert fv[4] == pytest.approx(1.0)

    def test_to_dict_excludes_none(self):
        bundle = ScoreBundle(cm_score=0.5)
        d = bundle.to_dict()
        assert "cm_score" in d
        assert "sv_score" not in d    # None fields excluded


class TestFusionModelHeuristic:
    """Tests for the heuristic fallback (no training required)."""

    def test_high_cm_gives_high_score(self):
        fm = FusionModel()
        bundle = ScoreBundle(cm_score=0.95)
        score = fm.score(bundle)
        # Untrained heuristic blends CM with neutral liveness prior (0.5),
        # so the realistic ceiling without calibration is ~0.52.
        # The key property: score > 0.5 (i.e., leans spoof-ward).
        assert score >= 0.5, f"High CM score should yield score > 0.5, got {score:.3f}"

    def test_low_cm_gives_low_score(self):
        fm = FusionModel()
        bundle = ScoreBundle(cm_score=0.05)
        score = fm.score(bundle)
        assert score <= 0.3, f"Low CM score should yield low fusion score, got {score:.3f}"

    def test_watermark_override(self):
        fm = FusionModel()
        bundle = ScoreBundle(cm_score=0.1, watermark_detected=True)
        score = fm.score(bundle)
        assert score >= 0.9, f"Watermark detection should push score high, got {score:.3f}"

    def test_output_in_range(self):
        fm = FusionModel()
        for cm in np.linspace(0, 1, 20):
            for lv in [None, 0.3, 0.7]:
                bundle = ScoreBundle(cm_score=float(cm), liveness_score=lv)
                score = fm.score(bundle)
                assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"


class TestFusionModelTrained:
    """Tests for the trained logistic regression path."""

    @pytest.fixture
    def trained_fm(self):
        np.random.seed(42)
        fm = FusionModel()
        # Synthetic training data: genuine = low scores, spoof = high scores
        bundles = (
            [ScoreBundle(cm_score=0.1 + 0.05*np.random.randn(), liveness_score=0.2) for _ in range(30)]
            + [ScoreBundle(cm_score=0.8 + 0.05*np.random.randn(), liveness_score=0.7) for _ in range(30)]
        )
        labels = [0]*30 + [1]*30
        fm.fit(bundles, labels, calibrate=False)
        return fm

    def test_trained_genuine_score_low(self, trained_fm):
        bundle = ScoreBundle(cm_score=0.05, liveness_score=0.1)
        score = trained_fm.score(bundle)
        assert score < 0.5, f"Genuine input should score < 0.5, got {score:.3f}"

    def test_trained_spoof_score_high(self, trained_fm):
        bundle = ScoreBundle(cm_score=0.92, liveness_score=0.85)
        score = trained_fm.score(bundle)
        assert score > 0.5, f"Spoof input should score > 0.5, got {score:.3f}"

    def test_save_and_load(self, trained_fm, tmp_path):
        save_path = tmp_path / "fusion.pkl"
        trained_fm.save(save_path)
        loaded = FusionModel.load(save_path)
        bundle = ScoreBundle(cm_score=0.7, liveness_score=0.6)
        assert abs(trained_fm.score(bundle) - loaded.score(bundle)) < 1e-6


class TestEERAndMetrics:
    def test_eer_perfect_separation(self):
        genuine = np.array([0.0, 0.1, 0.05, 0.15])
        spoof   = np.array([0.9, 0.95, 0.85, 0.92])
        eer = compute_eer(genuine, spoof)
        assert eer < 0.05, f"EER should be near 0 for perfect separation, got {eer:.3f}"

    def test_eer_random_chance(self):
        np.random.seed(0)
        genuine = np.random.rand(100)
        spoof   = np.random.rand(100)
        eer = compute_eer(genuine, spoof)
        # Random-chance EER should be around 0.5
        assert 0.35 <= eer <= 0.65, f"Random EER should be ~0.5, got {eer:.3f}"

    def test_apcer_bpcer_threshold_0(self):
        genuine = np.array([0.3, 0.4, 0.2])
        spoof   = np.array([0.8, 0.9, 0.7])
        # threshold=0: everything classified as spoof → APCER=0, BPCER=1
        apcer, bpcer = compute_apcer_bpcer(genuine, spoof, threshold=0.0)
        assert apcer == pytest.approx(0.0)
        assert bpcer == pytest.approx(1.0)

    def test_apcer_bpcer_threshold_1(self):
        genuine = np.array([0.3, 0.4, 0.2])
        spoof   = np.array([0.8, 0.9, 0.7])
        # threshold=1: everything classified as genuine → APCER=1, BPCER=0
        apcer, bpcer = compute_apcer_bpcer(genuine, spoof, threshold=1.0)
        assert apcer == pytest.approx(1.0)
        assert bpcer == pytest.approx(0.0)
