import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prevention import PreventionEngine, BlockedSessionError, PreventionResult
from src.risk_engine import PolicyEngine, PolicyDecision, RiskContext, Decision
from src.fusion import ScoreBundle, FusionModel


class TestBlockedSessionError:
    def test_attributes(self):
        err = BlockedSessionError("AI detected", risk_score=0.91, session_id="s1")
        assert err.reason     == "AI detected"
        assert err.risk_score == pytest.approx(0.91)
        assert err.session_id == "s1"

    def test_to_dict(self):
        err = BlockedSessionError("Spoof", risk_score=0.85, session_id="abc")
        d = err.to_dict()
        assert d["type"]       == "blocked"
        assert d["risk_score"] == pytest.approx(0.85, abs=1e-4)

    def test_is_exception(self):
        with pytest.raises(BlockedSessionError):
            raise BlockedSessionError("test", risk_score=0.99)


class TestPolicyEngineThresholds:
    @pytest.fixture
    def engine(self): return PolicyEngine()

    def test_block_at_high_score(self, engine):
        dec = engine.decide(RiskContext(fusion_score=0.92))
        assert dec.action == Decision.BLOCK

    def test_alert_at_medium_high(self, engine):
        dec = engine.decide(RiskContext(fusion_score=0.75))
        assert dec.action == Decision.ALERT

    def test_step_up_at_medium(self, engine):
        dec = engine.decide(RiskContext(fusion_score=0.55))
        assert dec.action == Decision.STEP_UP

    def test_allow_at_low(self, engine):
        dec = engine.decide(RiskContext(fusion_score=0.20))
        assert dec.action == Decision.ALLOW

    def test_decision_has_reason(self, engine):
        dec = engine.decide(RiskContext(fusion_score=0.50))
        assert dec.reason and len(dec.reason) > 5

    def test_contributing_signals_present(self, engine):
        dec = engine.decide(RiskContext(fusion_score=0.60))
        assert "acoustic_fusion_score" in dec.contributing_signals


class TestPolicyEnginePenalties:
    @pytest.fixture
    def engine(self): return PolicyEngine()

    def test_caller_id_spoofed_raises_score(self, engine):
        base    = engine.decide(RiskContext(fusion_score=0.40))
        flagged = engine.decide(RiskContext(fusion_score=0.40, caller_id_spoofed=True))
        assert flagged.final_risk_score > base.final_risk_score

    def test_liveness_penalty(self, engine):
        lo = engine.decide(RiskContext(fusion_score=0.50, liveness_score=0.30))
        hi = engine.decide(RiskContext(fusion_score=0.50, liveness_score=0.85))
        assert hi.final_risk_score > lo.final_risk_score

    def test_replay_penalty(self, engine):
        lo = engine.decide(RiskContext(fusion_score=0.50, replay_score=0.20))
        hi = engine.decide(RiskContext(fusion_score=0.50, replay_score=0.80))
        assert hi.final_risk_score > lo.final_risk_score

    def test_combined_flags_can_cross_block(self, engine):
        ctx = RiskContext(fusion_score=0.60, caller_id_spoofed=True,
            sim_swap_flagged=True, geolocation_mismatch=True, number_reputation_bad=True)
        assert engine.decide(ctx).action == Decision.BLOCK

    def test_score_never_exceeds_1(self, engine):
        ctx = RiskContext(fusion_score=0.99, caller_id_spoofed=True,
            sim_swap_flagged=True, liveness_score=1.0, replay_score=1.0)
        assert engine.decide(ctx).final_risk_score <= 1.0


class TestPreventionEngine:
    @pytest.fixture
    def engine(self): return PreventionEngine(challenge_mode="digits")

    def _d(self, action, score=0.5):
        return PolicyDecision(action=action, reason=f"Test {action.value}",
            final_risk_score=score, contributing_signals={"test": True})

    def test_block_raises(self, engine):
        with pytest.raises(BlockedSessionError) as ei:
            engine.apply(self._d(Decision.BLOCK, 0.93), session_id="s1")
        assert ei.value.risk_score == pytest.approx(0.93)

    def test_step_up_returns_challenge(self, engine):
        r = engine.apply(self._d(Decision.STEP_UP, 0.60))
        assert r.action == "STEP_UP"
        assert r.challenge is not None
        assert r.challenge.expires_in_s > 0

    def test_alert_no_challenge(self, engine):
        r = engine.apply(self._d(Decision.ALERT, 0.75))
        assert r.action == "ALERT" and r.challenge is None

    def test_allow_result(self, engine):
        r = engine.apply(self._d(Decision.ALLOW, 0.20))
        assert r.action == "ALLOW"
        assert r.risk_score == pytest.approx(0.20)

    def test_allow_dict_serialisable(self, engine):
        import json
        d = engine.apply(self._d(Decision.ALLOW, 0.15)).to_dict()
        json.dumps(d)
        assert d["action"] == "ALLOW"

    def test_step_up_dict_has_challenge(self, engine):
        import json
        d = engine.apply(self._d(Decision.STEP_UP, 0.60)).to_dict()
        json.dumps(d)
        assert "challenge" in d


class TestEndToEndFusionPrevention:
    def test_high_cm_risky(self):
        b = ScoreBundle(cm_score=0.95, liveness_score=0.80)
        d = PolicyEngine().decide(RiskContext(fusion_score=FusionModel().score(b), liveness_score=b.liveness_score))
        assert d.action in (Decision.ALERT, Decision.BLOCK, Decision.STEP_UP)

    def test_low_cm_allows(self):
        b = ScoreBundle(cm_score=0.05, liveness_score=0.10)
        d = PolicyEngine().decide(RiskContext(fusion_score=FusionModel().score(b)))
        assert d.action == Decision.ALLOW

    def test_watermark_forces_block(self):
        b = ScoreBundle(cm_score=0.1, watermark_detected=True)
        d = PolicyEngine().decide(RiskContext(fusion_score=FusionModel().score(b)))
        assert d.action == Decision.BLOCK

    def test_prevention_blocks_on_watermark(self):
        b = ScoreBundle(cm_score=0.1, watermark_detected=True)
        d = PolicyEngine().decide(RiskContext(fusion_score=FusionModel().score(b)))
        with pytest.raises(BlockedSessionError):
            PreventionEngine().apply(d, session_id="wm-test")
