"""
test_phase3_explainability.py
==============================
Comprehensive automated test suite for Phase 3: Evidence-Grounded Forecast Explainability.

Validates:
  1. Explanation schema compliance (all 8 canonical dimensions + provenance)
  2. Truthful evidence constraint (zero hallucinated sensors/alerts)
  3. Missing Snort alert handled honestly (available: False, "None correlated")
  4. Missing attack path handled honestly (available: False, "No active path")
  5. FastFlowDetector evidence represented accurately as anomaly suspicion, never proof of attack
  6. Model attribution explicitly targets chosen loss objective (attack_probability or stage_logit)
  7. Spatio-temporal attribution preserves relative timestep information (@ t, @ t-5s, etc.)
  8. Threat probability is decoupled from predictive uncertainty (retiring conflated confidence)
  9. MC-Dropout predictive uncertainty is reproducible under fixed seed
  10. Explanation generation is strictly side-effect free (eval mode preserved, identical predictions before/after)
  11. Backward compatibility for legacy callers (top_features list, confidence field)
  12. Evidence provenance tracking with valid source identifiers
  13. Stage transition dynamics only report progressions when historical stage exists
  14. Benign baseline forecasts do not fabricate attack rationales
"""

import pytest
import torch
import numpy as np

from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler
from preprocessing.state_encoder import STATE_FEATURE_KEYS
from explainability import (
    ForecastExplainer,
    GradientSaliencyExplainer,
    FEATURE_HUMAN_LABELS
)


@pytest.fixture
def test_setup():
    """Provides an initialized model, scaler, and deterministic sequence."""
    np.random.seed(42)
    torch.manual_seed(42)

    model = TemporalLSTMWorldModel(input_size=23, hidden_size=64, num_layers=2, dropout=0.2, num_stages=6)
    model.eval()

    # Create dummy 50-window training history and fit scaler
    train_history = np.random.uniform(10.0, 500.0, size=(50, 23)).astype(np.float32)
    scaler = StateScaler()
    scaler.fit(train_history)

    # 10-window physical sequence
    seq = np.random.uniform(50.0, 200.0, size=(10, 23)).astype(np.float32)
    # Inject a known feature elevation in syn_ratio (index 11) and unique_dst_ports (index 6)
    syn_idx = STATE_FEATURE_KEYS.index("syn_ratio")
    port_idx = STATE_FEATURE_KEYS.index("unique_dst_ports")
    seq[-1, syn_idx] = 0.95
    seq[-1, port_idx] = 800.0

    explainer = ForecastExplainer(model, scaler)
    return model, scaler, seq, explainer


def test_1_explanation_schema_validates(test_setup):
    """1. Validates that the full explanation object contains all required top-level keys."""
    model, scaler, seq, explainer = test_setup

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.15",
        predicted_stage="Lateral Movement",
        threat_probability=0.88,
        forecast_horizon_seconds=25
    )

    required_keys = [
        "forecast_id", "timestamp", "host_ip", "forecast", "narrative_summary",
        "model_evidence", "telemetry_evidence", "sensor_evidence",
        "stage_transition", "attack_path", "uncertainty", "provenance", "limitations"
    ]
    for k in required_keys:
        assert k in exp, f"Missing required top-level explanation key: {k}"

    assert exp["host_ip"] == "192.168.1.15"
    assert exp["forecast"]["predicted_stage"] == "Lateral Movement"
    assert exp["forecast"]["forecast_horizon_seconds"] == 25
    assert exp["forecast"]["threat_probability"] == 0.88
    assert exp["forecast"]["severity"] == "CRITICAL"


def test_2_truthful_evidence_no_hallucination(test_setup):
    """2. Verifies that when no sensor or path evidence is provided, none is synthesized."""
    model, scaler, seq, explainer = test_setup

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.20",
        predicted_stage="Normal",
        threat_probability=0.05,
        snort_match=None,
        fast_detection=None,
        active_paths=None
    )

    assert exp["sensor_evidence"]["snort"]["available"] is False
    assert exp["sensor_evidence"]["fastflow"]["available"] is False
    assert exp["attack_path"]["available"] is False


def test_3_missing_snort_handled_honestly(test_setup):
    """3. Verifies that missing Snort alert results in 'available: false' and 'None correlated'."""
    model, scaler, seq, explainer = test_setup

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.10",
        snort_match=None
    )

    snort_ev = exp["sensor_evidence"]["snort"]
    assert snort_ev["available"] is False
    assert snort_ev["sid"] is None
    assert "None correlated" in snort_ev["summary"]


def test_4_missing_attack_path_handled_honestly(test_setup):
    """4. Verifies that missing attack path produces 'available: false' and zero hops."""
    model, scaler, seq, explainer = test_setup

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.10",
        active_paths=[]
    )

    path_ev = exp["attack_path"]
    assert path_ev["available"] is False
    assert path_ev["hop_count"] == 0
    assert len(path_ev["hops"]) == 0
    assert "No active attack path" in path_ev["summary"]


def test_5_fastflow_evidence_accurate_wording(test_setup):
    """5. Verifies FastFlow evidence is labeled accurately as anomaly suspicion, not proof of attack."""
    model, scaler, seq, explainer = test_setup

    fast_det_sample = {
        "available": True,
        "status": "ACTIVE",
        "suspicious": True,
        "confidence": 0.94,
        "predicted_label": "Bot / Suspicious"
    }

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.10",
        fast_detection=fast_det_sample
    )

    ff = exp["sensor_evidence"]["fastflow"]
    assert ff["available"] is True
    assert ff["suspicious"] is True
    assert ff["score"] == 0.94
    assert "disclaimer" in ff
    assert "does not independently constitute proof of attack" in ff["disclaimer"]


def test_6_model_attribution_target_explicit(test_setup):
    """6. Verifies model attribution explicitly targets chosen objective (attack_probability vs stage_logit)."""
    model, scaler, seq, _ = test_setup

    saliency = GradientSaliencyExplainer(model, scaler)

    # Test attack_probability target
    res_prob = saliency.explain_instance(seq, target="attack_probability")
    assert res_prob["target"] == "attack_probability"

    # Test stage_logit target
    res_stage = saliency.explain_instance(seq, target="stage_logit", target_stage_id=3)
    assert res_stage["target"] == "stage_logit_3"


def test_7_temporal_attribution_preserves_timesteps(test_setup):
    """7. Verifies that temporal attribution identifies relative timesteps (@ t, @ t-5s, etc.)."""
    model, scaler, seq, _ = test_setup

    saliency = GradientSaliencyExplainer(model, scaler)
    res = saliency.explain_instance(seq)

    assert "top_temporal_features" in res
    assert len(res["top_temporal_features"]) > 0

    first_item = res["top_temporal_features"][0]
    assert "timestep" in first_item
    assert "relative_step" in first_item
    assert "@" in first_item["label"]
    assert first_item["direction"] in ["increases_risk", "decreases_risk", "neutral"]


def test_8_threat_probability_decoupled_from_uncertainty(test_setup):
    """8. Verifies threat_probability is distinct from predictive uncertainty."""
    model, scaler, seq, explainer = test_setup

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.15",
        threat_probability=0.88,
        uncertainty_samples=10,
        seed=42
    )

    # Threat probability is the point forecast
    assert exp["forecast"]["threat_probability"] == 0.88
    # Uncertainty is the empirical dispersion
    unc = exp["uncertainty"]
    assert unc["available"] is True
    assert "mean_probability" in unc
    assert "std" in unc
    assert "confidence_band" in unc
    assert unc["std"] >= 0.0


def test_9_mc_dropout_deterministic_with_seed(test_setup):
    """9. Verifies MC-Dropout uncertainty is deterministic under fixed seed."""
    model, scaler, seq, explainer = test_setup

    u1 = explainer.compute_mc_uncertainty(seq, num_samples=10, seed=123)
    u2 = explainer.compute_mc_uncertainty(seq, num_samples=10, seed=123)

    assert u1["mean_probability"] == u2["mean_probability"]
    assert u1["std"] == u2["std"]
    assert u1["confidence_band"] == u2["confidence_band"]


def test_10_explanation_generation_has_no_side_effects(test_setup):
    """10. Verifies explanation generation does not alter model predictions or leave model in train mode."""
    model, scaler, seq, explainer = test_setup

    seq_scaled = scaler.transform(seq[np.newaxis, :, :])
    t_in = torch.tensor(seq_scaled, dtype=torch.float32)

    # Prediction before explanation
    model.eval()
    with torch.no_grad():
        _, p_before, _ = model(t_in)
    val_before = float(p_before.numpy()[0, 0])

    # Generate full explanation
    _ = explainer.generate_explanation(seq, threat_probability=val_before, uncertainty_samples=10)

    # Model must be in eval mode
    assert model.training is False

    # Prediction after explanation must be identical
    with torch.no_grad():
        _, p_after, _ = model(t_in)
    val_after = float(p_after.numpy()[0, 0])

    assert np.isclose(val_before, val_after, atol=1e-7)


def test_11_backward_compatibility_payload(test_setup):
    """11. Verifies that top_features list (length <= 10) is preserved for legacy consumers."""
    model, scaler, seq, _ = test_setup

    saliency = GradientSaliencyExplainer(model, scaler)
    res = saliency.explain_instance(seq)

    assert "top_features" in res
    assert len(res["top_features"]) <= 10
    assert "feature" in res["top_features"][0]
    assert "attribution" in res["top_features"][0]
    assert "abs_importance" in res["top_features"][0]


def test_12_evidence_provenance_sources(test_setup):
    """12. Verifies evidence provenance tracks all active data sources with hashes."""
    model, scaler, seq, explainer = test_setup

    snort_match = {"sig_id": "1000003", "message": "TEST ALERT", "timestamp": "2026-09-07T20:00:00Z"}
    fast_det = {"available": True, "status": "ACTIVE", "suspicious": True, "confidence": 0.91}
    active_paths = [{
        "path_id": "path-test-1",
        "source": "192.168.1.10",
        "destination": "192.168.1.20",
        "severity": "HIGH",
        "hops": [{"hop_index": 1, "from": "192.168.1.10", "to": "192.168.1.20"}]
    }]

    exp = explainer.generate_explanation(
        sequence=seq,
        host_ip="192.168.1.10",
        snort_match=snort_match,
        fast_detection=fast_det,
        active_paths=active_paths,
        data_hash="abc123hash",
        blockchain_tx="0x999tx"
    )

    prov = exp["provenance"]
    assert "TemporalLSTM" in prov["evidence_sources"]
    assert "NetworkState" in prov["evidence_sources"]
    assert "FastFlowDetector" in prov["evidence_sources"]
    assert "Snort" in prov["evidence_sources"]
    assert "AttackPathReconstructor" in prov["evidence_sources"]
    assert "BlockchainLedger" in prov["evidence_sources"]
    assert prov["blockchain_registered"] is True
    assert prov["data_hash"] == "abc123hash"


def test_13_stage_transition_requires_historical_state(test_setup):
    """13. Verifies stage transition only indicates progression when a previous stage exists."""
    model, scaler, seq, explainer = test_setup

    # Case A: No previous stage
    exp_no_prev = explainer.generate_explanation(
        sequence=seq,
        predicted_stage="Lateral Movement",
        previous_stage=None
    )
    assert exp_no_prev["stage_transition"]["transition_detected"] is False

    # Case B: Genuine progression from Reconnaissance to Lateral Movement
    exp_prog = explainer.generate_explanation(
        sequence=seq,
        predicted_stage="Lateral Movement",
        previous_stage="Reconnaissance"
    )
    assert exp_prog["stage_transition"]["transition_detected"] is True
    assert "Reconnaissance toward Lateral Movement" in exp_prog["stage_transition"]["summary"]


def test_14_benign_forecast_no_fabricated_attack_rationale(test_setup):
    """14. Verifies that benign baseline forecasts do not fabricate attack rationales."""
    model, scaler, seq, explainer = test_setup

    # Nominal uniform baseline
    benign_seq = np.full((10, 23), 50.0, dtype=np.float32)

    exp_benign = explainer.generate_explanation(
        sequence=benign_seq,
        host_ip="192.168.1.50",
        predicted_stage="Normal",
        threat_probability=0.03,
        previous_stage="Normal"
    )

    assert exp_benign["forecast"]["severity"] == "NORMAL"
    assert exp_benign["stage_transition"]["transition_detected"] is False
    assert "Nominal" in exp_benign["stage_transition"]["summary"]
    assert exp_benign["sensor_evidence"]["snort"]["available"] is False
    assert exp_benign["sensor_evidence"]["fastflow"]["available"] is False
    assert exp_benign["attack_path"]["available"] is False
