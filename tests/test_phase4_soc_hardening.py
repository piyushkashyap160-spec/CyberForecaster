"""
tests/test_phase4_soc_hardening.py
===================================
Phase 4 Regression Tests: SOC Demo Hardening, End-to-End Reliability & Benchmark Integrity.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import fastapi_app, load_models_and_config, seed_hosts_and_history, ALERTS_DB, HOSTS_DB

# Ensure models, scaler, explainer, and host history are initialized for testing
load_models_and_config()
seed_hosts_and_history()

client = TestClient(fastapi_app)


def test_blockchain_status_endpoint():
    """Verify that /api/blockchain/status returns valid schema and truthful provider status."""
    res = client.get("/api/blockchain/status")
    assert res.status_code == 200
    data = res.json()
    assert "connected" in data
    assert isinstance(data["connected"], bool)
    assert "provider_url" in data
    assert "contract_deployed" in data
    assert "status" in data
    assert data["status"] in ["CONNECTED", "NODE_CONNECTED_NO_CONTRACT", "OFFLINE"]


def test_blockchain_verify_truthful_offline_or_not_found():
    """
    Verify that /api/blockchain/verify/{alert_id} truthfully reports false
    when an alert is not found, or when node is offline, without fabricating
    fake block numbers (such as 1042) or claiming fake authenticity.
    """
    # Non-existent alert
    res_fake = client.get("/api/blockchain/verify/non-existent-uuid-9999")
    assert res_fake.status_code == 200
    data_fake = res_fake.json()
    assert data_fake["isAuthentic"] is False
    assert data_fake["status"] == "NOT_FOUND"
    assert data_fake["blockchain"] is None

    # Alert in ALERTS_DB verified while Hardhat node is offline
    dummy_alert = {
        "_id": "test-alert-uuid-1234",
        "hostIp": "192.168.1.10",
        "severity": "HIGH",
        "predictedStage": "Lateral Movement",
        "threat_probability": 0.85,
        "attack_probability": 0.85,
        "dataHash": "d42f79f5fdbbd95f1234567890abcdef",
        "blockchainTxHash": None
    }
    ALERTS_DB.append(dummy_alert)
    try:
        res = client.get("/api/blockchain/verify/test-alert-uuid-1234")
        assert res.status_code == 200
        data = res.json()
        assert data["alertId"] == "test-alert-uuid-1234"
        # If Hardhat node is offline, must report NODE_OFFLINE or UNREGISTERED
        if not data.get("blockchain"):
            assert data["isAuthentic"] is False
            assert data["status"] in ["NODE_OFFLINE", "UNREGISTERED", "CONTRACT_NOT_DEPLOYED"]
            assert data.get("blockchain") is None
    finally:
        ALERTS_DB.remove(dummy_alert)


def test_explain_endpoint_nominal_host():
    """
    Verify that /api/forecasts/explain/{host_ip} returns a valid structured explanation
    for a nominal host without raising an unhandled exception or fabricating alerts.
    """
    res = client.get("/api/forecasts/explain/192.168.1.10")
    assert res.status_code == 200
    data = res.json()
    assert "forecast_id" in data
    assert "host_ip" in data
    assert data["host_ip"] == "192.168.1.10"
    assert "model_evidence" in data
    assert "telemetry_evidence" in data
    assert "sensor_evidence" in data
    assert "uncertainty" in data
    assert "provenance" in data
    assert "narrative_summary" in data

    # Verify provenance disclosures
    prov = data["provenance"]
    assert "model" in prov
    assert "scaler" in prov
    assert "evidence_sources" in prov
    assert "TemporalLSTM" in prov["evidence_sources"]


def test_canonical_benchmark_data_integrity():
    """
    Verify that the canonical Phase 2 benchmark endpoint serves the exact,
    unaltered scientific numbers from canonical_benchmark_results.json.
    """
    res = client.get("/api/benchmark/canonical")
    assert res.status_code == 200
    bench = res.json()

    # Classification test partition results
    lstm_cls = bench["classification_results"]["Temporal_LSTM_World_Model_23D"]
    assert lstm_cls["Precision"] == 0.9692
    assert lstm_cls["Recall"] == 0.9965
    assert lstm_cls["F1_Score"] == 0.9826
    assert lstm_cls["FPR"] == 0.0629

    # Future-state RMSE benchmarks
    horizons = bench["future_state_rmse_benchmarks"]["horizons"]
    assert horizons["K_1_step_5s_ahead"]["lstm_rollout_rmse"] == 2.0549
    assert horizons["K_1_step_5s_ahead"]["persistence_rmse"] == 11.7373
    assert horizons["K_1_step_5s_ahead"]["training_mean_rmse"] == 2.1498

    assert horizons["K_3_steps_15s_ahead"]["lstm_rollout_rmse"] == 2.3484
    assert horizons["K_5_steps_25s_ahead"]["lstm_rollout_rmse"] == 2.1577

    # Lead time disclosure
    lead = bench["lead_time_evaluation"]
    assert lead["validated_lead_time_seconds"] == 0.0
    assert lead["genuine_pre_onset_detections"] == 0


def test_missing_optional_fields_resilience():
    """
    Verify that the backend /api/inference endpoint handles requests with missing
    optional sequence/state_vector without crashing.
    """
    res = client.post("/api/inference", json={"hostIp": "192.168.1.10"})
    assert res.status_code == 200
    data = res.json()
    assert "threat_probability" in data
    assert "predictedStage" in data
    assert "rollout" in data
    assert isinstance(data["rollout"], list)
