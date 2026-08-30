"""
test_live_inference_endpoints.py — Unit tests for live world model status, benchmark, and inference endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import fastapi_app, convert_flows_to_cyberforecaster_dataframe
from preprocessing.state_encoder import encode_window_to_state

client = TestClient(fastapi_app)

def test_world_model_status_endpoint():
    """Verify that /api/world_model/status returns valid architecture and warm-up metadata."""
    res = client.get("/api/world_model/status")
    assert res.status_code == 200
    data = res.json()
    assert "model" in data
    assert "status" in data
    assert data["status"] in ["READY", "WARMING UP"]
    assert "canonical_rmse_reference" in data
    assert data["uncertainty_status"] == "Uncertainty unavailable"

def test_canonical_benchmark_endpoint():
    """Verify that /api/benchmark/canonical serves canonical results."""
    res = client.get("/api/benchmark/canonical")
    assert res.status_code == 200
    data = res.json()
    assert "classification_results" in data
    assert "future_state_rmse_benchmarks" in data
    assert "lead_time_evaluation" in data

def test_live_flow_to_state_conversion():
    """Verify that live flow converts to valid 23-D state vector matching scaler dimension."""
    sample_flow = {
        "src_ip": "192.168.1.15",
        "dst_ip": "142.250.190.46",
        "src_port": 51234,
        "dst_port": 443,
        "protocol": "TCP",
        "packet_count": 8,
        "byte_count": 3200,
        "duration": 0.25,
        "avg_packet_size": 400.0,
        "capture_timestamp": "2026-08-29T23:30:00Z",
        "syn_count": 1,
        "ack_count": 1,
        "rst_count": 0,
        "fin_count": 0
    }
    df = convert_flows_to_cyberforecaster_dataframe([sample_flow])
    state = encode_window_to_state(df, window_seconds=5.0)
    vec = state["vector_flow_only"]
    assert len(vec) == 23, "Live state vector must be 23-dimensional"
    assert vec[0] == 8.0, "Total packets must match flow"
    assert vec[1] == 3200.0, "Total bytes must match flow"

def test_socketio_handshake_endpoints():
    """Verify that Socket.IO handshake returns 200 OK on both fastapi_app and app."""
    from backend.main import app
    from starlette.testclient import TestClient as StarletteTestClient

    # Test on fastapi_app directly
    res_fastapi = client.get("/socket.io/?EIO=4&transport=polling")
    assert res_fastapi.status_code == 200, f"fastapi_app socket.io handshake failed: {res_fastapi.status_code}"
    assert "sid" in res_fastapi.text

    # Test on app (combined ASGI app)
    client_app = StarletteTestClient(app)
    res_app = client_app.get("/socket.io/?EIO=4&transport=polling")
    assert res_app.status_code == 200, f"app socket.io handshake failed: {res_app.status_code}"
    assert "sid" in res_app.text


def test_cold_start_warmup_status():
    """Verify that world model status correctly reflects cold-start status without fabrication."""
    res = client.get("/api/world_model/status")
    assert res.status_code == 200
    data = res.json()
    assert "windows_collected" in data
    assert "windows_required" in data
    assert data["windows_required"] == 10
    assert data["status"] in ["READY", "WARMING UP"]


def test_collector_status_bytes_captured():
    """Verify that collector status always returns a non-null bytes_captured integer."""
    from backend.main import LIVE_COLLECTOR
    res = client.get("/api/collector/status")
    assert res.status_code == 200
    data = res.json()
    assert "bytes_captured" in data
    assert isinstance(data["bytes_captured"], int)
    assert data["bytes_captured"] >= 0

    # Also test LiveNetworkCollector.get_status directly
    status = LIVE_COLLECTOR.get_status()
    assert "bytes_captured" in status
    assert isinstance(status["bytes_captured"], int)


def test_fast_flow_detector_endpoint():
    """Verify that fast flow detector endpoint reports status without fabrication."""
    res = client.get("/api/flow_detector/status")
    assert res.status_code == 200
    data = res.json()
    assert "model_type" in data
    assert "status" in data
    assert data["status"] in ["ACTIVE", "NOT_CONFIGURED"]


def test_snort_status_endpoint():
    """Verify that Snort correlator reports connection status accurately."""
    res = client.get("/api/snort/status")
    assert res.status_code == 200
    data = res.json()
    assert "connected" in data
    assert "status" in data
    assert data["status"] in ["CONNECTED", "NOT CONNECTED"]


def test_mitigations_endpoints_and_actions():
    """Verify that defensive host actions update state and are logged in mitigations table."""
    from backend.main import seed_hosts_and_history
    seed_hosts_and_history()

    # Enforce Rate Limit on 192.168.1.10
    res_act = client.post("/api/hosts/action", json={"ip": "192.168.1.10", "action": "RATE_LIMIT"})
    assert res_act.status_code == 200
    act_data = res_act.json()
    assert act_data["host"]["status"] == "RATE_LIMITED"

    # Query mitigations list
    res_mit = client.get("/api/mitigations")
    assert res_mit.status_code == 200
    mits = res_mit.json()
    assert len(mits) > 0
    assert mits[0]["hostIp"] == "192.168.1.10"
    assert mits[0]["action"] == "RATE_LIMIT"

    # Reset host back to ONLINE
    res_rst = client.post("/api/hosts/action", json={"ip": "192.168.1.10", "action": "RESET"})
    assert res_rst.status_code == 200
    assert res_rst.json()["host"]["status"] == "ONLINE"


def test_fast_flow_detector_8d_feature_extraction():
    """Verify exact 8-D feature vector extraction matching training order."""
    from preprocessing.flow_detector import FastFlowDetector
    detector = FastFlowDetector()
    sample_flow = {
        "duration": 0.55,
        "byte_count": 1280,
        "packet_count": 8,
        "protocol": "TCP",
        "avg_packet_size": 160.0,
        "syn_count": 1,
        "ack_count": 1,
        "rst_count": 0
    }
    feats = detector.extract_features(sample_flow)
    assert len(feats) == 8
    assert feats[0] == 0.55   # duration_sec
    assert feats[1] == 1280.0 # byte_count
    assert feats[2] == 8.0    # packet_count
    assert feats[3] == 1.0    # is_tcp (TCP=1.0)
    assert feats[4] == 160.0  # avg_packet_size
    assert feats[5] == 1.0    # syn_count
    assert feats[6] == 1.0    # ack_count
    assert feats[7] == 0.0    # rst_count


def test_fast_flow_detector_prediction_schema():
    """Verify live prediction schema and confidence score output."""
    from preprocessing.flow_detector import FastFlowDetector
    detector = FastFlowDetector()
    if detector.is_configured:
        sample_flow = {
            "duration": 0.05,
            "byte_count": 64000,
            "packet_count": 100,
            "protocol": "TCP",
            "avg_packet_size": 640.0,
            "syn_count": 1,
            "ack_count": 0,
            "rst_count": 0
        }
        res = detector.predict_flow(sample_flow)
        assert res["available"] is True
        assert res["status"] == "ACTIVE"
        assert isinstance(res["suspicious"], bool)
        assert isinstance(res["confidence"], float)
        assert 0.0 <= res["confidence"] <= 1.0
        assert "predicted_label" in res


def test_fast_flow_detector_missing_checkpoint_behavior():
    """Verify that detector reports NOT_CONFIGURED honestly when checkpoint is absent."""
    from preprocessing.flow_detector import FastFlowDetector
    detector = FastFlowDetector(model_path="models_weights/non_existent_detector.joblib")
    assert detector.is_configured is False
    res = detector.predict_flow({"duration": 0.1})
    assert res["available"] is False
    assert res["status"] == "NOT_CONFIGURED"
    assert res["confidence"] == 0.0
    assert res["suspicious"] is False
