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
