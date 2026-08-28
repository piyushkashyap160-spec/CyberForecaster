"""
test_backend_collector.py — Test backend live network collector API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import fastapi_app

client = TestClient(fastapi_app)

def test_collector_interfaces_endpoint():
    response = client.get("/api/collector/interfaces")
    assert response.status_code == 200
    data = response.json()
    assert "interfaces" in data
    assert isinstance(data["interfaces"], list)

def test_collector_status_endpoint():
    response = client.get("/api/collector/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "packets_captured" in data
    assert "flows_generated" in data

def test_collector_start_stop_workflow():
    # Test start
    start_resp = client.post("/api/collector/start", json={"interface": "eth0"})
    assert start_resp.status_code in [200, 400] # 400 if scapy/npcap unavailable on test machine

    # Test status after start attempt
    status_resp = client.get("/api/collector/status")
    assert status_resp.status_code == 200

    # Test stop
    stop_resp = client.post("/api/collector/stop")
    assert stop_resp.status_code == 200
    stop_data = stop_resp.json()
    assert stop_data.get("success") is True
