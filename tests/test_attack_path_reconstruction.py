"""
test_attack_path_reconstruction.py
==================================
Unit tests for real-time Attack Path Reconstruction and Multi-Hop Lateral Progression engine.
"""

import time
import pytest
from monitoring.attack_path_reconstructor import AttackPathReconstructor


def test_no_suspicious_flows_no_path():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)

    event1 = {
        "hostIp": "192.168.1.20",
        "dstIp": "192.168.1.10",
        "protocol": "TCP",
        "total_bytes": 1200,
        "fast_detection": {"suspicious": False},
        "snort_alert": None
    }
    event2 = {
        "hostIp": "192.168.1.10",
        "dstIp": "192.168.1.15",
        "protocol": "TCP",
        "total_bytes": 4500,
        "fast_detection": {"suspicious": False},
        "snort_alert": None
    }

    reconstructor.add_flow_event(event1)
    paths = reconstructor.add_flow_event(event2)
    assert len(paths) == 0, "Benign flows must not form an attack path."


def test_single_suspicious_hop_no_path():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)

    event = {
        "hostIp": "198.51.100.5",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "total_bytes": 2048,
        "fast_detection": {"suspicious": True, "predicted_label": "Bot", "confidence": 0.95},
        "snort_alert": None
    }
    paths = reconstructor.add_flow_event(event)
    assert len(paths) == 0, "A single hop cannot be a multi-hop lateral progression."


def test_two_sequential_suspicious_hops_creates_path():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)
    t0 = time.time()

    hop1 = {
        "hostIp": "198.51.100.5",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "total_bytes": 4096,
        "timestamp": "2026-08-31T01:00:00+00:00",
        "fast_detection": {"suspicious": True, "predicted_label": "Bot", "confidence": 0.98},
        "snort_alert": None
    }
    hop2 = {
        "hostIp": "192.168.1.20",
        "dstIp": "192.168.1.10",
        "protocol": "TCP",
        "total_bytes": 1024,
        "timestamp": "2026-08-31T01:00:05+00:00",
        "fast_detection": None,
        "snort_alert": {"sig_id": "1000003", "message": "Lateral Reconnaissance Detected"}
    }

    reconstructor.add_flow_event(hop1)
    paths = reconstructor.add_flow_event(hop2)

    assert len(paths) == 1, "Two sequential suspicious hops must form a valid attack path."
    p = paths[0]
    assert p["source"] == "198.51.100.5"
    assert p["destination"] == "192.168.1.10"
    assert p["hop_count"] == 2
    assert p["hops"][0]["from"] == "198.51.100.5"
    assert p["hops"][0]["to"] == "192.168.1.20"
    assert p["hops"][1]["from"] == "192.168.1.20"
    assert p["hops"][1]["to"] == "192.168.1.10"


def test_bidirectional_traffic_does_not_create_path():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)

    hop1 = {
        "hostIp": "192.168.1.20",
        "dstIp": "192.168.1.10",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:00+00:00",
        "fast_detection": {"suspicious": True, "predicted_label": "Bot", "confidence": 0.90},
        "snort_alert": None
    }
    hop2 = {
        "hostIp": "192.168.1.10",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:02+00:00",
        "fast_detection": {"suspicious": True, "predicted_label": "Bot", "confidence": 0.90},
        "snort_alert": None
    }

    reconstructor.add_flow_event(hop1)
    paths = reconstructor.add_flow_event(hop2)
    assert len(paths) == 0, "Bidirectional reverse traffic (A -> B, B -> A) must NOT form a lateral progression."


def test_self_loop_rejected():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)

    event = {
        "hostIp": "192.168.1.10",
        "dstIp": "192.168.1.10",
        "protocol": "TCP",
        "fast_detection": {"suspicious": True}
    }
    paths = reconstructor.add_flow_event(event)
    assert len(paths) == 0
    assert len(reconstructor.edge_buffer) == 0, "Self-loop edge must be discarded."


def test_temporal_window_expiry():
    reconstructor = AttackPathReconstructor(window_seconds=30.0)

    hop1 = {
        "hostIp": "198.51.100.5",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:00+00:00",
        "fast_detection": {"suspicious": True}
    }
    hop2 = {
        "hostIp": "192.168.1.20",
        "dstIp": "192.168.1.10",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:02:00+00:00",  # 120s later (> 30s)
        "fast_detection": {"suspicious": True}
    }

    reconstructor.add_flow_event(hop1)
    paths = reconstructor.add_flow_event(hop2)
    assert len(paths) == 0, "Events outside the sliding time window must not be linked."


def test_duplicate_edges_do_not_create_fake_hops():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)

    hop1 = {
        "hostIp": "198.51.100.5",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:00+00:00",
        "fast_detection": {"suspicious": True}
    }
    hop1_dup = {
        "hostIp": "198.51.100.5",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:00.200+00:00",
        "fast_detection": {"suspicious": True}
    }

    reconstructor.add_flow_event(hop1)
    paths = reconstructor.add_flow_event(hop1_dup)
    assert len(paths) == 0, "Duplicate edges must not create fake hops."


def test_three_hop_progression():
    reconstructor = AttackPathReconstructor(window_seconds=60.0)

    hop1 = {
        "hostIp": "198.51.100.5",
        "dstIp": "192.168.1.20",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:00+00:00",
        "fast_detection": {"suspicious": True}
    }
    hop2 = {
        "hostIp": "192.168.1.20",
        "dstIp": "192.168.1.10",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:04+00:00",
        "snort_alert": {"sig_id": "1000003", "message": "Lateral Reconnaissance"}
    }
    hop3 = {
        "hostIp": "192.168.1.10",
        "dstIp": "192.168.1.15",
        "protocol": "TCP",
        "timestamp": "2026-08-31T01:00:08+00:00",
        "fast_detection": {"suspicious": True, "predicted_label": "Bot"}
    }

    reconstructor.add_flow_event(hop1)
    reconstructor.add_flow_event(hop2)
    paths = reconstructor.add_flow_event(hop3)

    assert len(paths) == 1
    p = paths[0]
    assert p["source"] == "198.51.100.5"
    assert p["destination"] == "192.168.1.15"
    assert p["hop_count"] == 3
    assert p["severity"] == "CRITICAL"
