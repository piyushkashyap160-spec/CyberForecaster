"""
test_live_collector_flow_generation.py — Integration tests for packet-to-flow pipeline
"""

import time
import asyncio
import pytest
from preprocessing.live_collector import FlowAggregator, LiveNetworkCollector

def test_flow_aggregator_packet_insertion_and_timeout():
    """Verify that adding packets creates active flows and inactivity emits finalized flows."""
    agg = FlowAggregator(flow_timeout=0.2)

    pkt1 = {
        "src_ip": "192.168.1.50",
        "dst_ip": "93.184.216.34",
        "src_port": 50123,
        "dst_port": 443,
        "protocol": "TCP",
        "length": 64,
        "tcp_flags": {"S"}
    }
    pkt2 = {
        "src_ip": "192.168.1.50",
        "dst_ip": "93.184.216.34",
        "src_port": 50123,
        "dst_port": 443,
        "protocol": "TCP",
        "length": 128,
        "tcp_flags": {"A", "P"}
    }

    # First packet: must create active flow entry
    res1 = agg.add_packet(pkt1)
    assert res1 is None, "First packet should not finalize flow immediately"
    assert len(agg._active_flows) == 1, "Active flow table must have 1 entry"

    # Second packet of same 5-tuple: must update existing flow
    res2 = agg.add_packet(pkt2)
    assert res2 is None, "Second packet within timeout should update active flow"
    assert len(agg._active_flows) == 1, "Still 1 active flow"
    flow_state = list(agg._active_flows.values())[0]
    assert flow_state["packet_count"] == 2
    assert flow_state["byte_count"] == 192
    assert "S" in flow_state["tcp_flags"]
    assert "P" in flow_state["tcp_flags"]

    # Wait for flow inactivity timeout (> 0.2s)
    time.sleep(0.25)
    expired = agg.flush_expired()
    assert len(expired) == 1, "Flow should be expired and flushed"
    final_flow = expired[0]
    assert final_flow["packet_count"] == 2
    assert final_flow["byte_count"] == 192
    assert final_flow["syn_count"] == 1
    assert final_flow["ack_count"] == 1

def test_bidirectional_flow_normalization():
    """Verify that forward (A->B) and reverse (B->A) packets merge into the same canonical flow."""
    agg = FlowAggregator(flow_timeout=1.0)

    # Client -> Server request packet
    req_pkt = {
        "src_ip": "192.168.1.5",
        "dst_ip": "93.184.216.34",
        "src_port": 55432,
        "dst_port": 443,
        "protocol": "TCP",
        "length": 60,
        "tcp_flags": {"S"}
    }
    # Server -> Client response packet
    resp_pkt = {
        "src_ip": "93.184.216.34",
        "dst_ip": "192.168.1.5",
        "src_port": 443,
        "dst_port": 55432,
        "protocol": "TCP",
        "length": 1420,
        "tcp_flags": {"S", "A"}
    }

    agg.add_packet(req_pkt)
    assert len(agg._active_flows) == 1
    agg.add_packet(resp_pkt)
    assert len(agg._active_flows) == 1, "Bidirectional packets must be merged into 1 flow"

    flow_state = list(agg._active_flows.values())[0]
    assert flow_state["packet_count"] == 2
    assert flow_state["byte_count"] == 1480
    assert "S" in flow_state["tcp_flags"]
    assert "A" in flow_state["tcp_flags"]

def test_tcp_fin_rst_termination_flushing():
    """Verify that a FIN or RST packet immediately triggers flow finalization."""
    agg = FlowAggregator(flow_timeout=5.0)

    syn_pkt = {
        "src_ip": "192.168.1.10",
        "dst_ip": "10.0.0.1",
        "src_port": 49152,
        "dst_port": 80,
        "protocol": "TCP",
        "length": 64,
        "tcp_flags": {"S"}
    }
    fin_pkt = {
        "src_ip": "10.0.0.1",
        "dst_ip": "192.168.1.10",
        "src_port": 80,
        "dst_port": 49152,
        "protocol": "TCP",
        "length": 40,
        "tcp_flags": {"F", "A"}
    }

    res1 = agg.add_packet(syn_pkt)
    assert res1 is None
    assert len(agg._active_flows) == 1

    # FIN packet should immediately finalize the flow without waiting for timeout
    completed = agg.add_packet(fin_pkt)
    assert completed is not None, "FIN packet must immediately finalize and return completed flow"
    assert completed["packet_count"] == 2
    assert completed["fin_count"] == 1
    assert len(agg._active_flows) == 0, "Flow table should be empty after termination"

def test_live_collector_queue_emission():
    """Verify that collector emits flows to asyncio queue."""
    async def _async_test():
        collector = LiveNetworkCollector(flow_timeout=0.1)
        loop = asyncio.get_running_loop()
        collector.set_loop(loop)

        sample_flow = {
            "src_ip": "10.0.0.5",
            "dst_ip": "10.0.0.1",
            "src_port": 443,
            "dst_port": 51234,
            "protocol": "TCP",
            "packet_count": 5,
            "byte_count": 1500,
            "duration": 0.5,
            "avg_packet_size": 300.0,
            "capture_timestamp": "2026-08-29T12:00:00Z",
            "syn_count": 1,
            "ack_count": 1,
            "rst_count": 0,
            "fin_count": 0
        }

        # Simulate flow emission
        collector._emit_flow(sample_flow)
        assert collector.flows_generated == 1

        # Queue should receive the flow
        received_flow = await asyncio.wait_for(collector.flow_queue.get(), timeout=1.0)
        assert received_flow["src_ip"] == "10.0.0.5"
        assert received_flow["byte_count"] == 1500

    asyncio.run(_async_test())
