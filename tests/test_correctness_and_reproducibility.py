"""
test_correctness_and_reproducibility.py
=======================================
Comprehensive test suite verifying correctness, zero temporal leakage,
strict scaler discipline, timestamp parsing, temporal Snort correlation,
and honest lead-time calculation.
"""

import os
import yaml
import time
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from preprocessing.scaler import StateScaler
from preprocessing.window_builder import create_sequences
from monitoring.snort_correlator import SnortCorrelator
from forecasting.lead_time import compute_forecast_lead_time

def test_config_forecast_horizon_loading():
    """Verify config loads forecast horizon properly without undefined variable references."""
    config_path = "config.yaml"
    assert os.path.exists(config_path), "config.yaml must exist"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    seq_len = config['sequence']['sequence_length']
    forecast_horizon = config['sequence'].get('forecast_horizon', 5)
    assert isinstance(seq_len, int) and seq_len > 0
    assert isinstance(forecast_horizon, int) and forecast_horizon > 0

def test_zero_temporal_split_overlap():
    """Verify that sequences created within independent temporal state blocks do not cross split boundaries."""
    # Generate 100 synthetic sequential states S[0]..S[99]
    states = [
        {
            'vector': np.array([float(i)] * 23, dtype=np.float32),
            'is_attack': 1 if i >= 80 else 0,
            'stage': 2 if i >= 80 else 0,
            'timestamp': datetime.fromtimestamp(1700000000 + i * 5, tz=timezone.utc)
        }
        for i in range(100)
    ]
    
    # Chronological partition of raw states
    train_states = states[:60]
    val_states = states[60:80]
    test_states = states[80:]
    
    seq_len = 10
    forecast_steps = 5
    
    X_train, y_st_train, _, _, _ = create_sequences(train_states, sequence_length=seq_len, forecast_steps=forecast_steps)
    X_val, y_st_val, _, _, _ = create_sequences(val_states, sequence_length=seq_len, forecast_steps=forecast_steps)
    X_test, y_st_test, _, _, _ = create_sequences(test_states, sequence_length=seq_len, forecast_steps=forecast_steps)
    
    # Verify train sequences only contain states from 0 to 59
    max_train_val = np.max(X_train)
    assert max_train_val < 60.0, f"Train sequence leaked beyond state 59: max value was {max_train_val}"
    
    # Verify validation sequences only contain states from 60 to 79
    min_val_val = np.min(X_val)
    max_val_val = np.max(X_val)
    assert min_val_val >= 60.0, f"Validation sequence leaked into train: min value was {min_val_val}"
    assert max_val_val < 80.0, f"Validation sequence leaked into test: max value was {max_val_val}"
    
    # Verify test sequences only contain states from 80 to 99
    min_test_val = np.min(X_test)
    assert min_test_val >= 80.0, f"Test sequence leaked into validation: min value was {min_test_val}"

def test_scaler_discipline_train_only():
    """Verify StateScaler fits strictly on train split and transforms val/test without refitting."""
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(50, 10, 23))
    X_val = np.random.normal(loc=10.0, scale=2.0, size=(20, 10, 23))
    X_test = np.random.normal(loc=10.0, scale=2.0, size=(20, 10, 23))
    
    scaler = StateScaler()
    assert not scaler.is_fitted
    
    X_train_scaled = scaler.fit_transform(X_train)
    assert scaler.is_fitted
    
    train_mean = scaler.scaler.mean_.copy()
    train_var = scaler.scaler.var_.copy()
    
    # Transform validation and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Verify scaler parameters did NOT change during val/test transformation
    np.testing.assert_array_equal(scaler.scaler.mean_, train_mean)
    np.testing.assert_array_equal(scaler.scaler.var_, train_var)

def test_snort_timestamp_parsing():
    """Verify SnortCorrelator parses actual event timestamps from fast log lines."""
    correlator = SnortCorrelator(log_path="non_existent_test_log.fast")
    
    # Standard Snort format line with timestamp 08/29-23:45:12.123456
    sample_line = "08/29-23:45:12.123456 [**] [1:1000001:1] ICMP Ping Probe Detected [**] [Priority: 1] {ICMP} 192.168.1.5 -> 192.168.1.10"
    parsed = correlator._parse_snort_fast_line(sample_line)
    
    assert parsed is not None
    assert parsed["sig_id"] == "1000001"
    assert parsed["protocol"] == "ICMP"
    assert parsed["src_ip"] == "192.168.1.5"
    assert parsed["dst_ip"] == "192.168.1.10"
    assert "23:45:12" in parsed["timestamp"]
    assert isinstance(parsed["timestamp_epoch"], float)

def test_snort_temporal_correlation_recent_vs_stale():
    """Verify SnortCorrelator matches recent alerts and rejects stale alerts outside correlation window."""
    correlator = SnortCorrelator(log_path="non_existent_test_log.fast", correlation_window_seconds=15.0)
    correlator.is_connected = True
    
    t_now = time.time()
    
    recent_alert = {
        "sig_id": "1000003",
        "message": "IP Network Packet Detected",
        "protocol": "TCP",
        "src_ip": "192.168.1.5",
        "dst_ip": "192.168.1.10",
        "timestamp_epoch": t_now - 5.0,
        "timestamp": datetime.fromtimestamp(t_now - 5.0, tz=timezone.utc).isoformat()
    }
    
    stale_alert = {
        "sig_id": "1000003",
        "message": "IP Network Packet Detected (Stale)",
        "protocol": "TCP",
        "src_ip": "192.168.1.5",
        "dst_ip": "192.168.1.20",
        "timestamp_epoch": t_now - 60.0,
        "timestamp": datetime.fromtimestamp(t_now - 60.0, tz=timezone.utc).isoformat()
    }
    
    correlator.alerts_buffer = [stale_alert, recent_alert]
    
    # 1. Matching flow with recent alert -> MATCH
    flow_matching = {
        "src_ip": "192.168.1.5",
        "dst_ip": "192.168.1.10",
        "protocol": "TCP",
        "timestamp": t_now
    }
    match = correlator.correlate_flow(flow_matching)
    assert match is not None
    assert match["sig_id"] == "1000003"
    
    # 2. Matching flow with stale alert -> REJECTED (None)
    flow_stale_target = {
        "src_ip": "192.168.1.5",
        "dst_ip": "192.168.1.20",
        "protocol": "TCP",
        "timestamp": t_now
    }
    stale_match = correlator.correlate_flow(flow_stale_target)
    assert stale_match is None, "Stale alert older than correlation window must not match!"
    
    # 3. Wrong protocol -> REJECTED
    flow_wrong_proto = {
        "src_ip": "192.168.1.5",
        "dst_ip": "192.168.1.10",
        "protocol": "UDP",
        "timestamp": t_now
    }
    wrong_proto_match = correlator.correlate_flow(flow_wrong_proto)
    assert wrong_proto_match is None

def test_lead_time_post_onset_honesty_and_timestamp_accuracy():
    """Verify lead-time calculation properly penalizes post-onset detections and supports exact timestamps."""
    # Attack onset at index 4
    y_attack = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    
    # 1. Model detects 2 windows before onset (Index 2) -> +10.0s early warning
    probs_early = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    res_early = compute_forecast_lead_time(y_attack, probs_early, window_seconds=5.0, threshold=0.5)
    assert res_early["mean_lead_time_seconds"] == 10.0
    assert res_early["pre_onset_warnings"] == 1
    assert res_early["exact_onset_detections"] == 0
    assert res_early["post_onset_detections"] == 0
    
    # 2. Model detects at exact onset (Index 4) -> 0.0s exact onset
    probs_exact = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1])
    res_exact = compute_forecast_lead_time(y_attack, probs_exact, window_seconds=5.0, threshold=0.5)
    assert res_exact["mean_lead_time_seconds"] == 0.0
    assert res_exact["pre_onset_warnings"] == 0
    assert res_exact["exact_onset_detections"] == 1
    assert res_exact["post_onset_detections"] == 0
    
    # 3. Model detects 2 windows after onset (Index 6) -> -10.0s delay (NOT positive early warning)
    probs_late = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.9, 0.1])
    res_late = compute_forecast_lead_time(y_attack, probs_late, window_seconds=5.0, threshold=0.5)
    assert res_late["mean_lead_time_seconds"] == -10.0
    assert res_late["pre_onset_warnings"] == 0
    assert res_late["post_onset_detections"] == 1
    
    # 4. Real Timestamp list test
    base_t = datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)
    ts_list = [base_t + timedelta(seconds=i * 5) for i in range(len(y_attack))]
    res_ts = compute_forecast_lead_time(y_attack, probs_early, window_seconds=5.0, threshold=0.5, timestamps=ts_list)
    assert res_ts["mean_lead_time_seconds"] == 10.0
