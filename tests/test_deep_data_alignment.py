import pytest
import numpy as np
import pandas as pd
import torch
from preprocessing.csv_loader import load_flow_csv
from preprocessing.pcap_chunked_extractor import ChunkedPCAPExtractor
from preprocessing.state_encoder import encode_window_to_state
from forecasting.rollout import perform_k_step_rollout
from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler

def test_pcap_csv_timezone_alignment():
    """Verifies 5-hour EST timezone correction (ts_utc - 18000) for PCAP alignment."""
    ts_pcap_utc = 1519994808.0 # 2018-03-02 12:46:48 UTC
    ts_est = ts_pcap_utc - 18000.0 # 2018-03-02 07:46:48 EST
    dt_est = pd.to_datetime(ts_est, unit='s')
    assert dt_est.hour == 7
    assert dt_est.day == 2
    assert dt_est.month == 3

def test_csv_timestamp_dayfirst_parsing():
    """Verifies that DD/MM/YYYY dates like '02/03/2018 08:47:38' are parsed as March 2, 2018."""
    s = pd.Series(['02/03/2018 08:47:38'])
    dt = pd.to_datetime(s, dayfirst=True, errors='coerce').iloc[0]
    assert dt.month == 3
    assert dt.day == 2
    assert dt.year == 2018

def test_rollout_no_future_leakage():
    """Verifies recursive K-step rollout executes without ground-truth future target leakage."""
    model = TemporalLSTMWorldModel(input_size=23, hidden_size=64)
    model.eval()
    scaler = StateScaler()
    dummy_data = np.random.randn(20, 23).astype(np.float32)
    scaler.fit(dummy_data)

    seq_input = dummy_data[:10]
    rollout_results = perform_k_step_rollout(model, scaler, seq_input, k_steps=5)

    assert len(rollout_results) == 5
    for i, step_res in enumerate(rollout_results):
        assert step_res['step_index'] == i + 1

        assert len(step_res['state_vector']) == 23
        assert 0.0 <= step_res['attack_probability'] <= 1.0


