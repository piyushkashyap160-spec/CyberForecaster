import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.state_encoder import encode_window_to_state, STATE_FEATURE_KEYS
from preprocessing.scaler import StateScaler

def test_csv_loader():
    demo_path = "data/demo/demo_cicids2018.csv"
    assert os.path.exists(demo_path)
    df = load_flow_csv(demo_path)
    assert not df.empty
    assert "Timestamp" in df.columns
    assert "Src_IP" in df.columns
    assert "Dst_IP" in df.columns
    assert "Dst_Port" in df.columns

def test_window_builder():
    demo_path = "data/demo/demo_cicids2018.csv"
    df = load_flow_csv(demo_path)
    states = build_network_states(df, window_seconds=5.0)
    assert len(states) > 10

    X, y_state, y_attack, y_stage, ts = create_sequences(states, sequence_length=10)
    assert X.ndim == 3
    assert X.shape[1] == 10
    assert X.shape[2] == len(STATE_FEATURE_KEYS)
    assert len(y_state) == len(X)
    assert len(y_attack) == len(X)
    assert len(y_stage) == len(X)

def test_scaler():
    X = np.random.randn(50, 10, 23).astype(np.float32)
    scaler = StateScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape
    X_inv = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_inv, atol=1e-4)
