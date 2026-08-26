import os
import io
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

def test_csv_loader_uploaded_file_buffer():
    """
    Regression test: Verifies load_flow_csv handles file-like buffers
    such as Streamlit's UploadedFile (io.BytesIO / io.StringIO) without TypeError.
    """
    csv_content = (
        "Timestamp,Src_IP,Dst_IP,Src_Port,Dst_Port,Protocol,Tot_Pkts,Tot_Bytes,SYN_Cnt,ACK_Cnt,Label\n"
        "2026-08-25 10:00:00,192.168.1.10,10.0.0.1,1024,80,6,10,1000,1,9,Benign\n"
        "2026-08-25 10:00:05,192.168.1.20,10.0.0.1,2048,443,6,5,500,1,4,Benign\n"
    )
    
    # 1. Test StringIO buffer
    string_buffer = io.StringIO(csv_content)
    df_str = load_flow_csv(string_buffer)
    assert len(df_str) == 2
    assert "Src_IP" in df_str.columns

    # 2. Test BytesIO buffer (matches Streamlit UploadedFile)
    bytes_buffer = io.BytesIO(csv_content.encode('utf-8'))
    df_bytes = load_flow_csv(bytes_buffer)
    assert len(df_bytes) == 2
    assert "Dst_IP" in df_bytes.columns

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
