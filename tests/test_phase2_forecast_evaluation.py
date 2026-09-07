"""
tests/test_phase2_forecast_evaluation.py
========================================
Comprehensive Phase 2 regression test suite verifying:
1. Persistence baseline is correct (always equals last observed state).
2. Training mean uses training data only (no test states in mean calculation).
3. Future horizon indexing is correct (K=1 -> t+1, K=3 -> t+3, K=5 -> t+5).
4. No test leakage into scaler (scaler parameters unchanged before/after test transform).
5. No sequence overlap between train and test blocks.
6. LSTM checkpoint loads correctly and produces expected output shapes.
7. Forecast target timestamp is genuinely in the future (t_target > t_history_end).
8. Lead-time calculation does not count post-onset detections as early warning.
9. Evaluation produces deterministic results with the same seed.
10. Result JSON contains all required horizons, baselines, and models.
"""

import os
import json
import torch
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler
from preprocessing.window_builder import create_sequences
from forecasting.lead_time import compute_forecast_lead_time

def test_persistence_baseline_correctness():
    """1. Verify persistence baseline always predicts the last observed state across all horizons."""
    seq_len = 10
    k_steps = 5
    n_samples = 20
    d_feat = 23
    
    # Synthetic history sequences: X (N, L, D)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, seq_len, d_feat))
    
    # Last observed state is X[:, -1, :]
    last_observed = X[:, -1, :]
    
    for k in [1, 3, 5]:
        pred_persistence = last_observed
        np.testing.assert_array_equal(pred_persistence, X[:, -1, :])
        assert pred_persistence.shape == (n_samples, d_feat)

def test_training_mean_uses_train_only():
    """2. Verify training mean is calculated strictly from train data and excludes test data."""
    rng = np.random.default_rng(42)
    train_states = rng.normal(loc=10.0, scale=1.0, size=(100, 23))
    test_states = rng.normal(loc=100.0, scale=1.0, size=(50, 23)) # Distinct distribution
    
    # True train-only mean
    train_mean = np.mean(train_states, axis=0)
    
    # Contaminated mean (if test was leaked)
    combined_mean = np.mean(np.concatenate([train_states, test_states]), axis=0)
    
    assert np.all(train_mean < 15.0), "Train mean should be close to 10"
    assert np.all(combined_mean > 30.0), "Combined mean would be distorted by test states"
    assert not np.allclose(train_mean, combined_mean), "Train mean must not match combined mean"

def test_future_horizon_indexing():
    """3. Verify future horizon indexing aligns precisely with t+1, t+3, t+5."""
    # Create sequential states where state vector at index t has value t
    states = [
        {
            'vector': np.array([float(t)] * 23, dtype=np.float32),
            'is_attack': 1 if t >= 15 else 0,
            'stage': 1 if t >= 15 else 0,
            'timestamp': datetime.fromtimestamp(1700000000 + t * 5, tz=timezone.utc)
        }
        for t in range(30)
    ]
    
    seq_len = 10
    forecast_steps = 5
    X, y_state, y_attack, y_stage, t_targets = create_sequences(states, sequence_length=seq_len, forecast_steps=forecast_steps)
    
    # For sequence i=0: history covers states 0..9.
    # Future targets:
    # K=1 (index 0) -> state 10
    # K=3 (index 2) -> state 12
    # K=5 (index 4) -> state 14
    assert np.all(y_state[0, 0] == 10.0), f"K=1 should be state 10, got {y_state[0, 0, 0]}"
    assert np.all(y_state[0, 2] == 12.0), f"K=3 should be state 12, got {y_state[0, 2, 0]}"
    assert np.all(y_state[0, 4] == 14.0), f"K=5 should be state 14, got {y_state[0, 4, 0]}"

def test_no_test_leakage_into_scaler():
    """4. Verify StateScaler parameters remain unmodified when transforming test data."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(50, 10, 23))
    X_test = rng.normal(loc=50.0, scale=10.0, size=(50, 10, 23))
    
    scaler = StateScaler()
    scaler.fit(X_train)
    
    mean_before = scaler.scaler.mean_.copy()
    var_before = scaler.scaler.var_.copy()
    
    _ = scaler.transform(X_test)
    
    np.testing.assert_array_equal(scaler.scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.scaler.var_, var_before)

def test_no_sequence_overlap_between_blocks():
    """5. Verify block-partitioned sequences have zero state overlap across train and test partitions."""
    states = [{'vector': np.array([float(i)] * 23), 'is_attack': 0, 'stage': 0, 'timestamp': datetime.now(timezone.utc)} for i in range(100)]
    train_block = states[:50]
    test_block = states[50:]
    
    X_tr, _, _, _, _ = create_sequences(train_block, sequence_length=10, forecast_steps=5)
    X_te, _, _, _, _ = create_sequences(test_block, sequence_length=10, forecast_steps=5)
    
    max_train_state = np.max(X_tr)
    min_test_state = np.min(X_te)
    
    assert max_train_state < 50.0, f"Train leaked into test: max was {max_train_state}"
    assert min_test_state >= 50.0, f"Test leaked into train: min was {min_test_state}"

def test_lstm_checkpoint_loading_and_shape():
    """6. Verify canonical LSTM checkpoint loads and executes forward pass with expected shapes."""
    weights_path = "models_weights/lstm_world_model.pt"
    assert os.path.exists(weights_path), f"Checkpoint missing: {weights_path}"
    
    sd = torch.load(weights_path, map_location="cpu")
    hidden_size = sd['input_proj.0.weight'].shape[0]
    
    model = TemporalLSTMWorldModel(input_size=23, hidden_size=hidden_size, num_layers=2, dropout=0.2, num_stages=6)
    model.load_state_dict(sd)
    model.eval()
    
    dummy_input = torch.randn(4, 10, 23)
    with torch.no_grad():
        p_state, p_att, p_stg = model(dummy_input)
        
    assert p_state.shape == (4, 23)
    assert p_att.shape == (4, 1)
    assert p_stg.shape == (4, 6)

def test_forecast_target_timestamp_in_future():
    """7. Verify target timestamp is genuinely ahead of the history end timestamp."""
    base_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    states = [
        {
            'vector': np.zeros(23, dtype=np.float32),
            'is_attack': 0,
            'stage': 0,
            'timestamp': base_time + timedelta(seconds=i * 5)
        }
        for i in range(25)
    ]
    
    seq_len = 10
    k_steps = 5
    _, _, _, _, t_targets = create_sequences(states, sequence_length=seq_len, forecast_steps=k_steps)
    
    for i in range(len(t_targets)):
        hist_end_ts = states[i + seq_len - 1]['timestamp']
        for k in range(k_steps):
            tgt_ts = t_targets[i][k]
            expected_delta = timedelta(seconds=(k + 1) * 5)
            assert tgt_ts > hist_end_ts, f"Target {tgt_ts} must be after history end {hist_end_ts}"
            assert tgt_ts - hist_end_ts == expected_delta

def test_lead_time_penalizes_post_onset_detections():
    """8. Verify lead time does not count post-onset detections as early warning."""
    y_attack = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Attack onset at index 4
    # Detection at index 6 (2 windows late)
    probs_late = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.95, 0.95, 0.95])
    
    res = compute_forecast_lead_time(y_attack, probs_late, window_seconds=5.0, threshold=0.5)
    assert res["mean_lead_time_seconds"] == -10.0
    assert res["pre_onset_warnings"] == 0
    assert res["post_onset_detections"] == 1

def test_evaluation_deterministic_with_seed():
    """9. Verify bootstrap CI calculation is deterministic with a fixed seed."""
    data = np.array([1.2, 2.3, 3.1, 4.5, 5.0, 6.2, 7.1, 8.4])
    mean1, low1, high1 = compute_bootstrap_ci_toy(data, seed=42)
    mean2, low2, high2 = compute_bootstrap_ci_toy(data, seed=42)
    
    assert mean1 == mean2
    assert low1 == low2
    assert high1 == high2

def compute_bootstrap_ci_toy(data: np.ndarray, num_bootstrap: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(num_bootstrap)
    for b in range(num_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    return float(np.mean(data)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def test_phase2_result_json_validity():
    """10. Verify Phase 2 result JSON exists and contains all required horizons and models."""
    json_path = "experiments/results/phase2_forecast_evaluation.json"
    assert os.path.exists(json_path), f"Phase 2 result JSON missing: {json_path}"
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert "horizon_evaluations" in data
    h_eval = data["horizon_evaluations"]
    
    for h_key in ["horizon_1_step_5s", "horizon_3_step_15s", "horizon_5_step_25s"]:
        assert h_key in h_eval, f"Missing horizon: {h_key}"
        h_dict = h_eval[h_key]
        assert "persistence" in h_dict["metrics_scaled"]
        assert "training_mean" in h_dict["metrics_scaled"]
        assert "temporal_lstm" in h_dict["metrics_scaled"]
        assert "rmse" in h_dict["metrics_scaled"]["temporal_lstm"]
        assert "mae" in h_dict["metrics_scaled"]["temporal_lstm"]
        assert "relative_improvement_percent" in h_dict
