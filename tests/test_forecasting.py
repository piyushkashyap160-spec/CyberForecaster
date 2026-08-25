import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler
from forecasting.rollout import perform_k_step_rollout
from forecasting.risk_engine import calculate_network_risk

def test_k_step_rollout():
    model = TemporalLSTMWorldModel(input_size=23)
    scaler = StateScaler()
    dummy_X = np.random.randn(20, 23).astype(np.float32)
    scaler.fit(dummy_X)

    hist_seq = dummy_X[:10]
    rollout = perform_k_step_rollout(model, scaler, hist_seq, k_steps=5)

    assert len(rollout) == 5
    for i, r in enumerate(rollout):
        assert r['step_index'] == i + 1
        assert 0.0 <= r['attack_probability'] <= 1.0
        assert 0 <= r['predicted_stage_id'] <= 5
        assert len(r['state_vector']) == 23

def test_risk_engine():
    dummy_rollout = [
        {'attack_probability': 0.20, 'predicted_stage_id': 0},
        {'attack_probability': 0.50, 'predicted_stage_id': 1},
        {'attack_probability': 0.78, 'predicted_stage_id': 2},
    ]
    risk = calculate_network_risk(dummy_rollout, warning_threshold=0.70)
    assert risk['alert_triggered'] is True
    assert risk['max_future_risk'] == 0.78
    assert risk['threat_level'] == "HIGH WARNING"
