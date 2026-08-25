import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_world_model import TemporalLSTMWorldModel
from models.baseline_model import LogisticRegressionBaseline

def test_lstm_world_model_forward():
    model = TemporalLSTMWorldModel(input_size=23, hidden_size=64, num_layers=2)
    x = torch.randn(8, 10, 23)
    p_state, p_attack, p_stage = model(x)
    assert p_state.shape == (8, 23)
    assert p_attack.shape == (8, 1)
    assert p_stage.shape == (8, 6)

def test_baseline_logistic_regression():
    baseline = LogisticRegressionBaseline()
    X = np.random.randn(40, 10, 23).astype(np.float32)
    y = np.random.randint(0, 2, size=(40,))
    baseline.fit(X, y)
    probs = baseline.predict_proba(X)
    assert len(probs) == 40
    assert np.all((probs >= 0.0) & (probs <= 1.0))
