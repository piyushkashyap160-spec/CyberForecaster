import pytest
import torch
import numpy as np
from models.lstm_world_model import TemporalLSTMWorldModel

def test_mc_dropout_uncertainty_execution():
    model = TemporalLSTMWorldModel(input_size=23, hidden_size=64, num_layers=2, dropout=0.3, num_stages=6)
    model.eval()

    dummy_input = torch.randn(4, 10, 23)
    mean_state, mean_prob, var_prob, mean_stage = model.forward_with_mc_dropout(dummy_input, num_samples=10)

    assert mean_state.shape == (4, 23)
    assert mean_prob.shape == (4, 1)
    assert var_prob.shape == (4, 1)
    assert mean_stage.shape == (4, 6)
    assert torch.all(var_prob >= 0.0)

def test_mc_dropout_enriched_dimensions():
    model_30d = TemporalLSTMWorldModel(input_size=30, hidden_size=64, num_layers=2, dropout=0.3, num_stages=6)
    model_30d.eval()

    dummy_30d = torch.randn(2, 10, 30)
    mean_s, mean_p, var_p, mean_st = model_30d.forward_with_mc_dropout(dummy_30d, num_samples=5)

    assert mean_s.shape == (2, 30)
    assert mean_p.shape == (2, 1)
    assert var_p.shape == (2, 1)
