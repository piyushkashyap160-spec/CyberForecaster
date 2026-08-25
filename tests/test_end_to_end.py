import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.scaler import StateScaler
from models.lstm_world_model import TemporalLSTMWorldModel
from forecasting.rollout import perform_k_step_rollout
from explainability.shap_explainer import ModelExplainer
from training.evaluate import evaluate_models

def test_full_pipeline_smoke():
    demo_csv = "data/demo/demo_cicids2018.csv"
    assert os.path.exists(demo_csv)

    # 1. Load Data & Build State Sequences
    df = load_flow_csv(demo_csv)
    states = build_network_states(df, window_seconds=5.0)
    X, y_state, y_attack, y_stage, timestamps = create_sequences(states, sequence_length=10)

    # 2. Scaler Fit
    scaler = StateScaler()
    scaler.fit(X[:50])

    # 3. Model Forward Pass
    model = TemporalLSTMWorldModel(input_size=23)
    seq_sample = X[0]
    rollout = perform_k_step_rollout(model, scaler, seq_sample, k_steps=3)
    assert len(rollout) == 3

    # 4. Explainability Pass
    explainer = ModelExplainer(model, scaler)
    explanation = explainer.explain_instance(seq_sample)
    assert 'top_features' in explanation
    assert len(explanation['top_features']) <= 10

    # 5. Evaluate Benchmark Execution
    benchmarks = evaluate_models()
    assert 'Temporal_LSTM_WorldModel' in benchmarks
    assert 'Baseline_LogisticRegression' in benchmarks
