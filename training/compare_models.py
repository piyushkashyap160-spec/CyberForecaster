"""
compare_models.py — 3-Model comparative benchmark evaluation.

Design invariants:
  - Model 3 (Temporal GNN) uses model.graph_encoder for encoding — NOT a separate instance.
  - NodeFeatureScaler loaded from checkpoint; only transform() applied to val/test data.
  - Confusion matrices are reported for all models.
  - No threshold tuning: default 0.5 used throughout.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.graph_builder import build_window_graph
from preprocessing.scaler import StateScaler
from preprocessing.node_feature_scaler import NodeFeatureScaler
from models.lstm_world_model import TemporalLSTMWorldModel
from models.temporal_gnn_world_model import TemporalGNNWorldModel
from models.baseline_model import LogisticRegressionBaseline
from training.metrics import compute_classification_metrics, compute_regression_metrics
from training.train_temporal_gnn import encode_graph_sequence_batch


def _confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns TP, FP, TN, FN for binary predictions."""
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, FP, TN, FN


def _prob_stats(probs: np.ndarray, labels: np.ndarray, label_val: int) -> dict:
    subset = probs[labels == label_val]
    if len(subset) == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean":   round(float(np.mean(subset)), 4),
        "median": round(float(np.median(subset)), 4),
        "min":    round(float(np.min(subset)), 4),
        "max":    round(float(np.max(subset)), 4),
    }


def compare_all_models(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    csv_path = config['data']['demo_csv_path']
    print(f"Comparing models on dataset: {csv_path}...")
    df = load_flow_csv(csv_path)

    window_sec = config['data']['window_seconds']
    seq_len    = config['sequence']['sequence_length']

    states = build_network_states(df, window_seconds=window_sec)
    X, y_state, y_attack, y_stage, _ = create_sequences(states, sequence_length=seq_len)

    total_samples = len(X)
    train_end = int(total_samples * config['training']['train_split'])
    val_end   = train_end + int(total_samples * config['training']['val_split'])

    # --- Test split ---
    X_test       = X[val_end:]
    y_state_test = y_state[val_end:]
    y_attack_test = y_attack[val_end:]

    state_scaler = StateScaler()
    state_scaler.load(config['model']['scaler_path'])
    X_test_scaled      = state_scaler.transform(X_test)
    y_state_test_scaled = state_scaler.transform(y_state_test)

    # --- Model 1: Logistic Regression ---
    baseline = LogisticRegressionBaseline()
    baseline.load(config['model']['baseline_path'])
    m1_cls = baseline.evaluate(X_test_scaled, y_attack_test)
    baseline_probs = baseline.predict_proba(X_test_scaled)
    m1_pred = (baseline_probs >= 0.5).astype(int)
    m1_TP, m1_FP, m1_TN, m1_FN = _confusion_matrix_counts(y_attack_test.astype(int), m1_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # --- Model 2: Baseline Temporal LSTM World Model ---
    lstm_model = TemporalLSTMWorldModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)
    lstm_model.load_state_dict(torch.load(config['model']['weights_path'], map_location=device))
    lstm_model.eval()

    with torch.no_grad():
        p_state_m2, p_attack_m2, _ = lstm_model(X_tensor)
    p_attack_m2_np = p_attack_m2.cpu().numpy().flatten()
    m2_cls = compute_classification_metrics(y_attack_test, p_attack_m2_np)
    m2_reg = compute_regression_metrics(y_state_test_scaled, p_state_m2.cpu().numpy())
    m2_pred = (p_attack_m2_np >= 0.5).astype(int)
    m2_TP, m2_FP, m2_TN, m2_FN = _confusion_matrix_counts(y_attack_test.astype(int), m2_pred)

    # --- Model 3: Temporal GNN + LSTM World Model ---
    gnn_model = TemporalGNNWorldModel(
        node_dim=10,
        graph_embed_dim=64,
        state_dim=23,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    gnn_weights_path   = "models_weights/temporal_gnn_world_model.pt"
    node_scaler_path   = "models_weights/node_feature_scaler.joblib"
    gnn_trained = False

    if os.path.exists(gnn_weights_path):
        gnn_model.load_state_dict(torch.load(gnn_weights_path, map_location=device))
        gnn_model.eval()
        gnn_trained = True
        print("Loaded Temporal GNN checkpoint (includes trained GraphEncoder).")
    else:
        print(f"WARNING: GNN checkpoint not found at {gnn_weights_path}. Using random weights.")

    # Load NodeFeatureScaler fitted on training data
    node_scaler = NodeFeatureScaler()
    if os.path.exists(node_scaler_path):
        node_scaler.load(node_scaler_path)
    else:
        print("WARNING: NodeFeatureScaler not found — fitting on training data now (not recommended for eval).")
        train_graphs_all = []
        for i in range(train_end):
            df_win = states[i].get('df_window', pd.DataFrame())
            g_dict = build_window_graph(df_win, window_seconds=window_sec)
            train_graphs_all.append(g_dict['node_features'])
        node_scaler.fit(train_graphs_all)

    # Build per-sequence graph dicts for the test split
    print("Building test graph sequences...")
    test_graphs = []
    for i in range(val_end, len(states) - seq_len):
        seq_graphs = []
        for j in range(i, i + seq_len):
            df_win = states[j].get('df_window', pd.DataFrame())
            g_dict = build_window_graph(df_win, window_seconds=window_sec)
            seq_graphs.append(g_dict)
        test_graphs.append(seq_graphs)

    # Encode using the trained model's own graph_encoder (NOT a new random instance)
    # Process in batches of 16 to avoid memory issues on large test sets
    p_attack_m3_list = []
    p_state_m3_list  = []
    BATCH = 16
    with torch.no_grad():
        for start in range(0, len(test_graphs), BATCH):
            end_b   = min(start + BATCH, len(test_graphs))
            b_x     = X_tensor[start:end_b]

            batch_nodes = []
            batch_edges = []
            for seq_graphs in test_graphs[start:end_b]:
                step_nodes, step_edges = [], []
                for g_dict in seq_graphs:
                    x_norm = node_scaler.transform(g_dict['node_features'])
                    step_nodes.append(torch.tensor(x_norm,                dtype=torch.float32))
                    step_edges.append(torch.tensor(g_dict['edge_index'],  dtype=torch.long))
                batch_nodes.append(step_nodes)
                batch_edges.append(step_edges)

            # encode_graph_sequence_batch calls model.graph_encoder — trained weights
            b_g = encode_graph_sequence_batch(gnn_model, batch_nodes, batch_edges, device)
            p_s, p_a, _ = gnn_model.forward_graph_sequence(b_x, b_g)
            p_attack_m3_list.append(p_a.cpu().numpy())
            p_state_m3_list.append(p_s.cpu().numpy())

    p_attack_m3_np = np.concatenate(p_attack_m3_list).flatten()
    p_state_m3_np  = np.concatenate(p_state_m3_list)
    m3_cls = compute_classification_metrics(y_attack_test, p_attack_m3_np)
    m3_reg = compute_regression_metrics(y_state_test_scaled, p_state_m3_np)
    m3_pred = (p_attack_m3_np >= 0.5).astype(int)
    m3_TP, m3_FP, m3_TN, m3_FN = _confusion_matrix_counts(y_attack_test.astype(int), m3_pred)

    # Probability distribution stats
    m3_benign_stats = _prob_stats(p_attack_m3_np, y_attack_test.astype(int), 0)
    m3_attack_stats = _prob_stats(p_attack_m3_np, y_attack_test.astype(int), 1)
    m2_benign_stats = _prob_stats(p_attack_m2_np, y_attack_test.astype(int), 0)
    m2_attack_stats = _prob_stats(p_attack_m2_np, y_attack_test.astype(int), 1)

    comparison_results = {
        'evaluation_metadata': {
            'dataset': "Synthetic Demo Dataset (interleaved scenarios)",
            'test_samples': len(X_test),
            'benign_test_samples': int(np.sum(y_attack_test == 0)),
            'attack_test_samples': int(np.sum(y_attack_test == 1)),
            'decision_threshold': 0.5,
            'gnn_checkpoint_trained': gnn_trained,
        },
        'Model_1_Logistic_Regression_Baseline': {
            'Precision':    m1_cls['precision'],
            'Recall':       m1_cls['recall'],
            'F1_Score':     m1_cls['f1'],
            'FPR':          m1_cls['fpr'],
            'TP': m1_TP, 'FP': m1_FP, 'TN': m1_TN, 'FN': m1_FN,
            'NextState_MAE': "N/A", 'NextState_MSE': "N/A", 'NextState_RMSE': "N/A",
        },
        'Model_2_Temporal_LSTM_WorldModel': {
            'Precision':    m2_cls['precision'],
            'Recall':       m2_cls['recall'],
            'F1_Score':     m2_cls['f1'],
            'FPR':          m2_cls['fpr'],
            'TP': m2_TP, 'FP': m2_FP, 'TN': m2_TN, 'FN': m2_FN,
            'NextState_MAE':  m2_reg['mae'],
            'NextState_MSE':  m2_reg['mse'],
            'NextState_RMSE': round(float(np.sqrt(m2_reg['mse'])), 6),
            'Prob_Benign_Samples': m2_benign_stats,
            'Prob_Attack_Samples': m2_attack_stats,
        },
        'Model_3_Temporal_GNN_WorldModel': {
            'Precision':    m3_cls['precision'],
            'Recall':       m3_cls['recall'],
            'F1_Score':     m3_cls['f1'],
            'FPR':          m3_cls['fpr'],
            'TP': m3_TP, 'FP': m3_FP, 'TN': m3_TN, 'FN': m3_FN,
            'NextState_MAE':  m3_reg['mae'],
            'NextState_MSE':  m3_reg['mse'],
            'NextState_RMSE': round(float(np.sqrt(m3_reg['mse'])), 6),
            'Prob_Benign_Samples': m3_benign_stats,
            'Prob_Attack_Samples': m3_attack_stats,
        }
    }

    out_path = "experiments/results/benchmark_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison_results, f, indent=4)

    print(f"\nModel comparison complete. Report saved to {out_path}")
    print(json.dumps(comparison_results, indent=2))
    return comparison_results


if __name__ == "__main__":
    compare_all_models()
