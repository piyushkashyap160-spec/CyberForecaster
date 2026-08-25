"""
compare_models.py — 4-Model comparative benchmark evaluation framework.

Models Evaluated:
  1. Model 1: Static Logistic Regression Baseline (Baseline A: uses S(t))
  2. Model 2: Temporal Logistic Regression Baseline (Baseline B: uses flattened [S(t-9)...S(t)])
  3. Model 3: Temporal LSTM World Model (Proposed)
  4. Model 4: Temporal GNN + LSTM World Model (Experimental)

Outputs:
  - experiments/results/demo_results.json (Synthetic Demo Dataset benchmark results)
  - experiments/results/model_comparison.json (Unified side-by-side comparison report)
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
from models.baseline_model import StaticLogisticRegressionBaseline, TemporalLogisticRegressionBaseline
from training.metrics import compute_classification_metrics, compute_regression_metrics
from training.train_temporal_gnn import encode_graph_sequence_batch


def _confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray):
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
    print(f"Comparing 4 models on dataset: {csv_path}...")
    df = load_flow_csv(csv_path)

    window_sec = config['data']['window_seconds']
    seq_len    = config['sequence']['sequence_length']

    states = build_network_states(df, window_seconds=window_sec)
    X, y_state, y_attack, y_stage, _ = create_sequences(states, sequence_length=seq_len)

    total_samples = len(X)
    train_end = int(total_samples * config['training']['train_split'])
    val_end   = train_end + int(total_samples * config['training']['val_split'])

    # Test split
    X_test       = X[val_end:]
    y_state_test = y_state[val_end:]
    y_attack_test = y_attack[val_end:]

    state_scaler = StateScaler()
    state_scaler.load(config['model']['scaler_path'])
    X_test_scaled       = state_scaler.transform(X_test)
    y_state_test_scaled = state_scaler.transform(y_state_test)

    # 1. Model 1: Static Logistic Regression (Baseline A)
    static_baseline = StaticLogisticRegressionBaseline()
    static_baseline.load(config['model']['baseline_path'])
    m1_cls = static_baseline.evaluate(X_test_scaled, y_attack_test)
    m1_pred = static_baseline.predict(X_test_scaled)
    m1_TP, m1_FP, m1_TN, m1_FN = _confusion_matrix_counts(y_attack_test.astype(int), m1_pred)

    # 2. Model 2: Temporal Logistic Regression (Baseline B)
    temp_baseline_path = "models_weights/baseline_temporal_lr.joblib"
    temp_baseline = TemporalLogisticRegressionBaseline()
    if os.path.exists(temp_baseline_path):
        temp_baseline.load(temp_baseline_path)
    else:
        print("Training Temporal LR Baseline (Baseline B) on-the-fly...")
        X_train_scaled = state_scaler.transform(X[:train_end])
        temp_baseline.fit(X_train_scaled, y_attack[:train_end])

    m2_cls = temp_baseline.evaluate(X_test_scaled, y_attack_test)
    m2_pred = temp_baseline.predict(X_test_scaled)
    m2_TP, m2_FP, m2_TN, m2_FN = _confusion_matrix_counts(y_attack_test.astype(int), m2_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # 3. Model 3: Temporal LSTM World Model (Proposed)
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
        p_state_m3, p_attack_m3, _ = lstm_model(X_tensor)
    p_attack_m3_np = p_attack_m3.cpu().numpy().flatten()
    m3_cls = compute_classification_metrics(y_attack_test, p_attack_m3_np)
    m3_reg = compute_regression_metrics(y_state_test_scaled, p_state_m3.cpu().numpy())
    m3_pred = (p_attack_m3_np >= 0.5).astype(int)
    m3_TP, m3_FP, m3_TN, m3_FN = _confusion_matrix_counts(y_attack_test.astype(int), m3_pred)

    # 4. Model 4: Temporal GNN + LSTM World Model (Experimental)
    gnn_model = TemporalGNNWorldModel(
        node_dim=10,
        graph_embed_dim=64,
        state_dim=23,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    gnn_weights_path = "models_weights/temporal_gnn_world_model.pt"
    node_scaler_path = "models_weights/node_feature_scaler.joblib"
    if os.path.exists(gnn_weights_path):
        gnn_model.load_state_dict(torch.load(gnn_weights_path, map_location=device))
        gnn_model.eval()

    node_scaler = NodeFeatureScaler()
    if os.path.exists(node_scaler_path):
        node_scaler.load(node_scaler_path)

    test_graphs = []
    for i in range(val_end, len(states) - seq_len):
        seq_graphs = []
        for j in range(i, i + seq_len):
            df_win = states[j].get('df_window', pd.DataFrame())
            g_dict = build_window_graph(df_win, window_seconds=window_sec)
            seq_graphs.append(g_dict)
        test_graphs.append(seq_graphs)

    p_attack_m4_list, p_state_m4_list = [], []
    BATCH = 16
    with torch.no_grad():
        for start in range(0, len(test_graphs), BATCH):
            end_b = min(start + BATCH, len(test_graphs))
            b_x = X_tensor[start:end_b]

            batch_nodes, batch_edges = [], []
            for seq_graphs in test_graphs[start:end_b]:
                step_nodes, step_edges = [], []
                for g_dict in seq_graphs:
                    x_norm = node_scaler.transform(g_dict['node_features'])
                    step_nodes.append(torch.tensor(x_norm, dtype=torch.float32))
                    step_edges.append(torch.tensor(g_dict['edge_index'], dtype=torch.long))
                batch_nodes.append(step_nodes)
                batch_edges.append(step_edges)

            b_g = encode_graph_sequence_batch(gnn_model, batch_nodes, batch_edges, device)
            p_s, p_a, _ = gnn_model.forward_graph_sequence(b_x, b_g)
            p_attack_m4_list.append(p_a.cpu().numpy())
            p_state_m4_list.append(p_s.cpu().numpy())

    p_attack_m4_np = np.concatenate(p_attack_m4_list).flatten()
    p_state_m4_np  = np.concatenate(p_state_m4_list)
    m4_cls = compute_classification_metrics(y_attack_test, p_attack_m4_np)
    m4_reg = compute_regression_metrics(y_state_test_scaled, p_state_m4_np)
    m4_pred = (p_attack_m4_np >= 0.5).astype(int)
    m4_TP, m4_FP, m4_TN, m4_FN = _confusion_matrix_counts(y_attack_test.astype(int), m4_pred)

    m3_benign_stats = _prob_stats(p_attack_m3_np, y_attack_test.astype(int), 0)
    m3_attack_stats = _prob_stats(p_attack_m3_np, y_attack_test.astype(int), 1)
    m4_benign_stats = _prob_stats(p_attack_m4_np, y_attack_test.astype(int), 0)
    m4_attack_stats = _prob_stats(p_attack_m4_np, y_attack_test.astype(int), 1)

    comparison_results = {
        'evaluation_metadata': {
            'dataset': "Synthetic Demo Dataset (interleaved scenarios)",
            'total_flows': len(df),
            'total_states': len(states),
            'sequence_length': seq_len,
            'test_samples': len(X_test),
            'benign_test_samples': int(np.sum(y_attack_test == 0)),
            'attack_test_samples': int(np.sum(y_attack_test == 1)),
            'decision_threshold': 0.5,
        },
        'Model_1_Static_Logistic_Regression': {
            'Precision': m1_cls['precision'],
            'Recall': m1_cls['recall'],
            'F1_Score': m1_cls['f1'],
            'FPR': m1_cls['fpr'],
            'TP': m1_TP, 'FP': m1_FP, 'TN': m1_TN, 'FN': m1_FN,
            'NextState_MAE': "N/A", 'NextState_MSE': "N/A", 'NextState_RMSE': "N/A"
        },
        'Model_2_Temporal_Logistic_Regression': {
            'Precision': m2_cls['precision'],
            'Recall': m2_cls['recall'],
            'F1_Score': m2_cls['f1'],
            'FPR': m2_cls['fpr'],
            'TP': m2_TP, 'FP': m2_FP, 'TN': m2_TN, 'FN': m2_FN,
            'NextState_MAE': "N/A", 'NextState_MSE': "N/A", 'NextState_RMSE': "N/A"
        },
        'Model_3_Temporal_LSTM_WorldModel': {
            'Precision': m3_cls['precision'],
            'Recall': m3_cls['recall'],
            'F1_Score': m3_cls['f1'],
            'FPR': m3_cls['fpr'],
            'TP': m3_TP, 'FP': m3_FP, 'TN': m3_TN, 'FN': m3_FN,
            'NextState_MAE': m3_reg['mae'],
            'NextState_MSE': m3_reg['mse'],
            'NextState_RMSE': round(float(np.sqrt(m3_reg['mse'])), 6),
            'Prob_Benign_Samples': m3_benign_stats,
            'Prob_Attack_Samples': m3_attack_stats
        },
        'Model_4_Temporal_GNN_WorldModel': {
            'Precision': m4_cls['precision'],
            'Recall': m4_cls['recall'],
            'F1_Score': m4_cls['f1'],
            'FPR': m4_cls['fpr'],
            'TP': m4_TP, 'FP': m4_FP, 'TN': m4_TN, 'FN': m4_FN,
            'NextState_MAE': m4_reg['mae'],
            'NextState_MSE': m4_reg['mse'],
            'NextState_RMSE': round(float(np.sqrt(m4_reg['mse'])), 6),
            'Prob_Benign_Samples': m4_benign_stats,
            'Prob_Attack_Samples': m4_attack_stats
        }
    }

    # Save to both demo_results.json and model_comparison.json
    out_demo = "experiments/results/demo_results.json"
    out_comp = "experiments/results/model_comparison.json"
    os.makedirs(os.path.dirname(out_demo), exist_ok=True)

    with open(out_demo, "w") as f:
        json.dump(comparison_results, f, indent=4)

    with open(out_comp, "w") as f:
        json.dump(comparison_results, f, indent=4)

    print(f"Comparison complete. Saved to {out_demo} and {out_comp}")
    return comparison_results

if __name__ == "__main__":
    compare_all_models()
