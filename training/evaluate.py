import os
import sys
import json
import yaml
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.scaler import StateScaler
from preprocessing.state_encoder import STATE_FEATURE_KEYS
from models.lstm_world_model import TemporalLSTMWorldModel
from models.baseline_model import LogisticRegressionBaseline
from forecasting.rollout import perform_k_step_rollout
from training.metrics import compute_classification_metrics, compute_regression_metrics

STAGE_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command & Control",
    5: "Exfiltration"
}

def evaluate_models(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    csv_path = config['data']['demo_csv_path']
    print(f"Loading evaluation dataset from {csv_path}...")
    df = load_flow_csv(csv_path)

    window_sec = config['data']['window_seconds']
    seq_len = config['sequence']['sequence_length']
    k_horizon = config['sequence']['forecast_horizon']

    states = build_network_states(df, window_seconds=window_sec)
    X, y_state, y_attack, y_stage, timestamps = create_sequences(states, sequence_length=seq_len)

    total_samples = len(X)
    train_end = int(total_samples * config['training']['train_split'])
    val_end = train_end + int(total_samples * config['training']['val_split'])

    # Test Split (Chronological 15%)
    X_test = X[val_end:]
    y_state_test = y_state[val_end:]
    y_attack_test = y_attack[val_end:]
    y_stage_test = y_stage[val_end:]
    ts_test = timestamps[val_end:]

    # Test set composition
    n_test_samples = len(X_test)
    n_benign = int(np.sum(y_attack_test == 0))
    n_attack = int(np.sum(y_attack_test == 1))

    print("=" * 70)
    print(f"TEST SET COMPOSITION ({n_test_samples} total sequences):")
    print(f"  - Benign Windows (Class 0): {n_benign} ({n_benign/n_test_samples*100:.1f}%)")
    print(f"  - Attack Windows (Class 1): {n_attack} ({n_attack/n_test_samples*100:.1f}%)")
    print("=" * 70)

    scaler = StateScaler()
    scaler.load(config['model']['scaler_path'])
    X_test_scaled = scaler.transform(X_test)
    y_state_test_scaled = scaler.transform(y_state_test)

    # 1. Baseline Model Evaluation
    baseline = LogisticRegressionBaseline()
    baseline.load(config['model']['baseline_path'])
    baseline_metrics = baseline.evaluate(X_test_scaled, y_attack_test)

    # 2. PyTorch Temporal World Model Evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalLSTMWorldModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    model.load_state_dict(torch.load(config['model']['weights_path'], map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        p_state, p_attack, p_stage = model(X_tensor)
        p_state_np = p_state.cpu().numpy()
        p_attack_np = p_attack.cpu().numpy().flatten()
        p_stage_np = torch.argmax(p_stage, dim=1).cpu().numpy()

    world_model_cls_metrics = compute_classification_metrics(y_attack_test, p_attack_np)
    world_model_reg_metrics = compute_regression_metrics(y_state_test_scaled, p_state_np)

    # Calculate RMSE for Next State Prediction
    rmse_state = float(np.sqrt(world_model_reg_metrics['mse']))

    # Representative feature regression MAE
    p_state_orig = scaler.inverse_transform(p_state_np)
    feat_mae = {}
    for idx, name in enumerate(STATE_FEATURE_KEYS):
        mae_f = float(np.mean(np.abs(p_state_orig[:, idx] - y_state_test[:, idx])))
        feat_mae[name] = round(mae_f, 4)

    # 3. Horizon-Wise K-Step Forecast Evaluation (t+1 ... t+5)
    horizon_metrics = {}
    for k in range(1, k_horizon + 1):
        k_probs = []
        k_targets = []
        for idx in range(len(X_test) - k):
            hist_seq = X_test[idx] # Original scale
            rollout = perform_k_step_rollout(model, scaler, hist_seq, k_steps=k, device=device)
            k_probs.append(rollout[k-1]['attack_probability'])
            k_targets.append(y_attack_test[idx + k])

        if k_targets:
            h_metrics = compute_classification_metrics(np.array(k_targets), np.array(k_probs))
            horizon_metrics[f"t+{k}"] = {
                'Precision': h_metrics['precision'],
                'Recall': h_metrics['recall'],
                'F1_Score': h_metrics['f1'],
                'FPR': h_metrics['fpr']
            }

    # 4. Forecast Lead Time Calculation
    warning_thresh = config['forecasting']['warning_threshold']
    first_warning_ts = None
    first_attack_ts = None

    for idx in range(len(y_attack_test)):
        if first_warning_ts is None and p_attack_np[idx] >= warning_thresh:
            first_warning_ts = idx
        if first_attack_ts is None and y_attack_test[idx] == 1:
            first_attack_ts = idx

    lead_time_status = "Lead time could not be reliably measured on this dataset."
    lead_time_seconds = 0.0
    if first_warning_ts is not None and first_attack_ts is not None:
        lead_steps = first_attack_ts - first_warning_ts
        lead_time_seconds = max(0.0, float(lead_steps * window_sec))
        lead_time_status = f"{lead_time_seconds:.1f} seconds ({int(lead_steps)} windows lead time)"

    # 5. MITRE ATT&CK Stage Multi-Class Evaluation
    stage_prec = float(precision_score(y_stage_test, p_stage_np, average='weighted', zero_division=0))
    stage_rec = float(recall_score(y_stage_test, p_stage_np, average='weighted', zero_division=0))
    stage_f1 = float(f1_score(y_stage_test, p_stage_np, average='weighted', zero_division=0))
    stage_cm = confusion_matrix(y_stage_test, p_stage_np, labels=list(range(6))).tolist()

    # Compile Benchmark Results JSON
    benchmark_results = {
        'dataset_info': {
            'dataset_name': "Synthetic Demo Dataset (interleaved scenarios)",
            'note': "Synthetic demonstration data — not a substitute for real-world evaluation.",
            'total_test_samples': n_test_samples,
            'benign_samples': n_benign,
            'attack_samples': n_attack,
            'class_distribution': {
                'Benign_Ratio': round(n_benign / max(1, n_test_samples), 4),
                'Attack_Ratio': round(n_attack / max(1, n_test_samples), 4)
            }
        },
        'Baseline_LogisticRegression': {
            'Precision': baseline_metrics['precision'],
            'Recall': baseline_metrics['recall'],
            'F1_Score': baseline_metrics['f1'],
            'FPR': baseline_metrics['fpr'],
            'Confusion_Matrix': {
                'TP': baseline_metrics['tp'],
                'FP': baseline_metrics['fp'],
                'TN': baseline_metrics['tn'],
                'FN': baseline_metrics['fn']
            }
        },
        'Temporal_LSTM_WorldModel': {
            'Precision': world_model_cls_metrics['precision'],
            'Recall': world_model_cls_metrics['recall'],
            'F1_Score': world_model_cls_metrics['f1'],
            'FPR': world_model_cls_metrics['fpr'],
            'Confusion_Matrix': {
                'TP': world_model_cls_metrics['tp'],
                'FP': world_model_cls_metrics['fp'],
                'TN': world_model_cls_metrics['tn'],
                'FN': world_model_cls_metrics['fn']
            },
            'NextState_MAE': world_model_reg_metrics['mae'],
            'NextState_MSE': world_model_reg_metrics['mse'],
            'NextState_RMSE': round(rmse_state, 6),
            'Representative_Feature_MAE': {
                'syn_ratio': feat_mae.get('syn_ratio', 0.0),
                'port_entropy': feat_mae.get('port_entropy', 0.0),
                'total_bytes': feat_mae.get('total_bytes', 0.0),
                'mean_IAT': feat_mae.get('mean_IAT', 0.0)
            },
            'Forecast_Lead_Time': lead_time_status,
            'Forecast_Lead_Time_Seconds': lead_time_seconds
        },
        'Horizon_Wise_Forecasting': horizon_metrics,
        'Stage_Prediction_MITRE': {
            'Weighted_Precision': round(stage_prec, 4),
            'Weighted_Recall': round(stage_rec, 4),
            'Weighted_F1': round(stage_f1, 4),
            'Confusion_Matrix': stage_cm
        }
    }

    results_path = "models_weights/benchmark_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)

    print(f"Evaluation complete. Benchmark report saved to {results_path}")
    print(json.dumps(benchmark_results, indent=2))
    return benchmark_results

if __name__ == "__main__":
    evaluate_models()
