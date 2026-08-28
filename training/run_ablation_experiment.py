"""
run_ablation_experiment.py — Real Ablation Experiment: Flow-Only (23-D) vs. Flow + Packet Enriched (30-D).

Scientific Question:
Does genuine packet-level telemetry (real IP TTL distributions, packet size variance, IAT variance, port entropy)
materially improve temporal network-state forecasting and attack early warning?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.pcap_chunked_extractor import ChunkedPCAPExtractor

from preprocessing.csv_loader import load_flow_csv
from preprocessing.state_encoder import encode_window_to_state, STATE_FEATURE_KEYS_FLOW_ONLY, STATE_FEATURE_KEYS_ENRICHED
from preprocessing.window_builder import build_network_states, create_sequences

from preprocessing.scaler import StateScaler
from models.lstm_world_model import TemporalLSTMWorldModel
from forecasting.rollout import perform_k_step_rollout
from forecasting.lead_time import compute_forecast_lead_time

def train_and_eval_model(
    x_train: np.ndarray,
    y_att_train: np.ndarray,
    y_stg_train: np.ndarray,
    x_test: np.ndarray,
    y_att_test: np.ndarray,
    y_stg_test: np.ndarray,
    input_size: int,
    epochs: int = 10
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fit scaler strictly on train set
    scaler = StateScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Prepare targets (Next state target S_{t+1})
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_att_train_tensor = torch.tensor(y_att_train, dtype=torch.float32).unsqueeze(1)
    y_stg_train_tensor = torch.tensor(y_stg_train, dtype=torch.long)

    # Next state targets offset by 1 step
    y_state_train = x_train_scaled[:, -1, :] # (N, input_size)
    y_state_train_tensor = torch.tensor(y_state_train, dtype=torch.float32)

    dataset = TensorDataset(x_train_tensor, y_state_train_tensor, y_att_train_tensor, y_stg_train_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False) # Sequential order

    model = TemporalLSTMWorldModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2, num_stages=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for bx, by_state, by_att, by_stg in loader:
            bx = bx.to(device)
            by_state = by_state.to(device)
            by_att = by_att.to(device)
            by_stg = by_stg.to(device)

            optimizer.zero_grad()
            pred_state, attack_prob, stage_logits = model(bx)

            loss_state = mse_loss_fn(pred_state, by_state)
            loss_att = bce_loss_fn(attack_prob, by_att)
            loss_stg = ce_loss_fn(stage_logits, by_stg)

            loss = loss_state + 1.0 * loss_att + 0.5 * loss_stg
            loss.backward()
            optimizer.step()

    # Evaluation on Test Set
    model.eval()
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_state_scaled, attack_probs, stage_logits = model(x_test_tensor)

    pred_state_scaled_np = pred_state_scaled.cpu().numpy()
    attack_probs_np = attack_probs.cpu().numpy().flatten()
    pred_stages_np = torch.argmax(stage_logits, dim=1).cpu().numpy()

    # Next-state prediction metrics in original physical scale
    y_state_test_orig = x_test[:, -1, :] # Ground truth next state
    pred_state_orig = scaler.inverse_transform(pred_state_scaled_np)

    mae = float(np.mean(np.abs(pred_state_orig - y_state_test_orig)))
    mse = float(np.mean((pred_state_orig - y_state_test_orig) ** 2))
    rmse = float(np.sqrt(mse))

    # Classification metrics
    preds_binary = (attack_probs_np >= 0.5).astype(int)
    y_test_binary = (y_att_test >= 0.5).astype(int)

    tp = int(np.sum((preds_binary == 1) & (y_test_binary == 1)))
    fp = int(np.sum((preds_binary == 1) & (y_test_binary == 0)))
    tn = int(np.sum((preds_binary == 0) & (y_test_binary == 0)))
    fn = int(np.sum((preds_binary == 0) & (y_test_binary == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Forecast Lead Time
    lead_time_metrics = compute_forecast_lead_time(y_test_binary, attack_probs_np, window_seconds=5.0, threshold=0.5)

    # Multi-Horizon RMSE Evaluation (RMSE@1 to RMSE@5)
    horizon_rmses = {}
    for k in range(1, 6):
        rollout_k = []
        for i in range(len(x_test)):
            seq_sample = x_test[i]
            r = perform_k_step_rollout(model, scaler, seq_sample, k_steps=k, device=device)
            rollout_k.append(r[-1]['state_vector'])
        rollout_k_np = np.array(rollout_k) # (N, input_size)
        rmse_k = float(np.sqrt(np.mean((rollout_k_np - y_state_test_orig) ** 2)))
        horizon_rmses[f"RMSE@{k}"] = round(rmse_k, 4)

    return {
        "input_size": input_size,
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "FPR": round(fpr, 4),
        "NextState_MAE": round(mae, 4),
        "NextState_MSE": round(mse, 4),
        "NextState_RMSE": round(rmse, 4),
        "Lead_Time": lead_time_metrics,
        "MultiHorizon_RMSE": horizon_rmses
    }


def main():
    print("=== CYBERFORECASTER REAL ABLATION EXPERIMENT ===")
    print("Comparing Model A (23-D Flow-Only) vs. Model B (30-D Flow + Packet Enriched)...")

    # Extract PCAP packet telemetry records if available
    zip_path = "data/real/cicids2018/pcap.zip"
    pcap_records = []
    if os.path.exists(zip_path):
        extractor = ChunkedPCAPExtractor(window_seconds=5.0)
        pcap_records = extractor.extract_from_zip_stream(zip_path, max_files=10, max_packets_per_file=200000)

    # Load flow CSV
    csv_path = "data/real/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/demo/demo_cicids2018.csv"

    df_flows = load_flow_csv(csv_path, sample_nrows=50000)
    states = build_network_states(df_flows, window_seconds=5.0)


    # Build State DataFrames
    state_records_23 = []
    state_records_30 = []

    for idx, s in enumerate(states):
        pcap_rec = pcap_records[idx] if idx < len(pcap_records) else None
        df_w = s.get('df_window', pd.DataFrame())
        s_data = encode_window_to_state(df_w, window_seconds=5.0, pcap_record=pcap_rec)
        state_records_23.append(s_data)
        state_records_30.append(s_data)


    vecs_23 = np.array([r['vector_flow_only'] for r in state_records_23], dtype=np.float32)
    vecs_30 = np.array([r['vector_enriched'] for r in state_records_30], dtype=np.float32)
    y_atts = np.array([r['is_attack'] for r in state_records_23], dtype=np.int32)
    y_stgs = np.array([r['stage'] for r in state_records_23], dtype=np.int32)

    # Build sequence data (L=10)
    # create_sequences(states, sequence_length=10)
    mock_states_23 = [{'vector': v, 'is_attack': a, 'stage': s, 'timestamp': None} for v, a, s in zip(vecs_23, y_atts, y_stgs)]
    mock_states_30 = [{'vector': v, 'is_attack': a, 'stage': s, 'timestamp': None} for v, a, s in zip(vecs_30, y_atts, y_stgs)]

    x23_seq, _, y_att_seq, y_stg_seq, _ = create_sequences(mock_states_23, sequence_length=10)
    x30_seq, _, _, _, _ = create_sequences(mock_states_30, sequence_length=10)


    # Chronological Split (70% train, 30% test)
    split_idx = int(len(x23_seq) * 0.7)

    x23_train, x23_test = x23_seq[:split_idx], x23_seq[split_idx:]
    x30_train, x30_test = x30_seq[:split_idx], x30_seq[split_idx:]
    y_att_train, y_att_test = y_att_seq[:split_idx], y_att_seq[split_idx:]
    y_stg_train, y_stg_test = y_stg_seq[:split_idx], y_stg_seq[split_idx:]

    print(f"Dataset split: Train samples = {len(x23_train)}, Test samples = {len(x23_test)}")

    # Model A Evaluation (23-D Flow-Only)
    print("\n--- Training Model A (23-D Flow-Only) ---")
    res_A = train_and_eval_model(x23_train, y_att_train, y_stg_train, x23_test, y_att_test, y_stg_test, input_size=23, epochs=10)

    # Model B Evaluation (30-D Flow + Packet Enriched)
    print("\n--- Training Model B (30-D Flow + Packet Enriched) ---")
    res_B = train_and_eval_model(x30_train, y_att_train, y_stg_train, x30_test, y_att_test, y_stg_test, input_size=30, epochs=10)

    ablation_results = {
        "Model_A_Flow_Only_23D": res_A,
        "Model_B_Flow_Packet_Enriched_30D": res_B
    }

    print("\n=== ABLATION RESULTS SUMMARY ===")
    print(json.dumps(ablation_results, indent=4))

    out_file = "experiments/results/ablation_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=4)
    print(f"\nAblation report saved to {out_file}")

if __name__ == "__main__":
    main()
