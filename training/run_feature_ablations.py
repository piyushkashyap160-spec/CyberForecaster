"""
run_feature_ablations.py — Diagnostic Feature Ablation Investigation (Experiments A through G).

Investigates which specific PCAP packet feature impacts temporal forecasting model performance
when aligned using timestamp-keyed matching with EST timezone correction (UTC - 5 hours).
"""

import os
import sys
import json
import zipfile
import dpkt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.state_encoder import encode_window_to_state, STATE_FEATURE_KEYS_FLOW_ONLY
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.scaler import StateScaler
from models.lstm_world_model import TemporalLSTMWorldModel
from forecasting.rollout import perform_k_step_rollout
from forecasting.lead_time import compute_forecast_lead_time

def build_timestamp_keyed_pcap_index(zip_path: str, max_files: int = 15) -> dict:
    """
    Parses PCAP files, applies 5-hour EST timezone offset (ts_utc - 18000),
    and keys packet telemetry by 5-second epoch bin.
    """
    if not os.path.exists(zip_path):
        return {}

    window_buckets = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        pcap_names = [n for n in z.namelist() if not n.endswith('/') and 'pcap' in n.lower()][:max_files]
        for name in pcap_names:
            try:
                with z.open(name) as f:
                    pcap_reader = dpkt.pcap.Reader(f)
                    for ts_utc, buf in pcap_reader:
                        # 5-hour EST adjustment (UTC - 18000)
                        ts_est = ts_utc - 18000.0
                        epoch_bin = int(ts_est // 5.0) * 5

                        if epoch_bin not in window_buckets:
                            window_buckets[epoch_bin] = {
                                "lengths": [], "ttls": [], "dst_ports": {}
                            }

                        b = window_buckets[epoch_bin]
                        b["lengths"].append(len(buf))

                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                            if isinstance(eth.data, dpkt.ip.IP):
                                ip = eth.data
                                b["ttls"].append(ip.ttl)
                                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                                    dp = ip.data.dport
                                    b["dst_ports"][dp] = b["dst_ports"].get(dp, 0) + 1
                        except Exception:
                            pass
            except Exception:
                continue

    pcap_index = {}
    for epoch_bin, b in window_buckets.items():
        ttls = b["ttls"] if b["ttls"] else [64.0]
        lens = b["lengths"] if b["lengths"] else [100.0]

        # Port entropy
        p_counts = list(b["dst_ports"].values())
        tot_p = sum(p_counts)
        entropy = 0.0
        if tot_p > 0:
            for c in p_counts:
                p = c / tot_p
                entropy -= p * np.log2(p)

        pcap_index[epoch_bin] = {
            "pcap_ttl_mean": round(float(np.mean(ttls)), 2),
            "pcap_ttl_var": round(float(np.var(ttls)), 2),
            "pcap_ttl_min": float(np.min(ttls)),
            "pcap_ttl_max": float(np.max(ttls)),
            "pcap_pkt_size_var": round(float(np.var(lens)), 2),
            "pcap_iat_var": 0.01,
            "pcap_port_entropy": round(float(entropy), 4),
            "pcap_enriched_flag": 1.0
        }
    return pcap_index


def train_eval_ablation(x_train: np.ndarray, y_att_train: np.ndarray, x_test: np.ndarray, y_att_test: np.ndarray, epochs: int = 8) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = x_train.shape[2]

    scaler = StateScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_tr_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_st_tr_t = torch.tensor(x_train_scaled[:, -1, :], dtype=torch.float32)
    y_att_tr_t = torch.tensor(y_att_train, dtype=torch.float32).unsqueeze(1)
    y_stg_tr_t = torch.zeros((len(x_train),), dtype=torch.long)

    ds = TensorDataset(x_tr_t, y_st_tr_t, y_att_tr_t, y_stg_tr_t)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = TemporalLSTMWorldModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2, num_stages=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        for bx, by_st, by_att, by_stg in loader:
            bx, by_st, by_att = bx.to(device), by_st.to(device), by_att.to(device)
            opt.zero_grad()
            pred_st, p_att, _ = model(bx)
            loss = mse_fn(pred_st, by_st) + bce_fn(p_att, by_att)
            loss.backward()
            opt.step()

    model.eval()
    x_te_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_st_scaled, attack_probs, _ = model(x_te_t)

    pred_st_scaled_np = pred_st_scaled.cpu().numpy()
    probs_np = attack_probs.cpu().numpy().flatten()
    preds_bin = (probs_np >= 0.5).astype(int)
    y_test_bin = (y_att_test >= 0.5).astype(int)

    tp = int(np.sum((preds_bin == 1) & (y_test_bin == 1)))
    fp = int(np.sum((preds_bin == 1) & (y_test_bin == 0)))
    tn = int(np.sum((preds_bin == 0) & (y_test_bin == 0)))
    fn = int(np.sum((preds_bin == 0) & (y_test_bin == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Scaled Z-score State RMSE (unit normalized across dimensions)
    y_state_test_scaled = x_test_scaled[:, -1, :]
    scaled_rmse = float(np.sqrt(np.mean((pred_st_scaled_np - y_state_test_scaled) ** 2)))

    # Forecast Lead Time
    lead_time = compute_forecast_lead_time(y_test_bin, probs_np, window_seconds=5.0)

    # Multi-Horizon Scaled RMSE evaluated on sample of 100 test windows for fast execution
    horizon_rmses = {}
    sample_indices = np.linspace(0, len(x_test) - 1, num=min(100, len(x_test)), dtype=int)
    for k in range(1, 6):
        roll_k = []
        for i in sample_indices:
            r = perform_k_step_rollout(model, scaler, x_test[i], k_steps=k, device=device)
            roll_k.append(r[-1]['state_vector'])
        roll_k_scaled = scaler.transform(np.array(roll_k))
        y_sampled_scaled = y_state_test_scaled[sample_indices]
        h_rmse = float(np.sqrt(np.mean((roll_k_scaled - y_sampled_scaled) ** 2)))
        horizon_rmses[f"RMSE@{k}"] = round(h_rmse, 4)


    return {
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "FPR": round(fpr, 4),
        "State_RMSE_Scaled": round(scaled_rmse, 4),
        "Mean_Lead_Time": lead_time["mean_lead_time_seconds"],
        "Horizon_RMSE": horizon_rmses
    }


def main():
    print("=== CYBERFORECASTER FEATURE ABLATION INVESTIGATION (A through G) ===")
    csv_path = "data/real/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/demo/demo_cicids2018.csv"

    df_flows = load_flow_csv(csv_path, sample_nrows=10000)
    states = build_network_states(df_flows, window_seconds=5.0)

    zip_path = "data/real/cicids2018/pcap.zip"
    pcap_index = build_timestamp_keyed_pcap_index(zip_path, max_files=3)


    records = []
    for s in states:
        ts = s.get('timestamp')
        epoch_bin = int(ts.timestamp() // 5.0) * 5 if ts else 0
        p_rec = pcap_index.get(epoch_bin)
        df_w = s.get('df_window', pd.DataFrame())
        rec = encode_window_to_state(df_w, window_seconds=5.0, pcap_record=p_rec)
        records.append(rec)

    y_atts = np.array([r['is_attack'] for r in records], dtype=np.int32)

    # Prepare feature sets A through G
    base_23 = np.array([r['vector_flow_only'] for r in records], dtype=np.float32)
    s_dicts = [r['state_dict'] for r in records]

    feat_sets = {
        "A_23D_Flow_Only": base_23,
        "B_23D_plus_TTL_mean": np.column_stack([base_23, [d['pcap_ttl_mean'] for d in s_dicts]]),
        "C_23D_plus_TTL_var": np.column_stack([base_23, [d['pcap_ttl_var'] for d in s_dicts]]),
        "D_23D_plus_pkt_size_var": np.column_stack([base_23, [d['pcap_pkt_size_var'] for d in s_dicts]]),
        "E_23D_plus_pkt_iat_var": np.column_stack([base_23, [d['pcap_iat_var'] for d in s_dicts]]),
        "F_23D_plus_port_entropy": np.column_stack([base_23, [d['pcap_port_entropy'] for d in s_dicts]]),
        "G_30D_All_Packet_Feats": np.array([r['vector_enriched'] for r in records], dtype=np.float32)
    }

    results = {}
    for name, feat_mat in feat_sets.items():
        mock_st = [{'vector': v, 'is_attack': a, 'stage': 0, 'timestamp': None} for v, a in zip(feat_mat, y_atts)]
        x_seq, _, y_att_seq, _, _ = create_sequences(mock_st, sequence_length=10)

        split = int(len(x_seq) * 0.7)
        x_tr, x_te = x_seq[:split], x_seq[split:]
        y_tr, y_te = y_att_seq[:split], y_att_seq[split:]

        res = train_eval_ablation(x_tr, y_tr, x_te, y_te, epochs=8)
        results[name] = res
        print(f"[{name}] F1: {res['F1']} | Recall: {res['Recall']} | LeadTime: {res['Mean_Lead_Time']}s | Scaled RMSE: {res['State_RMSE_Scaled']}")

    out_file = "experiments/results/feature_ablations.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAblation results saved to {out_file}")

if __name__ == "__main__":
    main()
