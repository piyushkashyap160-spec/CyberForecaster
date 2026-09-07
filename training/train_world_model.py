import os
import sys
import yaml
import random

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.scaler import StateScaler
from models.lstm_world_model import TemporalLSTMWorldModel

import argparse
from preprocessing.pcap_parser import parse_pcap_file

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(config_path: str = "config.yaml", pcap_path: str = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['training']['seed'])

    if pcap_path:
        print(f"Loading PCAP dataset with real packet-level features from {pcap_path}...")
        df = parse_pcap_file(pcap_path)
        weights_path = "models_weights/lstm_world_model_pcap.pt"
    else:
        csv_path = config['data']['demo_csv_path']
        print(f"Loading flow dataset from {csv_path}...")
        df = load_flow_csv(csv_path)
        weights_path = config['model']['weights_path']

    window_sec = config['data']['window_seconds']
    seq_len = config['sequence']['sequence_length']
    forecast_steps = config.get('sequence', {}).get('forecast_horizon', 5)

    print(f"Constructing network state time windows ({window_sec}s)...")
    states = build_network_states(df, window_seconds=window_sec)
    print(f"Total time windows generated: {len(states)}")

    # Strict Zero-Leakage: Chronologically partition RAW states BEFORE creating rolling sequences
    total_states = len(states)
    train_split = config['training'].get('train_split', 0.70)
    val_split = config['training'].get('val_split', 0.15)

    train_end = int(total_states * train_split)
    val_end = train_end + int(total_states * val_split)

    train_states = states[:train_end]
    val_states = states[train_end:val_end]
    test_states = states[val_end:]
    print(f"State blocks -> Train: {len(train_states)}, Val: {len(val_states)}, Test: {len(test_states)}")

    min_required_states = seq_len + forecast_steps
    if len(train_states) < min_required_states:
        print(f"Warning: Insufficient train states ({len(train_states)} < {min_required_states}) for sequence length {seq_len}. Skipping training.")
        return

    print(f"Creating multi-step sequences within independent temporal blocks (unroll_steps={forecast_steps})...")
    X_train, y_state_train, y_attack_train, y_stage_train, _ = create_sequences(
        train_states, sequence_length=seq_len, forecast_steps=forecast_steps
    )

    if len(val_states) >= min_required_states:
        X_val, y_state_val, y_attack_val, y_stage_val, _ = create_sequences(
            val_states, sequence_length=seq_len, forecast_steps=forecast_steps
        )
    else:
        D_feat = config['model'].get('input_size', 23)
        X_val = np.empty((0, seq_len, D_feat), dtype=np.float32)
        y_state_val = np.empty((0, forecast_steps, D_feat), dtype=np.float32)
        y_attack_val = np.empty((0, forecast_steps), dtype=np.float32)
        y_stage_val = np.empty((0, forecast_steps), dtype=np.int64)

    if len(test_states) >= min_required_states:
        X_test, y_state_test, y_attack_test, y_stage_test, _ = create_sequences(
            test_states, sequence_length=seq_len, forecast_steps=forecast_steps
        )
    else:
        D_feat = config['model'].get('input_size', 23)
        X_test = np.empty((0, seq_len, D_feat), dtype=np.float32)
        y_state_test = np.empty((0, forecast_steps, D_feat), dtype=np.float32)
        y_attack_test = np.empty((0, forecast_steps), dtype=np.float32)
        y_stage_test = np.empty((0, forecast_steps), dtype=np.int64)

    print(f"Independent sequence shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Strict Scaler Discipline: Fit scaler strictly on Train split only
    scaler = StateScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    N_train, K_steps, D_feat = y_state_train.shape
    y_state_train_scaled = scaler.transform(y_state_train.reshape(-1, D_feat)).reshape(N_train, K_steps, D_feat)

    if len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)
        y_state_val_scaled = scaler.transform(y_state_val.reshape(-1, D_feat)).reshape(len(y_state_val), K_steps, D_feat)
    else:
        X_val_scaled = np.empty((0, seq_len, D_feat), dtype=np.float32)
        y_state_val_scaled = np.empty((0, K_steps, D_feat), dtype=np.float32)

    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
        y_state_test_scaled = scaler.transform(y_state_test.reshape(-1, D_feat)).reshape(len(y_state_test), K_steps, D_feat)
    else:
        X_test_scaled = np.empty((0, seq_len, D_feat), dtype=np.float32)
        y_state_test_scaled = np.empty((0, K_steps, D_feat), dtype=np.float32)

    scaler_path = config['model']['scaler_path']
    scaler.save(scaler_path)

    # PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_state_train_scaled, dtype=torch.float32),
        torch.tensor(y_attack_train, dtype=torch.float32),
        torch.tensor(y_stage_train, dtype=torch.long)
    )
    if len(X_val) > 0:
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_state_val_scaled, dtype=torch.float32),
            torch.tensor(y_attack_val, dtype=torch.float32),
            torch.tensor(y_stage_val, dtype=torch.long)
        )
    else:
        val_dataset = train_dataset

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Temporal LSTM World Model (Multi-Step Unrolling K={forecast_steps}) on device: {device}")

    model = TemporalLSTMWorldModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    # Loss Functions
    state_criterion = nn.MSELoss()
    attack_criterion = nn.BCELoss()
    stage_criterion = nn.CrossEntropyLoss()

    w_state = config['training']['loss_weights']['state_prediction']
    w_attack = config['training']['loss_weights']['attack_prob']
    w_stage = config['training']['loss_weights']['stage_class']

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    epochs = config['training']['epochs']
    best_val_loss = float('inf')

    weights_path = config['model']['weights_path']
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    discount_factors = [1.0, 0.5, 0.25]

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for b_x, b_y_state, b_y_attack, b_y_stage in train_loader:
            b_x = b_x.to(device)
            b_y_state = b_y_state.to(device)
            b_y_attack = b_y_attack.to(device)
            b_y_stage = b_y_stage.to(device)

            optimizer.zero_grad()

            total_step_loss = 0.0
            curr_seq = b_x

            # Unroll K steps ahead with scheduled sampling
            for k in range(forecast_steps):
                p_state, p_attack, p_stage = model(curr_seq)

                gt_state_k = b_y_state[:, k, :]
                gt_attack_k = b_y_attack[:, k].unsqueeze(1)
                gt_stage_k = b_y_stage[:, k]

                l_state = state_criterion(p_state, gt_state_k)
                l_attack = attack_criterion(p_attack, gt_attack_k)
                l_stage = stage_criterion(p_stage, gt_stage_k)

                step_loss = w_state * l_state + w_attack * l_attack + w_stage * l_stage
                discount = discount_factors[k] if k < len(discount_factors) else (0.5 ** k)
                total_step_loss = total_step_loss + discount * step_loss

                # Scheduled sampling for next input timestep
                if k < forecast_steps - 1:
                    # Append predicted state (detached or autoregressive) to sequence
                    next_input = p_state.unsqueeze(1)
                    curr_seq = torch.cat([curr_seq[:, 1:, :], next_input], dim=1)

            total_step_loss.backward()
            optimizer.step()

            train_loss += total_step_loss.item() * len(b_x)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_x, b_y_state, b_y_attack, b_y_stage in val_loader:
                b_x = b_x.to(device)
                b_y_state = b_y_state.to(device)
                b_y_attack = b_y_attack.to(device)
                b_y_stage = b_y_stage.to(device)

                total_step_loss = 0.0
                curr_seq = b_x

                for k in range(forecast_steps):
                    p_state, p_attack, p_stage = model(curr_seq)

                    gt_state_k = b_y_state[:, k, :]
                    gt_attack_k = b_y_attack[:, k].unsqueeze(1)
                    gt_stage_k = b_y_stage[:, k]

                    l_state = state_criterion(p_state, gt_state_k)
                    l_attack = attack_criterion(p_attack, gt_attack_k)
                    l_stage = stage_criterion(p_stage, gt_stage_k)

                    step_loss = w_state * l_state + w_attack * l_attack + w_stage * l_stage
                    discount = discount_factors[k] if k < len(discount_factors) else (0.5 ** k)
                    total_step_loss = total_step_loss + discount * step_loss

                    if k < forecast_steps - 1:
                        next_input = p_state.unsqueeze(1)
                        curr_seq = torch.cat([curr_seq[:, 1:, :], next_input], dim=1)

                val_loss += total_step_loss.item() * len(b_x)

        val_loss /= max(1, len(val_dataset))

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Multi-Step Loss: {train_loss:.4f} | Val Multi-Step Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)

    print(f"World Model multi-step training complete. Best model weights saved to {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal LSTM World Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--pcap-path", type=str, default=None, help="Path to raw PCAP file for PCAP feature ablation training")
    args = parser.parse_args()

    train(config_path=args.config, pcap_path=args.pcap_path)


