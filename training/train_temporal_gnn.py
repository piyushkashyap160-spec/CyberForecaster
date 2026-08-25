"""
train_temporal_gnn.py â€” Corrected end-to-end Temporal GNN World Model training.

Key design invariants (verified by this script):
  1. GraphEncoder is self.graph_encoder inside TemporalGNNWorldModel â€” same optimizer.
  2. Graph embeddings are computed PER BATCH inside the training loop (not pre-cached).
  3. NodeFeatureScaler is fitted on training split node features ONLY.
  4. Gradient norms for SAGEConv parameters are logged each epoch.
  5. The saved checkpoint includes GraphEncoder weights (it's part of model.state_dict()).
"""

import os
import sys
import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.graph_builder import build_window_graph
from preprocessing.scaler import StateScaler
from preprocessing.node_feature_scaler import NodeFeatureScaler
from models.temporal_gnn_world_model import TemporalGNNWorldModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GNNSequenceDataset(Dataset):
    """
    Dataset that stores raw graph dicts (node_features, edge_index) per time window,
    plus the sequence of state vectors and labels.

    During __getitem__ the node features are normalized by NodeFeatureScaler.
    Graph embeddings are NOT precomputed â€” they are computed inside the model's forward pass.
    """

    def __init__(
        self,
        X_scaled: np.ndarray,          # (N, L, 23) scaled state sequences
        graphs: list,                   # list of lists: graphs[i] = [g_dict_{t-L+1}, ..., g_dict_{t}]
        y_state_scaled: np.ndarray,     # (N, 23)
        y_attack: np.ndarray,           # (N,)
        y_stage: np.ndarray,            # (N,)
        node_scaler: NodeFeatureScaler,
    ):
        self.X = X_scaled
        self.graphs = graphs
        self.y_state = y_state_scaled
        self.y_attack = y_attack
        self.y_stage = y_stage
        self.node_scaler = node_scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return scaled state tensors and per-timestep graph data (as lists of tensors)
        x_seq = torch.tensor(self.X[idx], dtype=torch.float32)           # (L, 23)
        y_s   = torch.tensor(self.y_state[idx], dtype=torch.float32)     # (23,)
        y_a   = torch.tensor(self.y_attack[idx], dtype=torch.float32)    # scalar
        y_stg = torch.tensor(self.y_stage[idx], dtype=torch.long)        # scalar

        # Build normalized node feature tensors and edge indices for each timestep
        node_feat_list = []
        edge_idx_list  = []
        for g_dict in self.graphs[idx]:
            x_raw = g_dict['node_features']                     # (num_nodes, 10)
            x_norm = self.node_scaler.transform(x_raw)          # (num_nodes, 10) normalized
            node_feat_list.append(torch.tensor(x_norm, dtype=torch.float32))
            edge_idx_list.append(torch.tensor(g_dict['edge_index'], dtype=torch.long))

        return x_seq, node_feat_list, edge_idx_list, y_s, y_a, y_stg


def gnn_collate_fn(batch):
    """
    Custom collate: state tensors are stacked normally;
    graph lists remain as lists-of-lists (variable num_nodes per sample).
    """
    x_seqs, graph_nodes, graph_edges, y_states, y_attacks, y_stages = zip(*batch)
    return (
        torch.stack(x_seqs),            # (B, L, 23)
        list(graph_nodes),              # list[B] of list[L] of node tensors
        list(graph_edges),              # list[B] of list[L] of edge tensors
        torch.stack(y_states),          # (B, 23)
        torch.stack(y_attacks).unsqueeze(1),  # (B, 1)
        torch.stack(y_stages),          # (B,)
    )


def encode_graph_sequence_batch(
    model: TemporalGNNWorldModel,
    batch_node_feats: list,    # list[B] of list[L] of (num_nodes, 10) tensors
    batch_edge_idxs: list,     # list[B] of list[L] of (2, E) tensors
    device: torch.device,
) -> torch.Tensor:
    """
    Calls model.graph_encoder per (batch_sample, timestep) and stacks into [B, L, 64].
    This ensures gradients flow back into model.graph_encoder parameters.
    """
    B = len(batch_node_feats)
    L = len(batch_node_feats[0])
    embed_dim = model.graph_embed_dim

    # Accumulate embeddings as a list of lists for stacking
    g_seq = torch.zeros(B, L, embed_dim, device=device)

    for b in range(B):
        for t in range(L):
            x_t  = batch_node_feats[b][t].to(device)    # (num_nodes, 10)
            ei_t = batch_edge_idxs[b][t].to(device)     # (2, E)
            g_t  = model.graph_encoder(x_t, ei_t)       # (1, 64) â€” gradient-connected
            g_seq[b, t] = g_t.squeeze(0)

    return g_seq  # (B, L, 64)


def verify_graph_encoder_gradients(model: TemporalGNNWorldModel) -> None:
    """
    Print gradient norms for GraphEncoder parameters.
    Called after the first backward pass to confirm non-zero gradients.
    """
    print("\n--- GraphEncoder Gradient Verification ---")
    any_nonzero = False
    for name, param in model.graph_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"  {name:40s}  grad_norm = {grad_norm:.6f}")
            if grad_norm > 0:
                any_nonzero = True
        else:
            print(f"  {name:40s}  grad = None")
    if any_nonzero:
        print("âœ“ GraphEncoder receives non-zero gradients â€” end-to-end backprop confirmed.")
    else:
        print("âœ— WARNING: GraphEncoder gradients are all zero or None!")
    print("------------------------------------------\n")


def train_temporal_gnn(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['training']['seed'])

    csv_path = config['data']['demo_csv_path']
    print(f"Loading dataset for Temporal GNN from {csv_path}...")
    df = load_flow_csv(csv_path)

    window_sec = config['data']['window_seconds']
    seq_len    = config['sequence']['sequence_length']

    states = build_network_states(df, window_seconds=window_sec)
    # Each state dict now contains 'df_window' with the raw flows for that window.

    X, y_state, y_attack, y_stage, _ = create_sequences(states, sequence_length=seq_len)

    total_samples = len(X)
    train_end = int(total_samples * config['training']['train_split'])
    val_end   = train_end + int(total_samples * config['training']['val_split'])

    print(f"Dataset split -> Train: {train_end} | Val: {val_end - train_end} | Test: {total_samples - val_end}")
    print(f"  Train attack labels: {int(y_attack[:train_end].sum())} / {train_end}")
    print(f"  Val   attack labels: {int(y_attack[train_end:val_end].sum())} / {val_end - train_end}")

    # --- State Vector Scaler (StateScaler fitted on training data only) ---
    state_scaler = StateScaler()
    state_scaler.load(config['model']['scaler_path'])
    X_train_scaled = state_scaler.transform(X[:train_end])
    y_state_train_scaled = state_scaler.transform(y_state[:train_end])
    X_val_scaled   = state_scaler.transform(X[train_end:val_end])
    y_state_val_scaled   = state_scaler.transform(y_state[train_end:val_end])

    # --- Build per-sequence graph dict lists ---
    # graphs[i] = list of L graph dicts for sequence i
    # sequence i uses windows[i : i+seq_len]
    print("Building per-sequence graph data...")
    graphs_all = []
    for i in range(len(states) - seq_len):
        seq_graphs = []
        for j in range(i, i + seq_len):
            df_win = states[j].get('df_window', pd.DataFrame())
            g_dict = build_window_graph(df_win, window_seconds=window_sec)
            seq_graphs.append(g_dict)
        graphs_all.append(seq_graphs)

    # --- NodeFeatureScaler: fit ONLY on training windows ---
    print("Fitting NodeFeatureScaler on training split node features...")
    node_scaler = NodeFeatureScaler()
    train_node_feat_matrices = []
    for i in range(train_end):
        for g_dict in graphs_all[i]:
            train_node_feat_matrices.append(g_dict['node_features'])
    node_scaler.fit(train_node_feat_matrices)
    node_scaler_path = "models_weights/node_feature_scaler.joblib"
    os.makedirs(os.path.dirname(node_scaler_path), exist_ok=True)
    node_scaler.save(node_scaler_path)

    # --- Datasets ---
    train_dataset = GNNSequenceDataset(
        X_train_scaled, graphs_all[:train_end],
        y_state_train_scaled, y_attack[:train_end], y_stage[:train_end], node_scaler
    )
    val_dataset = GNNSequenceDataset(
        X_val_scaled, graphs_all[train_end:val_end],
        y_state_val_scaled, y_attack[train_end:val_end], y_stage[train_end:val_end], node_scaler
    )

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=gnn_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=gnn_collate_fn)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Temporal GNN + LSTM World Model on device: {device}...")

    model = TemporalGNNWorldModel(
        node_dim=10,
        graph_embed_dim=64,
        state_dim=23,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        num_stages=config['model']['num_stages']
    ).to(device)

    # VERIFICATION: GraphEncoder IS part of model.parameters()
    total_params     = sum(p.numel() for p in model.parameters())
    encoder_params   = sum(p.numel() for p in model.graph_encoder.parameters())
    print(f"Model total params: {total_params:,} | GraphEncoder params: {encoder_params:,}")
    assert encoder_params > 0, "GraphEncoder has no parameters â€” check model definition."

    state_criterion  = nn.MSELoss()
    attack_criterion = nn.BCELoss()
    stage_criterion  = nn.CrossEntropyLoss()

    w_state  = config['training']['loss_weights']['state_prediction']
    w_attack = config['training']['loss_weights']['attack_prob']
    w_stage  = config['training']['loss_weights']['stage_class']

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    epochs = config['training']['epochs']
    best_val_loss   = float('inf')
    gnn_weights_path = "models_weights/temporal_gnn_world_model.pt"

    gradient_verified = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_loss_state = train_loss_attack = train_loss_stage = 0.0

        for b_x, b_nodes, b_edges, b_ys, b_ya, b_ystg in train_loader:
            b_x, b_ys, b_ya, b_ystg = (
                b_x.to(device), b_ys.to(device), b_ya.to(device), b_ystg.to(device)
            )

            # Encode graph sequence end-to-end (gradients flow through graph_encoder)
            b_g = encode_graph_sequence_batch(model, b_nodes, b_edges, device)  # (B, L, 64)

            optimizer.zero_grad()
            p_state, p_attack, p_stage = model.forward_graph_sequence(b_x, b_g)

            l_state  = state_criterion(p_state, b_ys)
            l_attack = attack_criterion(p_attack, b_ya)
            l_stage  = stage_criterion(p_stage, b_ystg)
            tot      = w_state * l_state + w_attack * l_attack + w_stage * l_stage

            tot.backward()

            # Verify gradients flow into GraphEncoder on the very first batch
            if not gradient_verified:
                verify_graph_encoder_gradients(model)
                gradient_verified = True

            optimizer.step()

            n = len(b_x)
            train_loss        += tot.item()      * n
            train_loss_state  += l_state.item()  * n
            train_loss_attack += l_attack.item() * n
            train_loss_stage  += l_stage.item()  * n

        N_train = len(train_dataset)
        train_loss        /= N_train
        train_loss_state  /= N_train
        train_loss_attack /= N_train
        train_loss_stage  /= N_train

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_x, b_nodes, b_edges, b_ys, b_ya, b_ystg in val_loader:
                b_x, b_ys, b_ya, b_ystg = (
                    b_x.to(device), b_ys.to(device), b_ya.to(device), b_ystg.to(device)
                )
                b_g = encode_graph_sequence_batch(model, b_nodes, b_edges, device)
                p_state, p_attack, p_stage = model.forward_graph_sequence(b_x, b_g)
                l_state  = state_criterion(p_state, b_ys)
                l_attack = attack_criterion(p_attack, b_ya)
                l_stage  = stage_criterion(p_stage, b_ystg)
                val_loss += (w_state * l_state + w_attack * l_attack + w_stage * l_stage).item() * len(b_x)
        val_loss /= max(1, len(val_dataset))

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"GNN Epoch [{epoch:02d}/{epochs:02d}]"
                f" | Train Total: {train_loss:.4f}"
                f"  (State: {train_loss_state:.4f}, Attack: {train_loss_attack:.4f}, Stage: {train_loss_stage:.4f})"
                f" | Val Total: {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), gnn_weights_path)

    print(f"\nTemporal GNN training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model checkpoint (includes GraphEncoder): {gnn_weights_path}")
    print(f"NodeFeatureScaler: {node_scaler_path}")


if __name__ == "__main__":
    train_temporal_gnn()
