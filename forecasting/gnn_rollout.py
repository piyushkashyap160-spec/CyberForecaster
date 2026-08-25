"""
gnn_rollout.py — K-step recursive forward simulation for the Temporal GNN World Model.

The model encodes observed dynamic network graphs G(t-L+1)...G(t) and forecasts
future network state vectors S(t+1)...S(t+K). It does NOT synthesize future graph topology.
The most recent observed graph embedding is carried forward during the rollout.

Design invariant: model.graph_encoder is used for encoding — no separate instance is created.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import pandas as pd

from models.temporal_gnn_world_model import TemporalGNNWorldModel
from preprocessing.scaler import StateScaler
from preprocessing.node_feature_scaler import NodeFeatureScaler
from preprocessing.graph_builder import build_window_graph
from preprocessing.state_encoder import STATE_FEATURE_KEYS


def perform_gnn_k_step_rollout(
    model: TemporalGNNWorldModel,
    state_scaler: StateScaler,
    node_scaler: NodeFeatureScaler,
    historical_sequence: np.ndarray,          # (L, 23) original-scale state vectors
    historical_df_windows: Optional[List[pd.DataFrame]] = None,  # L raw window DataFrames
    k_steps: int = 5,
    device: torch.device = None,
) -> List[Dict]:
    """
    Performs recursive K-step forward simulation for the Temporal GNN World Model.

    Pipeline:
      Observed G(t-L+1)...G(t)  [via model.graph_encoder]
      + Observed S(t-L+1)...S(t)
        ↓
      LSTM → S(t+1), P(Attack t+1), Stage(t+1)
        ↓
      Slide window: discard S(t-L+1), append S(t+1)
      Reuse g(t) embedding as proxy for G(t+1) (future topology not synthesized)
        ↓
      Repeat for k=2..K

    NOTE: The model forecasts future state vectors and attack risk; it does not
    generate future graph topology. Future graph embeddings during rollout are
    approximated by repeating the most recently observed graph embedding g(t).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    L = historical_sequence.shape[0]
    embed_dim = model.graph_embed_dim

    # --- Encode historical graph sequence using model.graph_encoder ---
    g_seq = np.zeros((L, embed_dim), dtype=np.float32)

    if historical_df_windows is not None and len(historical_df_windows) == L:
        with torch.no_grad():
            for t, df_win in enumerate(historical_df_windows):
                g_dict  = build_window_graph(df_win)
                x_raw   = g_dict['node_features']                      # (num_nodes, 10)
                x_norm  = node_scaler.transform(x_raw)
                x_t     = torch.tensor(x_norm, dtype=torch.float32).to(device)
                ei_t    = torch.tensor(g_dict['edge_index'], dtype=torch.long).to(device)
                g_emb   = model.graph_encoder(x_t, ei_t)               # (1, 64) — trained weights
                g_seq[t] = g_emb.squeeze(0).cpu().numpy()
    else:
        # No window DataFrames provided: use zero embeddings (documented limitation)
        pass  # g_seq remains zeros

    # Scale state sequence
    current_seq_scaled = state_scaler.transform(historical_sequence[np.newaxis, :, :])  # (1, L, 23)
    current_g_seq = g_seq[np.newaxis, :, :]                                              # (1, L, 64)

    rollout_results = []

    with torch.no_grad():
        for k in range(1, k_steps + 1):
            seq_tensor = torch.tensor(current_seq_scaled, dtype=torch.float32).to(device)
            g_tensor   = torch.tensor(current_g_seq,      dtype=torch.float32).to(device)

            pred_state_scaled, attack_prob, stage_logits = model.forward_graph_sequence(
                seq_tensor, g_tensor
            )

            pred_state_np_scaled = pred_state_scaled.cpu().numpy()    # (1, 23)
            attack_prob_val      = float(attack_prob.cpu().numpy()[0, 0])
            stage_id             = int(torch.argmax(stage_logits, dim=1).cpu().numpy()[0])

            pred_state_orig = state_scaler.inverse_transform(pred_state_np_scaled)[0]
            state_dict = {
                STATE_FEATURE_KEYS[i]: float(pred_state_orig[i])
                for i in range(len(STATE_FEATURE_KEYS))
            }

            rollout_results.append({
                'horizon_step':        f"t+{k}",
                'step_index':          k,
                'attack_probability':  round(attack_prob_val, 4),
                'predicted_stage_id':  stage_id,
                'state_vector':        pred_state_orig,
                'state_dict':          state_dict,
            })

            # Slide state window forward
            next_step_scaled = pred_state_np_scaled[:, np.newaxis, :]          # (1, 1, 23)
            current_seq_scaled = np.concatenate(
                [current_seq_scaled[:, 1:, :], next_step_scaled], axis=1
            )

            # Slide graph window: reuse most recent observed graph embedding as proxy
            latest_g = current_g_seq[:, -1:, :]                                # (1, 1, 64)
            current_g_seq = np.concatenate(
                [current_g_seq[:, 1:, :], latest_g], axis=1
            )

    return rollout_results
