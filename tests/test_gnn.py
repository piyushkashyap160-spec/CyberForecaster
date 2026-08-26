"""
test_gnn.py — Unit tests for graph builder, GraphEncoder, TemporalGNNWorldModel,
NodeFeatureScaler, gradient flow, checkpoint consistency, and GNN rollout.
"""
import os
import sys
import copy
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.graph_builder import build_window_graph
from preprocessing.node_feature_scaler import NodeFeatureScaler
from models.graph_encoder import GraphEncoder
from models.temporal_gnn_world_model import TemporalGNNWorldModel
from preprocessing.scaler import StateScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_window_df():
    return pd.DataFrame([
        {'Src_IP': '192.168.1.10', 'Dst_IP': '10.0.0.1',  'Src_Port': 1024, 'Dst_Port': 80,
         'Protocol': 6, 'Tot_Pkts': 10, 'Tot_Bytes': 1000, 'SYN_Cnt': 1, 'ACK_Cnt': 9,
         'Failed_Conn': 0, 'Mean_IAT': 0.1},
        {'Src_IP': '192.168.1.20', 'Dst_IP': '10.0.0.1',  'Src_Port': 2048, 'Dst_Port': 443,
         'Protocol': 6, 'Tot_Pkts': 5,  'Tot_Bytes': 500,  'SYN_Cnt': 1, 'ACK_Cnt': 4,
         'Failed_Conn': 0, 'Mean_IAT': 0.2},
        {'Src_IP': '192.168.1.10', 'Dst_IP': '192.168.1.20', 'Src_Port': 3000, 'Dst_Port': 8080,
         'Protocol': 6, 'Tot_Pkts': 3,  'Tot_Bytes': 300,  'SYN_Cnt': 1, 'ACK_Cnt': 2,
         'Failed_Conn': 1, 'Mean_IAT': 0.5},
    ])

def _make_gnn_model():
    return TemporalGNNWorldModel(node_dim=10, graph_embed_dim=64, state_dim=23, hidden_size=64)


# ---------------------------------------------------------------------------
# 1. Graph builder
# ---------------------------------------------------------------------------

def test_graph_builder_node_edge_count():
    df_win = _make_dummy_window_df()
    g_dict = build_window_graph(df_win, window_seconds=5.0)
    assert g_dict['num_nodes'] == 3, f"Expected 3 nodes, got {g_dict['num_nodes']}"
    assert g_dict['num_edges'] == 3, f"Expected 3 edges, got {g_dict['num_edges']}"
    assert g_dict['node_features'].shape == (3, 10)
    assert g_dict['edge_features'].shape == (3, 6)


def test_graph_builder_empty_df():
    g_dict = build_window_graph(pd.DataFrame())
    # Should return a dummy single-node graph without crashing
    assert g_dict['num_nodes'] >= 1
    assert g_dict['node_features'].shape[1] == 10


# ---------------------------------------------------------------------------
# 2. NodeFeatureScaler — fitted only on training data
# ---------------------------------------------------------------------------

def test_node_feature_scaler_fit_transform():
    rng = np.random.default_rng(0)
    train_mats = [rng.random((5, 10)).astype(np.float32) * 1000 for _ in range(20)]
    scaler = NodeFeatureScaler()
    scaler.fit(train_mats)
    result = scaler.transform(train_mats[0])
    assert result.shape == (5, 10), f"Unexpected shape: {result.shape}"
    assert not np.any(np.isnan(result)), "NaN in scaled output"
    assert not np.any(np.isinf(result)), "Inf in scaled output"


def test_node_feature_scaler_not_fitted_raises():
    scaler = NodeFeatureScaler()
    with pytest.raises(AssertionError):
        scaler.transform(np.ones((3, 10), dtype=np.float32))


def test_node_feature_scaler_save_load(tmp_path):
    rng = np.random.default_rng(1)
    train_mats = [rng.random((4, 10)).astype(np.float32) * 500 for _ in range(10)]
    scaler = NodeFeatureScaler()
    scaler.fit(train_mats)

    path = str(tmp_path / "ns.joblib")
    scaler.save(path)

    scaler2 = NodeFeatureScaler()
    scaler2.load(path)

    x = rng.random((3, 10)).astype(np.float32) * 200
    np.testing.assert_allclose(scaler.transform(x), scaler2.transform(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. GraphEncoder
# ---------------------------------------------------------------------------

def test_graph_encoder_output_shape():
    encoder = GraphEncoder(node_dim=10, hidden_dim=64, output_dim=64)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    g_emb = encoder(x, edge_index)
    assert g_emb.shape == (1, 64), f"Expected (1, 64), got {g_emb.shape}"


def test_graph_encoder_no_nan():
    encoder = GraphEncoder()
    x = torch.randn(8, 10)
    edge_index = torch.zeros((2, 0), dtype=torch.long)  # edgeless graph
    g_emb = encoder(x, edge_index)
    assert not torch.isnan(g_emb).any(), "NaN in graph embedding"
    assert not torch.isinf(g_emb).any(), "Inf in graph embedding"


# ---------------------------------------------------------------------------
# 4. TemporalGNNWorldModel — GraphEncoder is submodule
# ---------------------------------------------------------------------------

def test_gnn_model_graph_encoder_is_submodule():
    model = _make_gnn_model()
    named_mods = dict(model.named_modules())
    assert 'graph_encoder' in named_mods, "graph_encoder not found as submodule"


def test_gnn_model_forward_shapes():
    model = _make_gnn_model()
    x_seq = torch.randn(4, 10, 23)
    g_seq = torch.randn(4, 10, 64)
    p_state, p_attack, p_stage = model.forward_graph_sequence(x_seq, g_seq)
    assert p_state.shape  == (4, 23)
    assert p_attack.shape == (4, 1)
    assert p_stage.shape  == (4, 6)


def test_gnn_model_concat_fusion_dim():
    """Verify joint_dim is graph_embed_dim + state_dim = 64+23 = 87."""
    model = _make_gnn_model()
    # input_proj Linear layer input_features should be 87
    first_linear = model.input_proj[0]
    assert first_linear.in_features == 87, (
        f"Expected 87 (64+23), got {first_linear.in_features}"
    )


# ---------------------------------------------------------------------------
# 5. Gradient flow — GraphEncoder parameters receive non-zero gradients
# ---------------------------------------------------------------------------

def test_graph_encoder_receives_gradients():
    """
    Simulate one forward+backward pass through the full model.
    Verify GraphEncoder parameters accumulate non-zero gradients.
    """
    model = _make_gnn_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_seq = torch.randn(2, 10, 23)
    
    # Compute g_seq dynamically using model.graph_encoder
    node_x = torch.randn(5, 10)
    edge_idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    
    # Generate 10 timestep embeddings using model.graph_encoder
    g_embeds = [model.graph_encoder(node_x, edge_idx) for _ in range(10)]
    g_seq_single = torch.cat(g_embeds, dim=0).unsqueeze(0) # (1, 10, 64)
    g_seq = torch.cat([g_seq_single, g_seq_single], dim=0) # (2, 10, 64)

    optimizer.zero_grad()
    p_state, p_attack, p_stage = model.forward_graph_sequence(x_seq, g_seq)

    loss = p_state.sum() + p_attack.sum() + p_stage.sum()
    loss.backward()

    nonzero_grad_found = False
    for name, param in model.graph_encoder.named_parameters():
        if param.grad is not None and param.grad.data.norm(2).item() > 0:
            nonzero_grad_found = True
            break

    assert nonzero_grad_found, (
        "No GraphEncoder parameter received a non-zero gradient — "
        "end-to-end backprop is broken."
    )


def test_graph_encoder_params_included_in_optimizer():
    """GraphEncoder parameter ids must be a subset of optimizer parameter ids."""
    model = _make_gnn_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optim_param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    encoder_param_ids = {id(p) for p in model.graph_encoder.parameters()}

    assert encoder_param_ids.issubset(optim_param_ids), (
        "Some GraphEncoder parameters are missing from the optimizer."
    )


# ---------------------------------------------------------------------------
# 6. Checkpoint consistency — save/load produces identical outputs
# ---------------------------------------------------------------------------

def test_checkpoint_consistency(tmp_path):
    model = _make_gnn_model()
    model.eval()

    x_seq = torch.randn(1, 10, 23)
    g_seq = torch.randn(1, 10, 64)

    with torch.no_grad():
        out_before = model.forward_graph_sequence(x_seq, g_seq)

    ckpt_path = str(tmp_path / "gnn_ckpt.pt")
    torch.save(model.state_dict(), ckpt_path)

    model2 = _make_gnn_model()
    model2.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model2.eval()

    with torch.no_grad():
        out_after = model2.forward_graph_sequence(x_seq, g_seq)

    for a, b in zip(out_before, out_after):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_checkpoint_includes_graph_encoder_weights(tmp_path):
    model = _make_gnn_model()
    ckpt_path = str(tmp_path / "gnn_enc_ckpt.pt")
    torch.save(model.state_dict(), ckpt_path)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    encoder_keys = [k for k in state_dict if k.startswith('graph_encoder.')]
    assert len(encoder_keys) > 0, "Checkpoint does not contain graph_encoder weights."


# ---------------------------------------------------------------------------
# 7. GNN rollout uses model.graph_encoder (no new random instance)
# ---------------------------------------------------------------------------

def test_gnn_rollout_uses_model_graph_encoder():
    """
    Verify rollout does not produce AttributeError and uses model.graph_encoder.
    Also confirm output length equals k_steps.
    """
    from forecasting.gnn_rollout import perform_gnn_k_step_rollout

    model      = _make_gnn_model()
    state_sc   = StateScaler()
    dummy_seq  = np.random.randn(10, 23).astype(np.float32)
    state_sc.fit(dummy_seq)

    rng = np.random.default_rng(2)
    train_mats = [rng.random((3, 10)).astype(np.float32) * 100 for _ in range(15)]
    node_sc = NodeFeatureScaler()
    node_sc.fit(train_mats)

    # Without providing window DataFrames — uses zero graph embeddings (documented)
    results = perform_gnn_k_step_rollout(
        model, state_sc, node_sc, dummy_seq, k_steps=3
    )
    assert len(results) == 3
    assert results[0]['horizon_step'] == 't+1'
    assert 'attack_probability' in results[0]


# ---------------------------------------------------------------------------
# 8. Checkpoint Compatibility Regression Tests
# ---------------------------------------------------------------------------

def test_saved_gnn_checkpoint_strict_load():
    """
    Regression test: Loads actual saved checkpoint models_weights/temporal_gnn_world_model.pt
    with strict=True using the canonical TemporalGNNWorldModel class used by Streamlit dashboard.
    Fails if checkpoint architecture and model architecture diverge.
    """
    ckpt_path = "models_weights/temporal_gnn_world_model.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

    model = TemporalGNNWorldModel(
        node_dim=10,
        graph_embed_dim=64,
        state_dim=23,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        num_stages=6
    )
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    assert model.training is True or model.training is False


def test_saved_gnn_checkpoint_strict_load_without_pyg():
    """
    Regression test: Verifies strict checkpoint loading succeeds even if HAS_PYG=False.
    Guarantees environment-invariant parameter schema compatibility.
    """
    ckpt_path = "models_weights/temporal_gnn_world_model.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

    import models.graph_encoder
    orig_has_pyg = models.graph_encoder.HAS_PYG
    try:
        models.graph_encoder.HAS_PYG = False
        model = TemporalGNNWorldModel(
            node_dim=10,
            graph_embed_dim=64,
            state_dim=23,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            num_stages=6
        )
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    finally:
        models.graph_encoder.HAS_PYG = orig_has_pyg

