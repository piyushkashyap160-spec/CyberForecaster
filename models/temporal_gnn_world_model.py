import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_encoder import GraphEncoder

class TemporalGNNWorldModel(nn.Module):
    """
    Advanced Multi-Task Temporal GNN + LSTM World Model.
    Learns temporal transition dynamics of dynamic network communication graphs P(S[t+1] | G[t]...G[t-L+1]).

    Outputs:
      1. Next State Vector S(t+1) (Regression)
      2. Future Attack Probability P(Attack) (Binary Classification)
      3. MITRE ATT&CK Stage Logits (Multi-Class Classification)
    """
    def __init__(self, node_dim: int = 10, graph_embed_dim: int = 64, state_dim: int = 23, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, num_stages: int = 6):
        super(TemporalGNNWorldModel, self).__init__()

        self.node_dim = node_dim
        self.graph_embed_dim = graph_embed_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stages = num_stages

        # Graph Encoder Backbone
        self.graph_encoder = GraphEncoder(node_dim=node_dim, hidden_dim=graph_embed_dim, output_dim=graph_embed_dim)

        # Joint Input Projection (Graph Embedding + State Vector)
        joint_dim = graph_embed_dim + state_dim
        self.input_proj = nn.Sequential(
            nn.Linear(joint_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        # Core Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Latent Representation Mapping
        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-Task Prediction Heads
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, state_dim)
        )

        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_stages)
        )

    def forward_graph_sequence(self, x_seq: torch.Tensor, graph_embed_seq: torch.Tensor):
        """
        Forward pass with pre-computed sequence of graph embeddings graph_embed_seq (batch_size, seq_len, graph_embed_dim)
        and state sequence x_seq (batch_size, seq_len, state_dim).
        """
        batch_size, seq_len, _ = x_seq.shape

        # Concatenate graph structural embeddings with temporal state vectors
        joint_inputs = torch.cat([graph_embed_seq, x_seq], dim=-1) # (B, L, 64 + 23)

        x_proj = self.input_proj(joint_inputs)

        lstm_out, _ = self.lstm(x_proj)
        last_step_embedding = lstm_out[:, -1, :]

        latent = self.latent_layer(last_step_embedding)

        pred_state = self.state_head(latent)
        attack_prob = self.attack_head(latent)
        stage_logits = self.stage_head(latent)

        return pred_state, attack_prob, stage_logits

    def forward(self, x_seq: torch.Tensor, graph_embed_seq: torch.Tensor = None):
        """
        Default forward pass. If graph_embed_seq is None, projects zero/dummy graph embeddings.
        """
        batch_size, seq_len, state_dim = x_seq.shape
        if graph_embed_seq is None:
            device = x_seq.device
            graph_embed_seq = torch.zeros((batch_size, seq_len, self.graph_embed_dim), dtype=torch.float32, device=device)

        return self.forward_graph_sequence(x_seq, graph_embed_seq)
