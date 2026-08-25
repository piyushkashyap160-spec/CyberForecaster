import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class GraphEncoder(nn.Module):
    """
    Graph Neural Network Encoder mapping dynamic graph G(t) to fixed-size vector embedding g(t) in R^64.
    Uses GraphSAGE message passing and global mean pooling over host nodes.
    """
    def __init__(self, node_dim: int = 10, hidden_dim: int = 64, output_dim: int = 64):
        super(GraphEncoder, self).__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if HAS_PYG:
            self.conv1 = SAGEConv(node_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        else:
            # Fallback PyTorch Linear projection + Mean Pooling if PyG is not imported
            self.fc1 = nn.Linear(node_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        x: (num_nodes, node_dim)
        edge_index: (2, num_edges)
        batch: (num_nodes,) optional graph batch indices
        Returns:
          g_t: (batch_size, output_dim) or (1, output_dim)
        """
        if HAS_PYG:
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            if batch is None:
                # Default single graph pooling
                g = torch.mean(h, dim=0, keepdim=True)
            else:
                g = global_mean_pool(h, batch)
        else:
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            g = torch.mean(h, dim=0, keepdim=True)

        return self.proj(g)
