import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_PYG = True
except (ImportError, Exception):
    HAS_PYG = False


class CustomSAGEConv(nn.Module):
    """
    Pure PyTorch GraphSAGE Message-Passing Layer.
    Preserves exact parameter schema (lin_l.weight, lin_l.bias, lin_r.weight)
    to guarantee 100% state_dict binary compatibility across environments.
    """
    def __init__(self, in_features: int, out_features: int):
        super(CustomSAGEConv, self).__init__()
        self.lin_l = nn.Linear(in_features, out_features, bias=True)
        self.lin_r = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim == 2 and edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            num_nodes = x.size(0)
            deg = torch.zeros((num_nodes, 1), device=x.device)
            out_neighbors = torch.zeros_like(x)
            out_neighbors.index_add_(0, dst, x[src])
            deg.index_add_(0, dst, torch.ones((edge_index.shape[1], 1), device=x.device))
            deg = torch.clamp(deg, min=1.0)
            out_neighbors = out_neighbors / deg
        else:
            out_neighbors = torch.zeros_like(x)

        return self.lin_l(x) + self.lin_r(out_neighbors)


class GraphEncoder(nn.Module):
    """
    Graph Neural Network Encoder mapping dynamic graph G(t) to fixed-size vector embedding g(t) in R^64.
    Uses GraphSAGE message passing and global mean pooling over host nodes.
    Deterministic parameter schema: conv1 (SAGEConv), conv2 (SAGEConv), proj (Sequential).
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
            self.conv1 = CustomSAGEConv(node_dim, hidden_dim)
            self.conv2 = CustomSAGEConv(hidden_dim, output_dim)

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
                g = torch.mean(h, dim=0, keepdim=True)
            else:
                g = global_mean_pool(h, batch)
        else:
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            g = torch.mean(h, dim=0, keepdim=True)

        return self.proj(g)
