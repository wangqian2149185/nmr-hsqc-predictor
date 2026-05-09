"""
Primary model: Graph Convolutional Network for per-residue HSQC prediction.

Output: (δ¹H, δ¹⁵N) for every node (residue) in the graph.
No global pooling — this is a node-level regression task.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NMRShiftGCN(torch.nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(node_feat_dim, hidden_dim))
        self.norms.append(torch.nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(torch.nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 2),  # (δ¹H, δ¹⁵N)
        )

    def forward(self, x, edge_index, batch=None):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.relu(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)  # (N_nodes, 2)
