"""
Baseline MLP — sanity check model (ignores graph structure).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NMRShiftMLP(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # (δ¹H, δ¹⁵N)
        )

    def forward(self, x, edge_index=None, batch=None):
        return self.net(x)
