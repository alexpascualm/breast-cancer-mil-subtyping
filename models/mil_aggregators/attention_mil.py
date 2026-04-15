"""
models/mil/attention_mil.py
---------------------------
Agregación MIL con atención gated.

La red de atención combina dos ramas (Tanh + Sigmoid) para calcular
un peso escalar por parche, que se usa para ponderar los logits.
"""

import torch
import torch.nn as nn
from models.mil_aggregators.base_mil import BaseMIL


class _GatedAttention(nn.Module):
    """Red de atención gated: dos ramas multiplicadas → escalar por parche."""

    def __init__(self, embed_dim: int):
        super().__init__()
        D = embed_dim // 2
        self.branch_tanh    = nn.Sequential(nn.Linear(embed_dim, D), nn.Tanh())
        self.branch_sigmoid = nn.Sequential(nn.Linear(embed_dim, D), nn.Sigmoid())
        self.out            = nn.Linear(D, 1)

    def forward(self, x):
        # x: (N, embed_dim)
        A = self.branch_tanh(x) * self.branch_sigmoid(x)   # (N, D)
        return self.out(A).squeeze(-1)                      # (N,)


class AttentionMIL(BaseMIL):
    """
    Gated Attention MIL.

    Aprende un peso por parche usando los embeddings del encoder.
    Los pesos se normalizan con softmax y se usan para ponderar los logits.
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)
        self.attention = _GatedAttention(embed_dim)

    def forward(self, logits_ws, feats_ws):
        if feats_ws.numel() == 0:
            return (
                torch.zeros(self.num_classes, device=logits_ws.device),
                torch.zeros(0, device=logits_ws.device),
            )

        weights  = torch.softmax(self.attention(feats_ws), dim=0)  # (N,)
        wsi_logit = (logits_ws * weights.unsqueeze(1)).sum(dim=0)  # (num_classes,)
        return wsi_logit, weights
