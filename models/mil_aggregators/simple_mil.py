"""
models/mil/simple_mil.py
------------------------
Estrategias de agregación MIL sin parámetros aprendibles:
    - MaxMIL  : max-pooling sobre los logits de parche.
    - MeanMIL : mean-pooling sobre los logits de parche.
"""

import torch
from models.mil_aggregators.base_mil import BaseMIL


class MaxMIL(BaseMIL):
    """Agrega tomando el máximo logit por clase entre todos los parches."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, logits_ws, feats_ws):
        if logits_ws.size(0) == 0:
            return torch.zeros(self.num_classes, device=logits_ws.device), None
        return torch.max(logits_ws, dim=0)[0], None


class MeanMIL(BaseMIL):
    """Agrega promediando los logits de parche."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, logits_ws, feats_ws):
        if logits_ws.size(0) == 0:
            return torch.zeros(self.num_classes, device=logits_ws.device), None
        return torch.mean(logits_ws, dim=0), None
