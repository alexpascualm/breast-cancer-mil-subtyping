"""
models/mil/simple_mil.py
------------------------
Estrategias de agregación MIL sin parámetros aprendibles (Max y Mean),
disponibles en dos modos cada una:

    Instance-level (max-instance-level, mean-instance-level):
        Operan directamente sobre los logits de parche ya calculados
        por el patch_classifier de WSIClassifier.
        El patch_classifier SÍ se usa. Estos módulos NO tienen
        clasificador propio.

    Bag-level (max-bag-level, mean-bag-level):
        Agregan los embeddings de parche (feats_ws) en un bag embedding
        y lo proyectan con un clasificador propio (bag_classifier).
        El patch_classifier de WSIClassifier NO interviene.
        logits_ws se ignora.
"""

import torch
import torch.nn as nn
from models.mil_aggregators.base_mil import BaseMIL


# ─────────────────────────────────────────────────────────────────────────────
# MAX — Instance-level
# ─────────────────────────────────────────────────────────────────────────────

class MaxInstanceMIL(BaseMIL):
    """
    Max-pooling sobre los logits de parche (instance-level).

    Toma el logit máximo por clase entre todos los parches.
    Clave CLI: "max-instance-level".
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, logits_ws, feats_ws):
        if logits_ws.size(0) == 0:
            return torch.zeros(self.num_classes, device=logits_ws.device), None
        return torch.max(logits_ws, dim=0)[0], None   # (num_classes,)


# ─────────────────────────────────────────────────────────────────────────────
# MAX — Bag-level
# ─────────────────────────────────────────────────────────────────────────────

class MaxBagMIL(BaseMIL):
    """
    Max-pooling sobre los embeddings de parche, seguido de un clasificador
    bag-level propio (bag-level).

    Pipeline:
        feats_ws  → max-pool por dim. de feature → z  (embed_dim,)
        z         → bag_classifier                → logit_WSI  (num_classes,)

    logits_ws se ignora.
    Clave CLI: "max-bag-level".
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)
        self.bag_classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, logits_ws, feats_ws):
        if feats_ws.size(0) == 0:
            return torch.zeros(self.num_classes, device=feats_ws.device), None
        z = torch.max(feats_ws, dim=0)[0]          # (embed_dim,)
        wsi_logit = self.bag_classifier(z)          # (num_classes,)
        return wsi_logit, None


# ─────────────────────────────────────────────────────────────────────────────
# MEAN — Instance-level
# ─────────────────────────────────────────────────────────────────────────────

class MeanInstanceMIL(BaseMIL):
    """
    Mean-pooling sobre los logits de parche (instance-level).

    Promedia los logits de parche por clase.
    Clave CLI: "mean-instance-level".
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, logits_ws, feats_ws):
        if logits_ws.size(0) == 0:
            return torch.zeros(self.num_classes, device=logits_ws.device), None
        return torch.mean(logits_ws, dim=0), None   # (num_classes,)


# ─────────────────────────────────────────────────────────────────────────────
# MEAN — Bag-level
# ─────────────────────────────────────────────────────────────────────────────

class MeanBagMIL(BaseMIL):
    """
    Mean-pooling sobre los embeddings de parche, seguido de un clasificador
    bag-level propio (bag-level).

    Pipeline:
        feats_ws  → mean-pool → z  (embed_dim,)
        z         → bag_classifier → logit_WSI  (num_classes,)

    logits_ws se ignora.
    Clave CLI: "mean-bag-level".
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)
        self.bag_classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, logits_ws, feats_ws):
        if feats_ws.size(0) == 0:
            return torch.zeros(self.num_classes, device=feats_ws.device), None
        z = torch.mean(feats_ws, dim=0)             # (embed_dim,)
        wsi_logit = self.bag_classifier(z)          # (num_classes,)
        return wsi_logit, None