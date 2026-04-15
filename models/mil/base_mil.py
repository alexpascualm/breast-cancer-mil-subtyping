"""
models/mil/base_mil.py
----------------------
Clase base abstracta para todas las estrategias de agregación MIL.

Interfaz:
    - forward(logits_ws, feats_ws) → (wsi_logit_vector, weights_or_None)
      Recibe los logits y embeddings de UNA sola WSI.
      Devuelve el vector agregado (num_classes,) y los pesos (N,) o None.

La lógica de batching (dividir por WSI, apilar resultados) vive en
MILAggregator, que es la clase que usan los encoders.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseMIL(ABC, nn.Module):
    """
    Interfaz común para estrategias de agregación MIL por WSI.

    Subclases obligadas a implementar:
        - forward(logits_ws, feats_ws) → (Tensor(num_classes,), Tensor(N,) | None)
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim

    @abstractmethod
    def forward(
        self,
        logits_ws: torch.Tensor,   # (N, num_classes)
        feats_ws:  torch.Tensor,   # (N, embed_dim)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Agrega los logits de parche de UNA WSI en un único vector.

        Returns:
            wsi_logits : Tensor (num_classes,)
            weights    : Tensor (N,) con los pesos por parche, o None
        """
        ...
