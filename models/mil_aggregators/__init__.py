"""
models/mil/__init__.py
----------------------
Registro central de estrategias MIL y clase MILAggregator.

Hay dos familias de estrategias:

    INSTANCE-BASED (max, mean, attentionMIL):
        Reciben patch_logits y patch_feats.
        El clasificador de parche vive en el encoder (BasePatchEncoder).
        El MIL solo agrega los logits de parche ya calculados.

    BAG-BASED (transMIL):
        Reciben patch_feats directamente y llevan su propio clasificador
        de WSI interno. Ignoran patch_logits.
        MILAggregator detecta estas estrategias via el flag HAS_OWN_CLASSIFIER
        y devuelve sus logits directamente sin pasar por el clasificador del encoder.

Para añadir una estrategia nueva:
    1. Crear su módulo heredando de BaseMIL.
    2. Añadir la clave al diccionario MIL_REGISTRY.
    3. Si lleva clasificador propio, añadirla a BAG_BASED_STRATEGIES.
"""

import torch
import torch.nn as nn

from models.mil_aggregators.base_mil      import BaseMIL
from models.mil_aggregators.simple_mil    import MaxMIL, MeanMIL
from models.mil_aggregators.attention_mil import AttentionMIL
from models.mil_aggregators.trans_mil     import TransMIL

# Registro: clave CLI → clase
MIL_REGISTRY: dict[str, type[BaseMIL]] = {
    "max":          MaxMIL,
    "mean":         MeanMIL,
    "attentionMIL": AttentionMIL,
    "transMIL":     TransMIL,
}

# Estrategias que llevan su propio clasificador de WSI
# (no usan el clasificador de parche del encoder)
BAG_BASED_STRATEGIES = {"transMIL"}


class MILAggregator(nn.Module):
    """
    Wrapper de batch sobre cualquier estrategia BaseMIL.

    Divide el tensor concatenado de parches por WSI, aplica la estrategia
    elegida a cada una y apila los vectores resultantes.

    Args:
        aggregation (str) : clave en MIL_REGISTRY.
        num_classes (int) : número de clases de salida.
        embed_dim (int)   : dimensión de los embeddings del encoder.
    """

    def __init__(self, aggregation: str, num_classes: int, embed_dim: int):
        super().__init__()

        if aggregation not in MIL_REGISTRY:
            raise ValueError(
                f"Estrategia MIL desconocida: '{aggregation}'. "
                f"Opciones: {list(MIL_REGISTRY.keys())}"
            )

        self.aggregation = aggregation
        self.is_bag_based = aggregation in BAG_BASED_STRATEGIES

        self.strategy: BaseMIL = MIL_REGISTRY[aggregation](
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    def forward(
        self,
        patch_logits: torch.Tensor,   # (N_total, num_classes)
        patch_feats:  torch.Tensor,   # (N_total, embed_dim)
        lengths:      list[int],
    ) -> tuple[torch.Tensor, list]:
        """
        Returns:
            wsi_logits   : Tensor (B, num_classes)
            weights_list : list[B] de Tensor(N_i,) o None según estrategia
        """
        logits_splits = torch.split(patch_logits, lengths, dim=0)
        feats_splits  = torch.split(patch_feats,  lengths, dim=0)

        wsi_logits_list, weights_list = [], []
        for logits_ws, feats_ws in zip(logits_splits, feats_splits):
            wsi_logit, weights = self.strategy(logits_ws, feats_ws)
            wsi_logits_list.append(wsi_logit)
            weights_list.append(weights)

        return torch.stack(wsi_logits_list, dim=0), weights_list


__all__ = [
    "BaseMIL",
    "MaxMIL",
    "MeanMIL",
    "AttentionMIL",
    "TransMIL",
    "MIL_REGISTRY",
    "BAG_BASED_STRATEGIES",
    "MILAggregator",
]
