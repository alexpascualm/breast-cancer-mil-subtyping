"""
models/wsi_classifier.py
-------------------------
Modelo principal que combina:
    - Un extractor de features de parches (patch_encoder)
    - Una estrategia de agregación MIL
    - Un clasificador (de parche o de WSI según la estrategia)

Dos modos de clasificación:

    INSTANCE-BASED (max-instance-level, mean-instance-level, abMIL-instance-level):
        patch → encoder → embedding → clasificador_parche → logit_parche
        logits_parche x N → MIL → logit_WSI

    BAG-BASED (max-bag-level, mean-bag-level, abMIL-bag-level, transMIL):
        patch → encoder → embedding
        embeddings x N → MIL (con clasificador interno) → logit_WSI
"""

import torch
import torch.nn as nn

from models.patch_encoders import BasePatchEncoder
from models.mil_aggregators import MILAggregator, BAG_BASED_STRATEGIES


class WSIClassifier(nn.Module):
    """
    Clasificador de WSI de extremo a extremo.

    Args:
        encoder (BasePatchEncoder)  : extractor de features ya instanciado.
        mil_aggregation_mode (str)  : clave en MIL_REGISTRY.
        num_classes (int)           : número de clases de salida.
        chunk_size (int)            : mini-batch interno de parches (control VRAM).
        embed_dim (int, optional)   : dimensión de los embeddings del encoder.
                                      Si se omite, se asume encoder.embed_dim.
    """

    def __init__(
        self,
        encoder: BasePatchEncoder | None,
        mil_aggregation_mode: str,
        num_classes: int,
        chunk_size: int = 512,
        embed_dim: int | None = None,   # requerido solo si encoder=None
    ):
        super().__init__()

        self.encoder   = encoder
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.is_bag_based = mil_aggregation_mode in BAG_BASED_STRATEGIES
        
        if encoder is None:
            assert embed_dim is not None, ("embed_dim es obligatorio cuando encoder=None (modo skip_encoder).")
        else:
            embed_dim = encoder.embed_dim
            assert embed_dim is not None, ("El encoder debe tener embed_dim asignado tras build_backbone().")
        
        # Clasificador de parche — solo para estrategias instance-based
        if not self.is_bag_based:
            self.patch_classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(embed_dim, num_classes))
        else:
            self.patch_classifier = None

        # Módulo MIL (para bag-based, incluye su propio clasificador de WSI)
        self.mil = MILAggregator(
            aggregation=mil_aggregation_mode,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

    # ------------------------------------------------------------------
    # El forward normal asume que se le pasan imágenes de parches sin procesar y se encarga de extraer features y clasificar.
    def forward(self, inputs_list: list[torch.Tensor]):
        """
        Args:
            inputs_list : list[Tensor] — un Tensor (N_i, C, H, W) por WSI del batch.

        Returns:
            wsi_logits        : Tensor (B, num_classes)
            patch_logits_list : list[B] de Tensor(N_i, num_classes), o list[None] si bag-based
            weights_list      : list[B] de Tensor(N_i,) o None según estrategia MIL
        """
        lengths = [x.size(0) for x in inputs_list]
        if sum(lengths) == 0:
            raise RuntimeError("El batch no contiene parches.")

        # ── Extracción de features ────────────────────────────────────────
        all_patches_cpu = torch.cat(inputs_list, dim=0)   # (N_total, C, H, W)

        if not self.encoder.unfreeze_backbone:
            with torch.no_grad():
                total_feats = self.encoder.forward(all_patches_cpu)  # (N_total, embed_dim)
        else:
            total_feats = self.encoder.forward(all_patches_cpu)  # (N_total, embed_dim)
        
        # ── Clasificación / agregación ────────────────────────────────────
        if self.is_bag_based:
            # El MIL agrega en espacio de features con su propio clasificador
            placeholder = torch.zeros(
                total_feats.size(0), self.num_classes, device=total_feats.device
            )
            wsi_logits, weights_list = self.mil(placeholder, total_feats, lengths)
            patch_logits_list = [None] * len(lengths)

        else:
            # Clasificador de parche → logits por parche → MIL agrega
            total_logits = self.patch_classifier(total_feats)   # (N_total, num_classes)
            wsi_logits, weights_list = self.mil(total_logits, total_feats, lengths)
            patch_logits_list = list(torch.split(total_logits, lengths, dim=0))

        return wsi_logits, patch_logits_list, weights_list

    # forward_from_embeddings asume que los embeddings ya están precomputados y se le pasan directamente, saltándose el encoder.
    def forward_from_embeddings(self, embeddings_list: list[torch.Tensor]):
        """
        Forward pass cuando los embeddings ya están precomputados.
        Salta el encoder completamente.

        Args:
            embeddings_list : list[Tensor(N_i, embed_dim)] — uno por WSI del batch.
                            Los tensores ya deben estar en el dispositivo correcto.

        Returns:
            wsi_logits, patch_logits_list, weights_list  (igual que forward)
        """
        lengths = [x.size(0) for x in embeddings_list]
        if sum(lengths) == 0:
            raise RuntimeError("El batch no contiene embeddings.")

        total_feats = torch.cat(embeddings_list, dim=0)   # (N_total, embed_dim)

        if self.is_bag_based:
            placeholder = torch.zeros(
                total_feats.size(0), self.num_classes, device=total_feats.device
            )
            wsi_logits, weights_list = self.mil(placeholder, total_feats, lengths)
            patch_logits_list = [None] * len(lengths)
        else:
            total_logits = self.patch_classifier(total_feats)
            wsi_logits, weights_list = self.mil(total_logits, total_feats, lengths)
            patch_logits_list = list(torch.split(total_logits, lengths, dim=0))

        return wsi_logits, patch_logits_list, weights_list
