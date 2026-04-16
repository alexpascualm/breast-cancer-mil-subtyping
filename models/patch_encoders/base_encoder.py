"""
models/patch_encoders/base_encoder.py
--------------------------------------
Clase base abstracta para todos los encoders de parches.

Soporta dos familias de estrategias MIL:

    INSTANCE-BASED (max, mean, attentionMIL):
        El encoder aplica un clasificador por parche → obtiene patch_logits.
        El MIL agrega esos logits en un logit de WSI.

    BAG-BASED (transMIL):
        El encoder solo extrae embeddings.
        El MIL lleva su propio clasificador de WSI y agrega en el espacio
        de features. El clasificador de parche NO se construye ni se usa.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BasePatchEncoder(ABC, nn.Module):
    """
    Interfaz común para todos los extractores de características de parches.

    Subclases obligadas a implementar:
        - build_backbone()         → construye y congela/descongela el backbone
        - extract_features(chunk)  → Tensor (N, embed_dim)

    La lógica de chunking, clasificación por parche y MIL se mantiene
    en esta clase para evitar duplicación entre encoders.
    """

    def __init__(
        self,
        num_classes: int,
        mil_aggregation_mode: str,
        chunk_size: int,
        unfreeze_backbone: bool,
    ):
        super().__init__()
        self.num_classes          = num_classes
        self.mil_aggregation_mode = mil_aggregation_mode
        self.chunk_size           = chunk_size
        self.unfreeze_backbone    = unfreeze_backbone

        self.backbone:    nn.Module      = None
        self.embed_dim:   int            = None
        self.classifier:  nn.Module      = None   # None si es bag-based
        self.is_bag_based: bool          = False

    # ------------------------------------------------------------------
    # API que cada subclase DEBE implementar
    # ------------------------------------------------------------------

    @abstractmethod
    def build_backbone(self) -> None:
        ...

    @abstractmethod
    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        ...

    # ------------------------------------------------------------------
    # Cabeza de clasificación y MIL (compartidos)
    # ------------------------------------------------------------------

    def _build_head_and_mil(self):
        from models.mil_aggregators import MILAggregator, BAG_BASED_STRATEGIES

        assert self.embed_dim is not None

        self.is_bag_based = self.mil_aggregation_mode in BAG_BASED_STRATEGIES

        if not self.is_bag_based:
            # Clasificador de parche solo para estrategias instance-based
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        self.mil = MILAggregator(
            aggregation=self.mil_aggregation_mode,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
        )

    # ------------------------------------------------------------------
    # Forward unificado
    # ------------------------------------------------------------------

    def forward(self, inputs_list):
        """
        Args:
            inputs_list: list[Tensor], cada uno (N_patches, C, H, W) por WSI.

        Returns:
            wsi_logits        : Tensor (B, num_classes)
            patch_logits_list : list[B Tensors] (N_i, num_classes) o list[None] si bag-based
            weights_list      : list[B Tensors] o list[None]
        """
        device = next(self.parameters()).device

        lengths = [x.size(0) for x in inputs_list]
        if sum(lengths) == 0:
            raise RuntimeError("El batch no contiene parches.")

        all_patches_cpu = torch.cat(inputs_list, dim=0)
        chunks = torch.split(all_patches_cpu, self.chunk_size, dim=0)

        all_embeddings = []
        all_logits     = []

        for chunk in chunks:
            chunk_gpu = chunk.to(device, non_blocking=True)
            emb = self.extract_features(chunk_gpu)          # (N_chunk, embed_dim)
            all_embeddings.append(emb)

            if not self.is_bag_based:
                logits = self.classifier(emb)               # (N_chunk, num_classes)
                all_logits.append(logits)

        total_feats = torch.cat(all_embeddings, dim=0)      # (N_total, embed_dim)

        if self.is_bag_based:
            # Para bag-based: el MIL usa feats; patch_logits es un placeholder vacío
            placeholder = torch.zeros(
                total_feats.size(0), self.num_classes, device=device
            )
            wsi_logits, weights_list = self.mil(placeholder, total_feats, lengths)
            patch_logits_list = [None] * len(lengths)
        else:
            total_logits = torch.cat(all_logits, dim=0)     # (N_total, num_classes)
            wsi_logits, weights_list = self.mil(total_logits, total_feats, lengths)
            patch_logits_list = list(torch.split(total_logits, lengths, dim=0))

        return wsi_logits, patch_logits_list, weights_list
