"""
base_encoder.py
---------------
Clase base abstracta para todos los encoders de parches.
Cualquier encoder nuevo debe heredar de BasePatchEncoder e implementar
los métodos abstractos.
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

        # Asignados en build_backbone()
        self.backbone:  nn.Module = None
        self.embed_dim: int       = None

    # ------------------------------------------------------------------
    # API que cada subclase DEBE implementar
    # ------------------------------------------------------------------

    @abstractmethod
    def build_backbone(self) -> None:
        """
        Instancia self.backbone y asigna self.embed_dim.
        Aplica freeze / unfreeze según self.unfreeze_backbone.
        """
        ...

    @abstractmethod
    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        """
        Recibe un mini-batch de parches (N, C, H, W) ya en GPU.
        Devuelve embeddings (N, embed_dim).
        """
        ...

    # ------------------------------------------------------------------
    # Cabeza de clasificación y MIL (compartidos)
    # ------------------------------------------------------------------

    def _build_head_and_mil(self):
        """
        Llama DESPUÉS de build_backbone() para construir el clasificador
        de parche y el módulo MIL.
        """
        from models.mil import MILAggregator

        assert self.embed_dim is not None, (
            "embed_dim debe asignarse en build_backbone() antes de llamar a _build_head_and_mil()"
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

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
            inputs_list: list[Tensor], cada uno (N_patches, C, H, W) para una WSI.

        Returns:
            wsi_logits        : Tensor (B, num_classes)
            patch_logits_list : list[B Tensors] de shape (N_patches_i, num_classes)
            weights_list      : list[B Tensors] o list[None]
        """
        device = next(self.parameters()).device

        lengths = [x.size(0) for x in inputs_list]
        if sum(lengths) == 0:
            raise RuntimeError("El batch no contiene parches.")

        all_patches_cpu = torch.cat(inputs_list, dim=0)
        chunks = torch.split(all_patches_cpu, self.chunk_size, dim=0)

        all_embeddings, all_logits = [], []
        for chunk in chunks:
            chunk_gpu = chunk.to(device, non_blocking=True)
            emb    = self.extract_features(chunk_gpu)   # (N_chunk, embed_dim)
            logits = self.classifier(emb)               # (N_chunk, num_classes)
            all_embeddings.append(emb)
            all_logits.append(logits)

        total_feats  = torch.cat(all_embeddings, dim=0)    # (N_total, embed_dim)
        total_logits = torch.cat(all_logits,     dim=0)    # (N_total, num_classes)

        wsi_logits, weights_list = self.mil(total_logits, total_feats, lengths)
        patch_logits_list = list(torch.split(total_logits, lengths, dim=0))

        return wsi_logits, patch_logits_list, weights_list
