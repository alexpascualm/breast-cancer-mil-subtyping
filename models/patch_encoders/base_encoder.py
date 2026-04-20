"""
models/patch_encoders/base_encoder.py
--------------------------------------
Clase base abstracta para extractores de características de parches.

Responsabilidad única: dado un batch de parches (N, C, H, W),
devolver embeddings (N, embed_dim).

"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BasePatchEncoder(ABC, nn.Module):
    """
    Interfaz común para todos los extractores de características de parches.

    Subclases obligadas a implementar:
        - build_backbone() → instancia self.backbone y asigna self.embed_dim
        - extract_features(chunk_gpu) → Tensor (N, embed_dim)

    El chunking para control de VRAM se gestiona aquí via encode().
    """

    def __init__(self, chunk_size: int, unfreeze_backbone: bool):
        super().__init__()
        self.chunk_size        = chunk_size
        self.unfreeze_backbone = unfreeze_backbone
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
    # Método público: extrae features de un tensor grande con chunking
    # ------------------------------------------------------------------

    def forward(self, patches_cpu: torch.Tensor) -> torch.Tensor:
        """
        Extrae embeddings de un tensor de parches con control de VRAM.

        Args:
            patches_cpu: Tensor (N, C, H, W) en CPU.

        Returns:
            Tensor (N, embed_dim) en el dispositivo del modelo.
        """
        device = next(self.parameters()).device
        chunks = torch.split(patches_cpu, self.chunk_size, dim=0)
        embeddings = []
        for chunk in chunks:
            emb = self.extract_features(chunk.to(device, non_blocking=True))
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)

