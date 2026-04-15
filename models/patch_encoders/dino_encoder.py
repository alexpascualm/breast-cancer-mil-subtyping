"""
dino_encoder.py
---------------
Encoder basado en ViT preentrenado con DINO / DINOv2 a través de timm.

Modelos habituales:
    "vit_base_patch16_224.dino"       - ViT-B/16  DINO
    "vit_large_patch14_dinov2.lvd142m" - ViT-L/14 DINOv2

"""

import torch
import timm

from models.patch_encoders.base_encoder import BasePatchEncoder


class DINOEncoder(BasePatchEncoder):
    """
    Encoder ViT DINO/DINOv2.

    Args:
        num_classes (int)           : número de clases de salida.
        mil_aggregation_mode (str)  : 'max' | 'mean' | 'attentionMIL'.
        chunk_size (int)            : mini-batches internos de parches (control VRAM).
        unfreeze_backbone (bool)    : descongelar los últimos 3 bloques del ViT.
        dino_model_name (str)       : nombre del modelo en timm.
        patch_size (int)            : resolución de entrada de los parches (px).
        patch_embedding_mode (int)  : 0 → CLS token | 1 → mean pooling de tokens espaciales.
    """

    def __init__(
        self,
        num_classes: int = 2,
        mil_aggregation_mode: str = "attentionMIL",
        chunk_size: int = 128,
        unfreeze_backbone: bool = False,
        dino_model_name: str = "vit_base_patch16_224.dino",
        patch_size: int = 224,
        patch_embedding_mode: int = 0,
    ):
        super().__init__(
            num_classes=num_classes,
            mil_aggregation_mode=mil_aggregation_mode,
            chunk_size=chunk_size,
            unfreeze_backbone=unfreeze_backbone,
        )
        self.dino_model_name = dino_model_name
        self.patch_size = patch_size
        self.patch_embedding_mode = patch_embedding_mode

        self.build_backbone()
        self._build_head_and_mil()

    # ------------------------------------------------------------------

    def build_backbone(self) -> None:
        self.backbone = timm.create_model(self.dino_model_name, pretrained=True, img_size=self.patch_size)
        self.backbone.reset_classifier(0)
        self.embed_dim = self.backbone.num_features

        # Congelar todo por defecto
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar selectivamente si se solicita
        if self.unfreeze_backbone:
            for param in self.backbone.blocks[-3:].parameters():   # últimos 3 bloques
                param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            self.backbone.set_grad_checkpointing(enable=True)

    # ------------------------------------------------------------------

    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(chunk_gpu)   # (N, N_tokens+1, D)

        if self.patch_embedding_mode == 0:
            return feats[:, 0, :]           # CLS token
        elif self.patch_embedding_mode == 1:
            return feats[:, 1:, :].mean(dim=1)   # mean pooling sin CLS
        else:
            raise ValueError(
                f"patch_embedding_mode desconocido: {self.patch_embedding_mode}. "
                "Valores válidos: 0 (CLS token) | 1 (mean pooling)."
            )
