"""
models/patch_encoders/dino_encoder.py
--------------------------------------
Encoder ViT DINO via timm.

Modelos habituales:
    "vit_base_patch16_224.dino"        - ViT-B/16 DINO
"""

import torch
import timm

from models.patch_encoders.base_encoder import BasePatchEncoder


class DINOEncoder(BasePatchEncoder):
    """
    Extractor de features ViT DINO.

    Args:
        chunk_size (int)           : mini-batch interno para control de VRAM.
        unfreeze_backbone (bool)   : descongelar los últimos 3 bloques del ViT.
        dino_model_name (str)      : nombre del modelo en timm.
        patch_size (int)           : resolución de entrada (px).
    """

    def __init__(
        self,
        chunk_size: int = 128,
        unfreeze_backbone: bool = False,
        dino_model_name: str = "vit_base_patch16_224.dino",
        patch_size: int = 224,
    ):
        super().__init__(chunk_size=chunk_size, unfreeze_backbone=unfreeze_backbone)
        self.dino_model_name    = dino_model_name
        self.patch_size         = patch_size
        self.build_backbone()

    def build_backbone(self) -> None:
        self.backbone = timm.create_model(self.dino_model_name, pretrained=True, img_size=self.patch_size, num_classes=0)
        self.embed_dim = self.backbone.num_features

        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.unfreeze_backbone:
            for param in self.backbone.blocks[-3:].parameters():   # últimos 3 bloques
                param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            self.backbone.set_grad_checkpointing(enable=True)

    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(chunk_gpu)   # (N, N_tokens+1, D)
        return feats[:, 0, :]                           # CLS token
