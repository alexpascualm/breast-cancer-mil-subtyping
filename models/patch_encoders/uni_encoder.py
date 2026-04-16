"""
uni_encoder.py
--------------
Encoder basado en UNI (MahmoodLab/UNI - MahmoodLab/UNI2-h).

Requiere:
    pip install timm huggingface_hub

El modelo es de acceso restringido en HuggingFace — es necesario:
    1. Aceptar las condiciones en https://huggingface.co/MahmoodLab/UNI
    2. Pasar un token HF con acceso al modelo:
       huggingface_hub.login(token="hf_...")
"""

import torch
import timm

from models.patch_encoders.base_encoder import BasePatchEncoder

class UNIEncoder(BasePatchEncoder):
    """
    Encoder UNI de MahmoodLab.

    Args:
        chunk_size (int)            : mini-batches internos de parches.
        unfreeze_backbone (bool)    : descongelar los últimos 3 bloques.
        uni_model_name (str)        : repo HuggingFace ("MahmoodLab/UNI" o "MahmoodLab/UNI2-h").
        patch_size (int)            : resolución de entrada (px). UNI usa 224 por defecto.
    """

    def __init__(
        self,
        chunk_size: int = 128,
        unfreeze_backbone: bool = False,
        uni_model_name: str = "MahmoodLab/UNI",
        patch_size: int = 224,
    ):
        super().__init__(chunk_size=chunk_size, unfreeze_backbone=unfreeze_backbone)
        self.uni_model_name = uni_model_name
        self.patch_size = patch_size
        self.build_backbone()

    # ------------------------------------------------------------------

    def build_backbone(self) -> None:
        self.backbone = timm.create_model(
            f"hf-hub:{self.uni_model_name}", 
            pretrained=True, 
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0)
        
        self.embed_dim = self.backbone.num_features

        # Congelar parámetros (por defecto vienen desbloqueados)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar selectivamente si se solicita
        if self.unfreeze_backbone:
            for param in self.backbone.blocks[-3:].parameters():  # últimos 3 bloques
                param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------

    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        # UNI expone forward_features estándar de timm → CLS token en pos 0
        feats = self.backbone.forward_features(chunk_gpu)   # (N, N_tokens+1, D)
        return feats[:, 0, :]   # CLS token
