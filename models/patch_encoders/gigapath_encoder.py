"""
models/patch_encoders/gigapath_encoder.py
------------------------------------------
Encoder basado en GigaPath (prov-gigapath/prov-gigapath).

GigaPath es un ViT-G/14 entrenado por Microsoft Research sobre 1.300 millones
de parches histopatológicos procedentes de 171.189 WSIs del Providence Health System.

Acceso:
    El modelo es de acceso restringido en HuggingFace. Requiere:
    1. Solicitar acceso en https://huggingface.co/prov-gigapath/prov-gigapath
    2. Pasar el token HF con permisos al arrancar con --hf_token desde train.py / extract_embeddings.py.

Instalación:
    pip install timm huggingface_hub

Referencia:
    Xu, H., Usuyama, N., Bagga, J. et al. "A whole-slide foundation model for digital pathology from real-world data" Nature 630, 181-188 (2024).
    https://doi.org/10.1038/s41586-024-07441-w
    https://huggingface.co/prov-gigapath/prov-gigapath
"""

import torch
import timm

from models.patch_encoders.base_encoder import BasePatchEncoder


class GigaPathEncoder(BasePatchEncoder):
    """
    Encoder GigaPath de Microsoft Research / Providence Health System.

    Carga el tile-encoder ViT-G/14 directamente desde HuggingFace Hub
    a través de timm. Devuelve el CLS token (1536 dims) como embedding
    de cada parche.

    Args:
        chunk_size (int)              : mini-batches internos para control de VRAM.
                                        ViT-G es grande — reducir a 16 o incluso 8 si hay OOM con la GPU disponible.
        unfreeze_backbone (bool)      : si True, descongela los últimos 3 bloques del
                                        ViT-G y la capa de normalización final.
        gigapath_model_name (str)     : repo HuggingFace. Por defecto el oficial.
        patch_size (int)              : resolución de entrada esperada (224 px).
                                        Se almacena como referencia; el redimensionado
                                        real lo controla args.patch_resize_size.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        unfreeze_backbone: bool = False,
        gigapath_model_name: str = "prov-gigapath/prov-gigapath",
        patch_size: int = 224, #TODO Revisar (HF dice 256 pero luego hacen center crop a 224)
    ):
        super().__init__(chunk_size=chunk_size, unfreeze_backbone=unfreeze_backbone)
        self.gigapath_model_name = gigapath_model_name
        self.patch_size          = patch_size
        self.build_backbone()

    # ------------------------------------------------------------------
    # build_backbone
    # ------------------------------------------------------------------

    def build_backbone(self) -> None:
        """
        Instancia el ViT-G/14 de GigaPath vía timm y configura freeze/unfreeze.

        timm carga el modelo directamente desde HuggingFace Hub con el prefijo
        "hf-hub:". No se necesitan argumentos adicionales de inicialización
        (a diferencia de UNI, que requiere init_values=1e-5).

        embed_dim se lee de num_features, que timm asigna al instanciar el modelo
        (1536 para ViT-G/14).
        """
        self.backbone = timm.create_model(
            f"hf-hub:{self.gigapath_model_name}",
            pretrained=True,
            num_classes=0,          # elimina la cabeza de clasificación
        )

        self.embed_dim = self.backbone.num_features   # 1536

        # Congelar todos los parámetros por defecto
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar selectivamente si se solicita fine-tuning parcial
        if self.unfreeze_backbone:
            # ViT-G/14 tiene 40 bloques; descongelamos los 3 últimos
            for param in self.backbone.blocks[-3:].parameters():
                param.requires_grad = True
            # Normalización final (equivalente a backbone.norm en timm ViTs)
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # extract_features
    # ------------------------------------------------------------------

    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        """
        Extrae el CLS token de GigaPath para un chunk de parches.

        Args:
            chunk_gpu : Tensor (N, 3, 224, 224) normalizado ImageNet, en GPU.

        Returns:
            Tensor (N, 1536) — embeddings crudos (sin normalización L2).
        """
        # forward_features devuelve (N, N_tokens + 1, D) en timm ViTs estándar.
        # El CLS token siempre ocupa la posición 0.
        feats = self.backbone.forward_features(chunk_gpu)   # (N, N_tokens+1, 1536)
        return feats[:, 0, :]                                # CLS token → (N, 1536)
