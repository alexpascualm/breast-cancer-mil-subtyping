"""
keep_encoder.py
---------------
Encoder basado en KEEP (Astaxanthin/KEEP - MAGIC-AI4Med).

KEEP es un modelo visión-lenguaje (ViT-L/16 imagen + BERT texto).
Este encoder expone únicamente la rama de imagen para extraer embeddings
de parches.

Arquitectura interna (del custom code del repo):
    self.visual      : ViT-L/16 (timm) → 1024 dims
    self.visual_head : Linear(1024→768) + GELU + Linear(768→768)
    encode_image()   : visual + visual_head + L2-norm → 768 dims

NOTA: Se instancia directamente sin AutoModel.from_pretrained() para evitar
el bug de meta tensors en transformers>=5.x con modelos que usan timm
internamente en su __init__.

Referencia:
    https://huggingface.co/Astaxanthin/KEEP
    https://github.com/MAGIC-AI4Med/KEEP
"""

import os
import sys
import glob

import torch

from models.patch_encoders.base_encoder import BasePatchEncoder


class KEEPEncoder(BasePatchEncoder):
    """
    Encoder KEEP de MAGIC-AI4Med (rama de imagen).

    Instancia el modelo directamente desde el custom code cacheado por
    HuggingFace, evitando el contexto de meta-tensors de transformers>=5.x
    que es incompatible con timm.

    Args:
        chunk_size (int)         : mini-batches internos de parches.
        unfreeze_backbone (bool) : descongelar los últimos 3 bloques del ViT
                                   y el visual_head.
        keep_model_name (str)    : repo HuggingFace ("Astaxanthin/KEEP").
        patch_size (int)         : resolución de entrada en px (224).
    """

    def __init__(
        self,
        chunk_size: int = 128,
        unfreeze_backbone: bool = False,
        keep_model_name: str = "Astaxanthin/KEEP",
        patch_size: int = 224,
    ):
        super().__init__(chunk_size=chunk_size, unfreeze_backbone=unfreeze_backbone)
        self.keep_model_name = keep_model_name
        self.patch_size = patch_size
        self._ensure_custom_code_cached()
        self.build_backbone()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_custom_code_cached(self) -> None:
        """
        Descarga el código custom y la config de KEEP sin instanciar el modelo.
        Necesario para que el módulo esté disponible en disco antes de importarlo.
        """
        from transformers import AutoConfig
        AutoConfig.from_pretrained(self.keep_model_name, trust_remote_code=True)

    def _get_custom_code_dir(self) -> str:
        """Localiza el directorio del custom code de KEEP en la caché de HF."""
        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/"
            "Astaxanthin/KEEP/*/modeling_keep.py"
        )
        matches = glob.glob(cache_pattern)
        if not matches:
            raise FileNotFoundError(
                "No se encontró el custom code de KEEP en caché. "
                "Asegúrate de tener conexión a internet en el primer uso."
            )
        # Si hay varias versiones, tomar la más reciente
        return os.path.dirname(sorted(matches)[-1])

    # ------------------------------------------------------------------
    # build_backbone
    # ------------------------------------------------------------------

    def build_backbone(self) -> None:
        module_dir = self._get_custom_code_dir()
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        # Importar directamente el custom code — sin pasar por AutoModel,
        # evitando el contexto de meta tensors de transformers>=5.x
        from modeling_keep import KEEPModel, KEEPConfig  # noqa: E402
        from transformers import AutoConfig

        raw_config = AutoConfig.from_pretrained(
            self.keep_model_name,
            trust_remote_code=True,
        )

        # Construir el modelo en CPU real (sin meta tensors)
        keep_config = KEEPConfig(
            vision_config=raw_config.vision_config,
            text_config=raw_config.text_config,
            projection_dim=raw_config.projection_dim,
        )
        self.backbone = KEEPModel(keep_config)

        # Cargar pesos desde HuggingFace Hub
        self._load_weights()

        # embed_dim = salida de encode_image = projection_dim
        self.embed_dim = raw_config.projection_dim  # 768

        # Congelar todos los parámetros
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Descongelar selectivamente si se solicita
        if self.unfreeze_backbone:
            for param in self.backbone.visual.blocks[-3:].parameters():
                param.requires_grad = True
            for param in self.backbone.visual.norm.parameters():
                param.requires_grad = True
            for param in self.backbone.visual_head.parameters():
                param.requires_grad = True

    def _load_weights(self) -> None:
        """Descarga y carga los pesos del modelo desde HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        # Intentar safetensors primero, caer a pytorch_model.bin si no existe
        for filename in ("model.safetensors", "pytorch_model.bin"):
            try:
                weights_path = hf_hub_download(
                    repo_id=self.keep_model_name,
                    filename=filename,
                )
                break
            except Exception:
                weights_path = None

        if weights_path is None:
            raise FileNotFoundError(
                f"No se encontraron pesos en {self.keep_model_name}. "
                "Comprueba el nombre del repo y la conexión a internet."
            )

        if weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        else:
            state_dict = torch.load(weights_path, map_location="cpu")

        self.backbone.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def extract_features(self, chunk_gpu: torch.Tensor) -> torch.Tensor:
        """
        Extrae embeddings de imagen para un chunk de parches.

        Args:
            chunk_gpu : Tensor (N, 3, 224, 224) normalizado, en GPU.

        Returns:
            Tensor (N, 768) — embedding L2-normalizado (espacio CLIP proyectado).
        """
        # encode_image: visual(x) → visual_head → L2-norm → (N, 768)
        return self.backbone.encode_image(chunk_gpu)
