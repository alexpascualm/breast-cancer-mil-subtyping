"""
models/patch_encoders/__init__.py
----------------------------------
Registro central de encoders de parches disponibles.

Para añadir un nuevo encoder:
    1. Crear su módulo en esta carpeta heredando de BasePatchEncoder.
    2. Añadir la clave y la clase al diccionario ENCODER_REGISTRY.
"""

from models.patch_encoders.base_encoder   import BasePatchEncoder
from models.patch_encoders.dino_encoder   import DINOEncoder
from models.patch_encoders.uni_encoder    import UNIEncoder
# from models.patch_encoders.conch_encoder  import CONCHEncoder
# from models.patch_encoders.keep_encoder   import KEEPEncoder

# Registro: clave CLI → clase
ENCODER_REGISTRY: dict[str, type[BasePatchEncoder]] = {
    "dino":  DINOEncoder,
    "uni":   UNIEncoder,
    # "uni2":  UNIEncoder,
    # "conch": CONCHEncoder,
    # "keep":  KEEPEncoder,
}

__all__ = [
    "BasePatchEncoder",
    "DINOEncoder",
    "UNIEncoder",
    # "CONCHEncoder",
    # "KEEPEncoder",
    "ENCODER_REGISTRY",
]
