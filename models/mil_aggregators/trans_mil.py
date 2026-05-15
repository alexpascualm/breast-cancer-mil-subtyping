"""
models/mil/trans_mil.py
-----------------------
TransMIL adaptado a la interfaz BaseMIL del proyecto.

Referencia:
    Shao et al. "TransMIL: Transformer based Correlated Multiple Instance
    Learning for Whole Slide Image Classification." NeurIPS 2021.
    https://github.com/szc19990412/TransMIL

Requisito:
    pip install nystrom-attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.mil_aggregators.base_mil import BaseMIL

try:
    from nystrom_attention import NystromAttention
except ImportError as e:
    raise ImportError(
        "TransMIL requiere el paquete 'nystrom-attention'.\n"
        "Instálalo con: pip install nystrom-attention\n"
        f"Error original: {e}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Bloques internos de TransMIL (adaptados del código original de Shao et al.)
# ──────────────────────────────────────────────────────────────────────────────

class _TransLayer(nn.Module):
    """Un bloque Transformer con Nystrom self-attention + LayerNorm."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class _PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator.
    Recoloca los tokens en 2D, aplica convoluciones de distintos tamaños
    para inyectar información espacial local, y vuelve a aplanar.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        # Tres convoluciones depthwise con distintos campos receptivos
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H: int, W: int):
        B, _, C = x.shape
        cls_token, feat_tokens = x[:, 0], x[:, 1:]          # separa CLS

        cnn_feat = feat_tokens.transpose(1, 2).view(B, C, H, W)
        pe = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        pe = pe.flatten(2).transpose(1, 2)                   # (B, H*W, C)

        return torch.cat((cls_token.unsqueeze(1), pe), dim=1)  # reune CLS + patches


# ──────────────────────────────────────────────────────────────────────────────
# Módulo TransMIL adaptado a BaseMIL
# ──────────────────────────────────────────────────────────────────────────────

class TransMIL(BaseMIL):
    """
    TransMIL adaptado a la interfaz BaseMIL del proyecto.

    A diferencia de AttentionMIL, este módulo:
      1. Proyecta patch_feats a dim_model (512 por defecto).
      2. Hace pad cuadrado de la secuencia de parches.
      3. Añade un CLS token aprendible.
      4. Pasa por TransLayer1 → PPEG → TransLayer2.
      5. Extrae el CLS token como embedding de WSI.
      6. Aplica el clasificador sobre ese CLS token.

    IMPORTANTE: ignora patch_logits — agrega en el espacio de features.
    El clasificador de WSI vive AQUÍ, no en BasePatchEncoder.

    Args:
        num_classes (int) : número de clases de salida.
        embed_dim (int)   : dimensión de los embeddings del encoder (entrada).
        dim_model (int)   : dimensión interna del Transformer (default 512).
    """

    def __init__(self, num_classes: int, embed_dim: int, dim_model: int = 512):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

        self.dim_model = dim_model

        # Proyección de entrada: embed_dim → dim_model
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, dim_model),
            nn.ReLU(inplace=True),
        )

        # CLS token aprendible
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))

        # Bloques Transformer
        self.layer1 = _TransLayer(dim=dim_model)
        self.layer2 = _TransLayer(dim=dim_model)

        # Positional encoding espacial
        self.pos_layer = _PPEG(dim=dim_model)

        # Normalización final
        self.norm = nn.LayerNorm(dim_model)

        # Clasificador de WSI (sobre CLS token)
        self.bag_classifier = nn.Linear(dim_model, num_classes)
        # self.bag_classifier = nn.Sequential(
        #                             nn.Linear(dim_model, dim_model//2),
        #                             nn.ReLU(inplace=True),
        #                             nn.Dropout(0.2),
        #                             nn.Linear(dim_model//2, num_classes)
        #                             ) 

    # ------------------------------------------------------------------

    def forward(
        self,
        logits_ws: torch.Tensor,   # (N, num_classes) — NO se usa
        feats_ws:  torch.Tensor,   # (N, embed_dim)   — entrada real
    ):
        """
        Agrega los embeddings de parche de UNA WSI con Transformer.

        Args:
            logits_ws : ignorado (requerido por la interfaz BaseMIL).
            feats_ws  : Tensor (N, embed_dim) con los embeddings del encoder.

        Returns:
            wsi_logit : Tensor (num_classes,)
            weights   : None  (TransMIL no produce pesos escalares por parche)
        """
        if feats_ws.numel() == 0:
            return torch.zeros(self.num_classes, device=feats_ws.device), None

        # 1. Proyección de entrada
        h = self.input_proj(feats_ws)        # (N, dim_model)
        h = h.unsqueeze(0)                   # (1, N, dim_model)  ← batch dim

        # 2. Pad cuadrado (TransMIL necesita secuencia de longitud N² para PPEG)
        N = h.shape[1]
        _H = _W = math.ceil(math.sqrt(N))
        pad_len = _H * _W - N
        if pad_len > 0:
            h = torch.cat([h, h[:, :pad_len, :]], dim=1) # opción original: repite los primeros tokens para rellenar
            # pad_tensor = torch.zeros((1, pad_len, h.size(-1)),device=h.device,dtype=h.dtype)
            # h = torch.cat([h, pad_tensor], dim=1) # opción alternativa: rellena con ceros
    

        # 3. CLS token
        cls = self.cls_token.to(h.device)   # (1, 1, dim_model)
        h = torch.cat([cls, h], dim=1)      # (1, N_pad+1, dim_model)

        # 4. TransLayer 1 — correlación morfológica entre instancias
        h = self.layer1(h)                  # (1, N_pad+1, dim_model)

        # 5. PPEG — positional encoding espacial condicional
        h = self.pos_layer(h, _H, _W)      # (1, N_pad+1, dim_model)

        # 6. TransLayer 2 — correlación profunda
        h = self.layer2(h)                  # (1, N_pad+1, dim_model)

        # 7. Extrae CLS token y normaliza
        cls_out = self.norm(h)[:, 0, :]    # (1, dim_model)

        # 8. Clasificador de WSI
        wsi_logit = self.bag_classifier(cls_out).squeeze(0)   # (num_classes,)

        return wsi_logit, None
