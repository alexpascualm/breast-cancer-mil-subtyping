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
# Bloques internos de TransMIL (Shao et al., NeurIPS 2021)
# ──────────────────────────────────────────────────────────────────────────────

class _TransLayer(nn.Module):
    """Bloque Transformer con Nystrom self-attention + LayerNorm."""

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
    para inyectar información posicional local, y vuelve a aplanar.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        # Tres convoluciones depthwise con distintos campos receptivos
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H: int, W: int):
        B, _, C = x.shape
        cls_token, feat_tokens = x[:, 0], x[:, 1:]          # Separa CLS y patches

        cnn_feat = feat_tokens.transpose(1, 2).view(B, C, H, W)
        pe = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        pe = pe.flatten(2).transpose(1, 2)  # (B, H*W, C)

        return torch.cat((cls_token.unsqueeze(1), pe), dim=1) # Reune CLS + patches


# ──────────────────────────────────────────────────────────────────────────────
# Módulo de atención cruzada CLS → parches (solo interpretabilidad)
# ──────────────────────────────────────────────────────────────────────────────

class _CLSCrossAttention(nn.Module):
    """
    Cross-attention de un único paso: el CLS token actúa como query
    y los patch tokens como keys/values.

    Produce un peso escalar por parche que indica cuánto contribuye
    cada uno a la representación del CLS, de forma análoga a la
    atención en ViT/DINO para interpretabilidad.

    IMPORTANTE: Este módulo se llama siempre bajo torch.no_grad()
    desde TransMIL.forward(). Sus parámetros NO reciben gradiente
    de la loss de clasificación.
    Args:
        dim   : dimensión del espacio de tokens (= dim_model).
        heads : número de cabezas de atención (por defecto 8).
    """

    def __init__(self, dim: int = 512, heads: int = 8):
        super().__init__()
        assert dim % heads == 0, "dim debe ser divisible por heads"
        self.heads    = heads
        self.dim_head = dim // heads
        self.scale    = self.dim_head ** -0.5

        self.to_q   = nn.Linear(dim, dim, bias=False)
        self.to_k   = nn.Linear(dim, dim, bias=False)
        self.to_v   = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        cls_token:    torch.Tensor,  # (1, 1,     dim)
        patch_tokens: torch.Tensor,  # (1, N_pad, dim)
    ) -> torch.Tensor:
        """
        Devuelve:
            weights : (N_pad,) — pesos de atención promediados entre cabezas.

        Solo devuelve los pesos; el CLS enriquecido NO se usa fuera de aquí.
        """
        H = self.heads
        D = self.dim_head

        q = self.to_q(cls_token)     # (1, 1,     dim)
        k = self.to_k(patch_tokens)  # (1, N_pad, dim)
        v = self.to_v(patch_tokens)  # (1, N_pad, dim)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            B_, S, _ = t.shape
            return t.view(B_, S, H, D).transpose(1, 2)  # (B, H, S, D)

        q = split_heads(q)  # (1, H, 1,     D)
        k = split_heads(k)  # (1, H, N_pad, D)
        v = split_heads(v)  # (1, H, N_pad, D)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (1, H, 1, N_pad)
        attn   = scores.softmax(dim=-1)

        # Promedio entre cabezas → (N_pad,)
        weights = attn.mean(dim=1).squeeze(0).squeeze(0)

        return weights


# ──────────────────────────────────────────────────────────────────────────────
#  Módulo TransMIL adaptado a BaseMIL — fiel al paper, con interpretabilidad opcional
# ──────────────────────────────────────────────────────────────────────────────

class TransMIL(BaseMIL):
    """
    TransMIL adaptado a la interfaz BaseMIL del proyecto.

    Pipeline (fiel al paper de Shao et al.):
        1. Proyección embed_dim → dim_model (512 por defecto).
        2. Pad cuadrado de la secuencia de patches.
        3. CLS token aprendible prepended.
        4. TransLayer1 (Nystrom self-attention).
        5. PPEG (positional encoding condicional).
        6. TransLayer2 (segunda ronda de correlación).
        7. LayerNorm → extrae CLS token.
        8. Clasificador MLP sobre el CLS token.  ← igual que el paper
        9. (Extra no presente en el paper) CLSCrossAttention sobre los patch tokens procesados,
           bajo no_grad, solo para devolver pesos de interpretabilidad.

         IMPORTANTE: ignora patch_logits — agrega en el espacio de features.

    Args:
        num_classes (int)      : número de clases de salida.
        embed_dim (int)        : dimensión de los embeddings del encoder (entrada).
        dim_model (int)        : dimensión interna del Transformer (default 512).
        ca_heads (int)         : cabezas de la cross-attention (default 8).
    """

    def __init__(self, num_classes: int, embed_dim: int, dim_model: int  = 512, ca_heads: int  = 8):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

        self.dim_model = dim_model

        # Proyección de entrada: embed_dim → dim_model
        if dim_model != embed_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(embed_dim, dim_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_proj = nn.Identity()

        # ── CLS token aprendible ──────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))

        # ── Bloques Transformer ───────────────────────────────────────────────
        self.layer1 = _TransLayer(dim=dim_model)
        self.layer2 = _TransLayer(dim=dim_model)

        # ── Positional encoding espacial ──────────────────────────────────────
        self.pos_layer = _PPEG(dim=dim_model)

        # ── Normalización final ───────────────────────────────────────────────
        self.norm = nn.LayerNorm(dim_model)

        # ── Clasificador de WSI (sobre CLS token directo, igual que el paper) ─
        self.bag_classifier = nn.Linear(dim_model, num_classes)

        # ── Cross-attention opcional (solo interpretabilidad) ─────────────────
        self.cross_attn = _CLSCrossAttention(dim=dim_model, heads=ca_heads)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        logits_ws: torch.Tensor,  # (N, num_classes) — ignorado (interfaz BaseMIL)
        feats_ws:  torch.Tensor,  # (N, embed_dim)   — embeddings del encoder
    ):
        """
        Agrega los embeddings de parche de UNA WSI con Transformer.

        Args:
            logits_ws : ignorado (requerido por la interfaz BaseMIL).
            feats_ws  : Tensor (N, embed_dim).

        Returns:
            wsi_logit : Tensor (num_classes,).
            weights   : Tensor (N,) con pesos de atención por parche en el mismo orden que feats_ws (sin padding)
        """
        if feats_ws.numel() == 0:
            return torch.zeros(self.num_classes, device=feats_ws.device), None

        N_orig = feats_ws.shape[0]

        # 1. Proyección de entrada: embed_dim → dim_model
        h = self.input_proj(feats_ws).unsqueeze(0)  # (1, N, dim_model)

        # 2. Pad cuadrado (TransMIL necesita secuencia de longitud N² para PPEG)
        N = h.shape[1]
        _H = _W = math.ceil(math.sqrt(N))
        pad_len = _H * _W - N
        if pad_len > 0:
            h = torch.cat([h, h[:, :pad_len, :]], dim=1) # opción original: repite los primeros tokens para rellenar
            
            # pad_tensor = torch.zeros((1, pad_len, h.size(-1)),device=h.device,dtype=h.dtype)
            # h = torch.cat([h, pad_tensor], dim=1) # opción alternativa: rellena con ceros
    

        # 3. CLS token
        cls = self.cls_token  # (1, 1, dim_model)
        h = torch.cat([cls, h], dim=1)      # (1, N_pad+1, dim_model)

        # 4. TransLayer 1 — correlación morfológica entre instancias
        h = self.layer1(h)                  # (1, N_pad+1, dim_model)

        # 5. PPEG — positional encoding espacial condicional
        h = self.pos_layer(h, _H, _W)      # (1, N_pad+1, dim_model)

        # 6. TransLayer 2 — correlación profunda
        h = self.layer2(h)                # (1, N_pad+1, dim_model)

        # 7. LayerNorm y extracción del CLS token
        h_norm        = self.norm(h)
        cls_token_out  = h_norm[:, :1, :]  # (1, 1, dim_model)
        patch_tokens   = h_norm[:, 1:, :]  # (1, N_pad, dim_model)

        # 8. Clasificador de WSI — SOBRE EL CLS DIRECTO (fiel al paper)
        #    MLP(LN(CLS)) exactamente como en el Algoritmo 2 de Shao et al.
        wsi_logit = self.bag_classifier(cls_token_out.squeeze(0)).squeeze(0)

        # 9. Pesos de interpretabilidad via cross-attention
        #    Ejecutado bajo no_grad: NUNCA afecta gradientes ni el entrenamiento.
        weights = None
        with torch.no_grad():
            weights_padded = self.cross_attn(cls_token_out, patch_tokens)
            # Recortar padding y renormalizar
            weights = weights_padded[:N_orig]
            weights = weights / (weights.sum() + 1e-8)

        return wsi_logit, weights