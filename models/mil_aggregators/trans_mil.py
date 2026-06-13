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
#  Módulo TransMIL adaptado a BaseMIL — fiel al paper, con interpretabilidad
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
        8. Clasificador MLP sobre el CLS token.

    Interpretabilidad (sin parámetros entrenables):
        Los pesos por parche se obtienen como la similitud coseno entre el
        CLS token procesado y cada patch token procesado, ambos extraídos
        tras el LayerNorm final.

        Justificación: el CLS token es la representación de la bolsa entera
        que el clasificador usa directamente. Su similitud con cada patch token
        —en el espacio aprendido por el modelo— mide cuánto se "parece" cada
        parche a esa representación agregada, lo que es una proxy directa de
        contribución. No se introduce ningún parámetro extra: los únicos pesos
        que influyen son los del modelo principal, entrenados end-to-end con
        la loss de clasificación.

    Args:
        num_classes (int)  : número de clases de salida.
        embed_dim   (int)  : dimensión de los embeddings del encoder (entrada).
        dim_model   (int)  : dimensión interna del Transformer (default 512).
    """

    def __init__(self, num_classes: int, embed_dim: int, dim_model: int = 512):
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

        # ── Clasificador de WSI (sobre CLS token, igual que el paper) ─────────
        self.bag_classifier = nn.Linear(dim_model, num_classes)

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_attention_weights(
        cls_token:    torch.Tensor,   # (1, 1,     D)
        patch_tokens: torch.Tensor,   # (1, N_pad, D)
    ) -> torch.Tensor:
        """
        Calcula pesos de atención como similitud coseno entre el CLS token
        y cada patch token, seguida de un softmax para normalizar a [0, 1].

        No contiene ningún parámetro entrenable: la única información que
        usa son las representaciones aprendidas por el modelo principal.

        Por qué similitud coseno y no producto escalar (dot-product):
            - El producto escalar mezcla magnitud y ángulo, por lo que parches
              con norma alta parecerían más "importantes" independientemente de
              su orientación respecto al CLS. La similitud coseno aísla la
              componente direccional, que es lo semánticamente relevante.
            - En espacios de alta dimensión los vectores tienden a tener normas
              similares, pero normalizar explícitamente elimina cualquier sesgo.

        Por qué softmax sobre los scores coseno y no los scores crudos:
            - Normalizar con softmax convierte los scores en una distribución
              de probabilidad sumando 1, directamente interpretable como
              "fracción de atención" por parche.
            - Temperatura implícita: los scores coseno están acotados en [-1,1],
              lo que da un softmax con temperatura natural, sin colapsar en
              distribuciones demasiado peaked ni demasiado uniformes.

        Args:
            cls_token    : (1, 1, D)     — CLS tras LN final.
            patch_tokens : (1, N_pad, D) — patches tras LN final.

        Returns:
            weights : (N_pad,) — pesos en [0, 1] sumando 1.
        """
        # Normalizar L2 para obtener similitud coseno pura
        cls_norm   = F.normalize(cls_token,    dim=-1)   # (1, 1,     D)
        patch_norm = F.normalize(patch_tokens, dim=-1)   # (1, N_pad, D)

        # Similitud coseno: (1, 1, D) x (1, D, N_pad) → (1, 1, N_pad)
        cos_sim = torch.matmul(cls_norm, patch_norm.transpose(-2, -1))

        # Softmax sobre la dimensión de parches → distribución de atención
        weights = cos_sim.softmax(dim=-1)   # (1, 1, N_pad)

        return weights.squeeze(0).squeeze(0)  # (N_pad,)

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
            weights   : Tensor (N,) — pesos de atención por parche en el mismo
                        orden que feats_ws (sin padding), sumando 1.
                        Interpretación: el parche i contribuye una fracción
                        weights[i] a la representación final de la bolsa.
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
        h = self.layer2(h)                 # (1, N_pad+1, dim_model)

        # 7. LayerNorm final
        #    Aquí es donde se extraen las representaciones para interpretabilidad:
        #    tanto el CLS como los patches han pasado por todo el modelo y están
        #    en el mismo espacio normalizado que usa el clasificador.
        h_norm       = self.norm(h)
        cls_token_out = h_norm[:, :1, :]   # (1, 1,     dim_model)
        patch_tokens  = h_norm[:, 1:, :]   # (1, N_pad, dim_model)

        # 8. Clasificador de WSI — MLP(LN(CLS)), fiel al paper (Algorithm 2, paso 5)
        wsi_logit = self.bag_classifier(cls_token_out.squeeze(0)).squeeze(0)

        # 9. Pesos de interpretabilidad — similitud coseno sin parámetros entrenables
        #
        #    Se calcula bajo no_grad porque:
        #      a) No aporta información nueva al grafo de cómputo (los tensores
        #         h_norm ya están en el grafo si hace falta backprop).
        #      b) Evita almacenar activaciones intermedias innecesarias en memoria
        #         durante el entrenamiento.
        #
        #    El resultado es determinista dado el estado del modelo: para un modelo
        #    entrenado, siempre producirá los mismos pesos para la misma WSI.
        with torch.no_grad():
            weights_padded = self._cosine_attention_weights(cls_token_out, patch_tokens)
            # Recortar tokens de padding — solo nos interesan los N_orig originales
            weights = weights_padded[:N_orig]
            # Renormalizar tras el recorte para que sigan sumando 1
            weights = weights / (weights.sum() + 1e-8)

        return wsi_logit, weights