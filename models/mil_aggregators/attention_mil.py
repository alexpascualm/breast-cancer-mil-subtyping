"""
models/mil_aggregators/attention_mil.py
---------------------------------------
Dos variantes de Attention-Based MIL (Ilse et al., ICML 2018):

    AttentionMIL_InstanceLevel  → clave CLI: "abMIL-instance-level"
    ─────────────────────────────────────────────────────────────────
    Enfoque INSTANCE-BASED.

    Pipeline:
        patch → encoder → h_k  (embedding)
        h_k   → patch_classifier → logit_k   (clasificador de parche, en WSIClassifier)
        {logit_k} x softmax_attention(h_k) → logit_WSI

    La red de atención calcula pesos sobre los embeddings (feats_ws), y
    esos pesos ponderan los logits de parche ya calculados (logits_ws).
    El patch_classifier de WSIClassifier SÍ se usa. Este módulo NO tiene
    clasificador propio.

    AttentionMIL_BagLevel  → clave CLI: "abMIL-bag-level"
    ────────────────────────────────────────────────────────
    Enfoque EMBEDDING-BASED. Es el método
    principal propuesto por Ilse et al. y el que obtiene mejores resultados.

    Pipeline:
        patch → encoder → h_k  (embedding)
        a_k   = softmax( w^T [ tanh(V h_k^T) ⊙ sigm(U h_k^T) ] )
        z     = Σ_k  a_k · h_k          (bag embedding, R^{embed_dim})
        z     → bag_classifier           (clasificador bag-level propio)

    El patch_classifier de WSIClassifier NO interviene: este módulo
    lleva su propio nn.Linear (bag_classifier). Por eso está registrado
    como BAG_BASED_STRATEGY. logits_ws se ignora.

    Referencia:
        Ilse M., Tomczak J.M., Welling M.
        "Attention-based Deep Multiple Instance Learning." ICML 2018.
        https://arxiv.org/abs/1802.04712
"""

import torch
import torch.nn as nn

from models.mil_aggregators.base_mil import BaseMIL


# ─────────────────────────────────────────────────────────────────────────────
# Red de atención gated (compartida por ambas variantes)
# Ec. (9) del paper: a_k ∝ exp{ w^T [ tanh(V h_k^T) ⊙ sigm(U h_k^T) ] }
# ─────────────────────────────────────────────────────────────────────────────

class _GatedAttentionNet(nn.Module):
    """
    Red de atención gated de dos ramas (Ilse et al. Ec. 9).

    Dado un conjunto de embeddings H ∈ R^{N x embed_dim},
    devuelve scores escalares no normalizados ∈ R^N.
    La normalización softmax se realiza en el módulo llamante.

    Args:
        embed_dim (int)  : dimensión de los embeddings de entrada (M del paper).
        attn_dim  (int)  : dimensión oculta L del paper. None → embed_dim // 2.
        dropout   (float): dropout aplicado tras la multiplicación de ramas. Default 0.2.
    """

    def __init__(self, embed_dim: int, attn_dim: int | None = None, dropout: float = 0.2):
        super().__init__()
        L = attn_dim if attn_dim is not None else embed_dim // 2

        # Rama tanh:    V ∈ R^{L × M}
        self.branch_tanh    = nn.Sequential(nn.Linear(embed_dim, L), nn.Tanh())
        # Rama sigmoid: U ∈ R^{L × M}  (gate)
        self.branch_sigmoid = nn.Sequential(nn.Linear(embed_dim, L), nn.Sigmoid())
        # Dropout sobre el gate conjunto
        self.drop           = nn.Dropout(dropout)
        # Proyección:   w ∈ R^{L × 1}
        self.out            = nn.Linear(L, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor (N, embed_dim)

        Returns:
            scores : Tensor (N,)  — scores no normalizados
        """
        gate = self.branch_tanh(x) * self.branch_sigmoid(x)   # (N, L)
        return self.out(self.drop(gate)).squeeze(-1)            # (N,)

# ─────────────────────────────────────────────────────────────────────────────
# Variante 1 — Instance-level
# ─────────────────────────────────────────────────────────────────────────────

class AttentionMIL_InstanceLevel(BaseMIL):
    """
    ABMIL instance-level: pondera logits de parche con pesos de atención
    calculados sobre los embeddings.

    Registrado como estrategia INSTANCE-BASED en MIL_REGISTRY.
    El patch_classifier de WSIClassifier SÍ se usa.

    Args:
        num_classes (int) : número de clases.
        embed_dim   (int) : dimensión de los embeddings del encoder (entrada).
        dim_model   (int) : dimensión de entrada del modelo de atención (default 512).
        attn_dim    (int) : dimensión oculta L de la red de atención. None → embed_dim // 2.
        dropout     (float) : dropout en la red de atención. Default 0.2.
    """

    def __init__(self, num_classes: int, embed_dim: int, dim_model: int = 512, attn_dim: int | None = None, dropout: float = 0.2):
        super().__init__(num_classes=num_classes, embed_dim=embed_dim)

        # Proyección de entrada: embed_dim → dim_model
        if dim_model != embed_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(embed_dim, dim_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_proj = nn.Identity()

        self.attention = _GatedAttentionNet(dim_model, attn_dim=attn_dim, dropout=dropout)

    def forward(
        self,
        logits_ws: torch.Tensor,   # (N, num_classes) — logits de parche
        feats_ws:  torch.Tensor,   # (N, embed_dim)      — embeddings de parche
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            wsi_logit : Tensor (num_classes,)
            weights   : Tensor (N,)
        """
        if feats_ws.numel() == 0:
            return (
                torch.zeros(self.num_classes, device=logits_ws.device),
                torch.zeros(0,               device=logits_ws.device),
            )

        h       = self.input_proj(feats_ws)                        # (N, dim_model)
        scores  = self.attention(h)                                # (N,)
        weights = torch.softmax(scores, dim=0)                     # (N,)

        # Agrega logits de parche ponderados por atención
        wsi_logit = (logits_ws * weights.unsqueeze(1)).sum(dim=0)  # (num_classes,)

        return wsi_logit, weights


# ─────────────────────────────────────────────────────────────────────────────
# Variante 2 — Bag-level (método principal del paper)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionMIL_BagLevel(BaseMIL):
    """
    ABMIL bag-level (embedding-based): agrega EMBEDDINGS con pesos de
    atención y clasifica el embedding resultante con un clasificador propio.

    Este es el método principal de Ilse et al. 2018.

    Flujo:
        h       = input_proj(feats_ws)          # (N, embed_dim)
        scores  = _GatedAttentionNet(h)          # (N,)
        weights = softmax(scores)                # (N,)
        z       = Σ_k weights_k · h_k           # (embed_dim,)
        logit   = bag_classifier(z)              # (num_classes,)

    Registrado como BAG_BASED_STRATEGY: el patch_classifier de
    WSIClassifier NO interviene. logits_ws se ignora.

    Args:
        num_classes (int) : número de clases.
        embed_dim   (int) : dimensión interna del modelo.
        dim_model   (int) : dimensión de entrada del modelo de atención (default 512).
        attn_dim    (int) : dimensión oculta L de la red de atención. None → embed_dim // 2.
        dropout     (float) : dropout en la red de atención. Default 0.2.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim:   int,
        dim_model:   int = 512,
        attn_dim:    int | None = None,
        dropout:     float = 0.2,
    ):
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

        self.attention      = _GatedAttentionNet(dim_model, attn_dim=attn_dim, dropout=dropout)
        # Clasificador bag-level: g_φ del paper
        self.bag_classifier = nn.Linear(dim_model, num_classes)
    def forward(
        self,
        logits_ws: torch.Tensor,   # (N, num_classes) — ignorado en bag-level
        feats_ws:  torch.Tensor,   # (N, embed_dim)      — entrada real
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            wsi_logit : Tensor (num_classes,)
            weights   : Tensor (N,)
        """
        if feats_ws.numel() == 0:
            return (
                torch.zeros(self.num_classes, device=feats_ws.device),
                torch.zeros(0,               device=feats_ws.device),
            )

        h       = self.input_proj(feats_ws)               # (N, dim_model)

        # Pesos de atención sobre embeddings proyectados (Ec. 8/9)
        scores  = self.attention(h)                        # (N,)
        weights = torch.softmax(scores, dim=0)             # (N,)

        # Agregación de embeddings (Ec. 7): z = Σ_k a_k · h_k
        z = (h * weights.unsqueeze(1)).sum(dim=0)          # (dim_model,)

        # Clasificación bag-level
        wsi_logit = self.bag_classifier(z)                 # (num_classes,)

        return wsi_logit, weights
