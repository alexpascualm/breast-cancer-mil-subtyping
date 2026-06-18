"""
compute_srg.py
--------------
Calcula el Symmetric Relevance Gain (SRG) de una WSI siguiendo exactamente
la metodología del paper:

    "Beyond Attention Heatmaps: How to Get Better Explanations for
     Multiple Instance Learning Models in Histopathology"
     Jamshidi Idaji et al., 2026.  §3.3.1, Ecuaciones 1-3.

La función `compute_srg` es agnóstica al método de explicación: acepta
cualquier vector de pesos por parche (atención, LRP, Single, IG, gradientes,
aleatorio...) y cualquier modelo que exponga `forward_from_embeddings`.

Protocolo del paper:
    1. Ordenar los N parches por score de explicación.
    2. Dividirlos en 100 mini-bags (percentiles 1 %..100 %).
    3. Eliminar los mini-bags progresivamente en dos órdenes:
         · Descending  (más relevantes primero → MORF)
         · Ascending   (menos relevantes primero → MORL)
    4. En cada paso re-inferir con el modelo y registrar la predicción.
    5. AUPC = media de las 101 predicciones (paso 0 incluido).
    6. SRG  = AUPC(ascending) - AUPC(descending).
       Un SRG mayor indica un heatmap más fiel al modelo.

Nota sobre el paso m=100 (bolsa vacía):
    El paper pasa un array de ceros al modelo cuando la bolsa queda vacía.
    Aquí se replica ese comportamiento con un único embedding de ceros.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Interfaz pública
# ---------------------------------------------------------------------------

def compute_srg(
    model: nn.Module,
    embeddings: torch.Tensor,
    patch_scores: np.ndarray | torch.Tensor,
    target_class: int = 0,
    n_bins: int = 100,
    device: torch.device | str | None = None,
) -> float:
    """
    Calcula el SRG de una WSI.

    Parameters
    ----------
    model : nn.Module
        Modelo con método `forward_from_embeddings(list[Tensor]) -> (wsi_logits, ...)`.
        Compatible con WSIClassifier del proyecto.
    embeddings : Tensor de forma (N, embed_dim)
        Embeddings de los N parches de la WSI. Ya deben estar en el dispositivo
        correcto o se moverán automáticamente si se proporciona `device`.
    patch_scores : array-like de longitud N
        Puntuaciones de relevancia por parche generadas por cualquier método
        (atención, LRP, Single, IG, gradientes, aleatorio…).
        No necesitan estar normalizadas.
    target_class : int
        Índice de la clase cuyo logit/probabilidad se monitoriza durante el
        patch-flipping. Por defecto 0.
    n_bins : int
        Número de percentiles en los que se divide la bolsa.
        El paper usa 100 (valor por defecto). Cambiarlo rompe la comparabilidad
        con los resultados del paper.
    device : torch.device | str | None
        Dispositivo donde ejecutar la inferencia. Si es None se usa el
        dispositivo de `embeddings` (o CPU si es una constante).

    Returns
    -------
    float
        SRG = AUPC(ascending) - AUPC(descending).
        Valor más alto → heatmap más fiel.
    """
    # ── Preparación ──────────────────────────────────────────────────────────
    patch_scores = np.asarray(patch_scores, dtype=np.float32)
    N = len(patch_scores)
    assert embeddings.shape[0] == N, (
        f"embeddings.shape[0]={embeddings.shape[0]} != len(patch_scores)={N}"
    )

    if device is None:
        device = embeddings.device if hasattr(embeddings, "device") else torch.device("cpu")
    device = torch.device(device)
    embeddings = embeddings.to(device)

    # ── Partición en 100 percentiles (Eq. 1 del paper) ──────────────────────
    # π ordena de menor a mayor score. El bin i contiene los parches cuyos
    # scores caen entre el percentil (100-i) y (100-i+1).
    # En la práctica es más directo trabajar con los índices ordenados y
    # asignar cada parche a uno de los 100 bins equitativamente.
    sorted_asc_idx = np.argsort(patch_scores)          # índices: score ↑
    bins: list[np.ndarray] = _split_into_bins(sorted_asc_idx, n_bins)
    # bins[0] = parches con scores más BAJOS (percentil 1 en orden ascendente)
    # bins[-1] = parches con scores más ALTOS (percentil 100)

    # ── Predicción con bolsa completa (m=0) ──────────────────────────────────
    pred_full = _infer(model, embeddings, device, target_class)

    # ── Predicción con bolsa vacía (m=100, array de ceros) ───────────────────
    zero_emb = torch.zeros(1, embeddings.shape[1], device=device)
    pred_zero = _infer(model, zero_emb, device, target_class)

    # ── Curvas de perturbación ────────────────────────────────────────────────
    aupc_desc = _perturbation_curve(
        model, embeddings, bins,
        order="descending",          # más relevante primero (MORF)
        pred_full=pred_full,
        pred_zero=pred_zero,
        target_class=target_class,
        device=device,
    )

    aupc_asc = _perturbation_curve(
        model, embeddings, bins,
        order="ascending",           # menos relevante primero (MORL)
        pred_full=pred_full,
        pred_zero=pred_zero,
        target_class=target_class,
        device=device,
    )

    # ── SRG (Eq. 3) ──────────────────────────────────────────────────────────
    srg = aupc_asc - aupc_desc
    return float(srg)


# ---------------------------------------------------------------------------
# Funciones auxiliares internas
# ---------------------------------------------------------------------------

def _split_into_bins(sorted_idx: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """
    Divide los índices ordenados en n_bins grupos de tamaño lo más uniforme
    posible, replicando la lógica de percentiles del paper (strategy='1%-of-all').

    El paper agrupa los parches en los percentiles 1%..100% de sus scores;
    aquí se hace de forma idéntica usando np.array_split, que distribuye
    los parches sobrantes en los primeros bins.
    """
    return np.array_split(sorted_idx, n_bins)


def _infer(
    model: nn.Module,
    embeddings: torch.Tensor,
    device: torch.device,
    target_class: int,
) -> float:
    """
    Inferencia de una bolsa de embeddings. Devuelve el logit (o la
    probabilidad si el modelo ya aplica softmax) de `target_class`.
    Usa el logit bruto, igual que el paper en clasificación.
    """
    model.eval()
    with torch.no_grad():
        wsi_logits, *_ = model.forward_from_embeddings([embeddings.to(device)])
    # wsi_logits: (1, num_classes)  →  escalar para target_class
    return wsi_logits[0, target_class].item()


def _perturbation_curve(
    model: nn.Module,
    embeddings: torch.Tensor,
    bins: list[np.ndarray],
    order: str,                   # "descending" o "ascending"
    pred_full: float,
    pred_zero: float,
    target_class: int,
    device: torch.device,
) -> float:
    """
    Construye una curva de perturbación y calcula su AUPC (Eq. 2).

    En order="descending" (MORF):
        Bin[-1] (más relevante) se elimina primero.
        Partimos de la bolsa completa y vamos eliminando.

    En order="ascending" (MORL):
        Bin[0] (menos relevante) se elimina primero.
        Partimos de la bolsa completa y vamos eliminando.

    La curva tiene n_bins+1 puntos: f(X_0) ... f(X_100).
    f(X_0) = predicción con bolsa completa.
    f(X_100) = predicción con bolsa vacía (zeros).
    """
    n_bins = len(bins)

    # Orden de eliminación de bins
    if order == "descending":
        # eliminamos primero los de mayor relevancia (bins al final del sorted_asc)
        elimination_order = list(range(n_bins - 1, -1, -1))   # [99, 98, ..., 0]
    else:
        # eliminamos primero los de menor relevancia (bins al inicio del sorted_asc)
        elimination_order = list(range(n_bins))                # [0, 1, ..., 99]

    # Conjunto de índices que quedan en la bolsa (empezamos con todos)
    remaining = set(range(embeddings.shape[0]))

    predictions = [pred_full]   # f(X_0): bolsa completa

    for bin_pos in elimination_order:
        # Eliminar el bin correspondiente
        to_remove = set(bins[bin_pos].tolist())
        remaining -= to_remove

        if len(remaining) == 0:
            # Bolsa vacía → zeros (replicando comportamiento del paper)
            predictions.append(pred_zero)
        else:
            idx = torch.tensor(sorted(remaining), dtype=torch.long, device=device)
            bag = embeddings[idx]                  # (|remaining|, embed_dim)
            predictions.append(_infer(model, bag, device, target_class))

    # AUPC = media aritmética de las 101 predicciones (Eq. 2)
    aupc = float(np.mean(predictions))
    return aupc