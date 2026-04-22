"""
utils/train_utils.py
--------------------
Utilidades de entrenamiento: semillas, get_model, splits, run_epoch.
"""

import torch
import torch.nn as nn
import os
import pandas as pd
import random
import numpy as np
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

from models.patch_encoders import ENCODER_REGISTRY
from models.wsi_classifier import WSIClassifier
from utils.metrics_utils import compute_breast_metrics
from sklearn.metrics import confusion_matrix


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN FIJA POR ENCODER
# Modifica aquí si quieres cambiar el modelo concreto de cada familia.
# ──────────────────────────────────────────────────────────────────────────────

ENCODER_CONFIGS = {
    "dino": {
        "dino_model_name":      "vit_base_patch16_224.dino",
        "patch_size":           224,
        "embed_dim":            768,    # ViT-B/16
    },
    "uni": {
        "uni_model_name": "MahmoodLab/UNI",
        "patch_size":     224,
        "embed_dim":      1024,         # ViT-L/16
    },
    # "conch": {
    #     "conch_model_name": "MahmoodLab/conchv1_5",
    #     "patch_size":       448,
    #     "embed_dim":        768,
    # },
    # "keep": {
    #     "keep_model_name": "Astaxanthin/KEEP",
    #     "patch_size":      224,
    #     "embed_dim":       1024,   
    # },
}


# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────

def set_multiple_seed(seed: int = 42, deterministic: bool = True):
    """Fija semillas para reproducibilidad en Python, NumPy y PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True

    print(f"[INFO] Semilla fijada en {seed} (deterministic={deterministic})")


# ──────────────────────────────────────────────────────────────────────────────
# GET MODEL
# ──────────────────────────────────────────────────────────────────────────────

def get_model(
    patch_encoder_type: str = "dino",
    mil_aggregation_mode: str = "attentionMIL",
    unfreeze_backbone: bool = False,
    num_classes: int = 2,
    chunk_size: int = 512,
    skip_encoder: bool = False,
) -> WSIClassifier:
    """
    Construye y devuelve un WSIClassifier listo para entrenar.

    Dos modos de operación:

    skip_encoder=False (modo normal):
        1. Instancia el encoder de parches desde ENCODER_REGISTRY.
        2. Instancia WSIClassifier combinando encoder + MIL + clasificador.
        El embed_dim se lee de ENCODER_CONFIGS y se extrae antes de pasarlo al encoder para evitar kwargs inesperados.

    skip_encoder=True (modo embeddings precomputados):
        Solo instancia el MIL + clasificador, sin cargar ningún backbone.
        El embed_dim se lee directamente de ENCODER_CONFIGS.
        Requiere usar WSIClassifier.forward_from_embeddings() en lugar de forward().

    Args:
        patch_encoder_type (str) : 'dino' | 'uni' | 'conch' | 'keep'.
        mil_aggregation_mode (str): 'max' | 'mean' | 'abMIL-instance-level' | 'abMIL-bag-level' | 'transMIL'.
        unfreeze_backbone (bool) : descongelar últimos bloques del backbone (ignorado si skip_encoder=True).
        num_classes (int)        : número de clases de salida.
        chunk_size (int)         : mini-batch interno de parches (ignorado si skip_encoder=True).
        skip_encoder (bool)      : si True, no instancia el encoder — usar con embeddings precomputados.

    Returns:
        WSIClassifier
    """
    key = patch_encoder_type.lower()
    if key not in ENCODER_REGISTRY:
        raise ValueError(
            f"Encoder desconocido: '{patch_encoder_type}'. "
            f"Opciones: {list(ENCODER_REGISTRY.keys())}"
        )

    if skip_encoder:
        # Solo MIL + clasificador, sin backbone. Útil cuando los embeddings ya están precomputados en disco y no se quiere cargar el ViT en memoria.
        model = WSIClassifier(
            encoder=None,
            mil_aggregation_mode=mil_aggregation_mode,
            num_classes=num_classes,
            embed_dim=ENCODER_CONFIGS[key]["embed_dim"],
        )
    else:
        encoder_kwargs = {k: v for k, v in ENCODER_CONFIGS[key].items() if k != "embed_dim"} # embed_dim se descarta porque el encoder lo asigna internamente tras build_backbone() y no se le pasa como argumento directo.

        encoder = ENCODER_REGISTRY[key](
            chunk_size=chunk_size,
            unfreeze_backbone=unfreeze_backbone,
            **encoder_kwargs,
        )

        model = WSIClassifier(
            encoder=encoder,
            mil_aggregation_mode=mil_aggregation_mode,
            num_classes=num_classes,
            chunk_size=chunk_size,
        )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels, num_classes):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de clase.

    Args:
        labels: array-like con etiquetas enteras.
        num_classes (int): número total de clases.

    Returns:
        Tensor (num_classes,) con los pesos normalizados.
    """
    labels  = torch.tensor(labels, dtype=torch.long)
    counts  = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# SPLITS
# ──────────────────────────────────────────────────────────────────────────────

def make_splits(df, label_col, seed):
    """
    Divide el DataFrame en índices de train / val / test respetando grupos
    de paciente y estratificando por etiqueta (StratifiedGroupKFold).
    El DataFrame recibido ya debe estar filtrado (sin etiquetas -1) y tener
    la columna 'patient_id' como string.
    Proporciones: 70% train · 15% val · 15% test.
    Imprime las proporciones de clase del DataFrame completo y de cada split.
    Args:
        df        : pd.DataFrame filtrado para la tarea.
        label_col : str  columna de etiquetas (== task name).
        seed      : int  semilla para reproducibilidad.
    Returns:
        train_idx, val_idx, test_idx  — arrays de índices posicionales sobre df.
    """
    def get_proportions_str(name, indices):
        subset = df.iloc[indices] if not isinstance(indices, pd.Index) else df.loc[indices]
        counts = subset[label_col].value_counts().sort_index()
        props  = subset[label_col].value_counts(normalize=True).sort_index()
        lines  = [f"\n  [{name}]  n={len(subset)}"]
        for cls in counts.index:
            lines.append(f"    clase {cls}: {counts[cls]:>4d}  ({props[cls]:.1%})")
        return "\n".join(lines)

    print("=" * 45)
    print(f"SPLITS — tarea: '{label_col}'  seed={seed}")
    print("=" * 45)

    report_lines = ["=" * 45, f"SPLITS — tarea: '{label_col}'  seed={seed}", "=" * 45]

    total_str = get_proportions_str("TOTAL", df.index)
    print(total_str)
    report_lines.append(total_str)

    # Test split
    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(
        sgkf_test.split(df.index, df[label_col], groups=df["patient_id"])
    )

    # Val split
    sub = df.iloc[train_val_idx]
    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    rel_train_idx, rel_val_idx = next(
        sgkf_val.split(sub.index, sub[label_col], groups=sub["patient_id"])
    )

    train_idx = train_val_idx[rel_train_idx]
    val_idx   = train_val_idx[rel_val_idx]

    for name, idx in [("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)]:
        split_str = get_proportions_str(name, idx)
        print(split_str)
        report_lines.append(split_str)

    print("=" * 45)
    report_lines.append("=" * 45)

    report_str = "\n".join(report_lines)

    return train_idx, val_idx, test_idx, report_str


# ──────────────────────────────────────────────────────────────────────────────
# EPOCH RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model,
    dataloader,
    device,
    accelerator,
    criterion,
    optimizer=None,
    task: str = "molecular_subtype",
    mode: str = "train",
    inference_heatmap: bool = False,
    precomputed_embeddings: bool = False
):
    """
    Ejecuta una época completa (train / val / test).

    Args:
        model: nn.Module.
        dataloader: DataLoader con collate_fn de breast_wsi_dataset.
        device: torch.device.
        accelerator: Accelerator de HuggingFace.
        criterion: función de pérdida (CrossEntropyLoss).
        optimizer: optimizador (solo en mode='train').
        task (str): tarea activa — determina cómo interpretar etiquetas y métricas.
        mode (str): 'train' | 'val' | 'test'.
        inference_heatmap (bool): si True, acumula info para generar heatmaps.
        precomputed_embeddings (bool): si True, el dataloader ya devuelve embeddings en vez de imágenes y se usa model.forward_from_embeddings().

    Returns:
        metrics (dict): métricas de la época.
        conf_mat_df (pd.DataFrame): matriz de confusión.
        epoch_loss (float): pérdida media.
        heatmap_df (pd.DataFrame | None): DataFrame con info de heatmap o None.
    """
    assert mode in ("train", "val", "test")

    if mode == "train":
        model.train()
    else:
        model.eval()

    torch.set_grad_enabled(mode == "train")

    num_classes = 4 if task == "molecular_subtype" else 2

    all_y_true, all_y_pred, all_y_prob = [], [], []
    heatmap_list = []

    total_loss_sum = 0.0
    total_seen = 0

    for batch_info in tqdm(
        dataloader,
        desc=mode.upper(),
        disable=not accelerator.is_main_process,
        file=sys.stdout,
    ):
        if inference_heatmap:
            inputs, labels, wsi_ids, coords = batch_info
        else:
            inputs, labels, wsi_ids, _ = batch_info

        labels = labels.to(device)
        
        if precomputed_embeddings:
            # inputs es lista de tensores de embeddings — se mueven a device y se procesan con model.forward_from_embeddings()
            inputs_device = [emb.to(device, non_blocking=True) for emb in inputs]
            wsi_logits, patch_logits_list, weights_list = model.forward_from_embeddings(inputs_device)
        else:
            # inputs es lista de tensores de imágenes — se procesan con model.forward() y se mueven a device dentro del encoder.forward
            wsi_logits, patch_logits_list, weights_list = model(inputs)

        loss = criterion(wsi_logits, labels)

        if mode == "train":
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        probs = torch.softmax(wsi_logits.detach(), dim=1)
        preds = probs.argmax(dim=1)

        # Loss
        batch_size = labels.size(0)

        loss_sum = loss.detach() * batch_size

        loss_sum_reduced = accelerator.reduce(loss_sum, reduction="sum")
        count_reduced    = accelerator.reduce(torch.tensor(batch_size, device=device), reduction="sum")

        total_loss_sum += loss_sum_reduced.item()
        total_seen     += count_reduced.item()

        # Gather métricas
        gat_labels, gat_preds, gat_probs = accelerator.gather_for_metrics(
            (labels, preds, probs)
        )

        if accelerator.is_main_process:
            all_y_true.append(gat_labels.cpu())
            all_y_pred.append(gat_preds.cpu())
            all_y_prob.append(gat_probs.cpu())

        # Heatmap: SOLO acumulación local (sin gather aquí)
        if inference_heatmap:
            for i in range(len(inputs)):
                # patch_logits_list[i] puede ser None en estrategias bag-based (transMIL)
                patch_scores = (
                    torch.argmax(patch_logits_list[i], dim=1).cpu().numpy()
                    if patch_logits_list[i] is not None
                    else None
                )
                patch_weights = (
                    weights_list[i].detach().cpu().numpy()
                    if weights_list[i] is not None
                    else None
                )
                heatmap_list.append({
                    "wsi_id":       wsi_ids[i],
                    "pred_label":   preds[i].item(),
                    "true_label":   labels[i].item(),
                    "coords":       coords[i],
                    "patch_scores": patch_scores,
                    "weights":      patch_weights,
                })

    epoch_loss = total_loss_sum / total_seen

    metrics = {}
    conf_mat_df = pd.DataFrame()
    heatmap_df = None

    if accelerator.is_main_process:
        y_true = torch.cat(all_y_true).numpy()
        y_pred = torch.cat(all_y_pred).numpy()
        y_prob = torch.cat(all_y_prob).numpy()

        metrics = compute_breast_metrics(task, y_true, y_pred, y_prob)

        conf_mat = confusion_matrix(
            y_true, y_pred, labels=np.arange(num_classes)
        )

        conf_mat_df = pd.DataFrame(
            conf_mat,
            index=[f"true_{i}" for i in range(conf_mat.shape[0])],
            columns=[f"pred_{i}" for i in range(conf_mat.shape[1])],
        )

    # Gather heatmaps al final
    if inference_heatmap:
        gathered_heatmaps = accelerator.gather_object(heatmap_list)

        if accelerator.is_main_process:
            # flatten lista de listas
            flat_heatmaps = [item for sublist in gathered_heatmaps for item in sublist]
            flat_heatmaps = sorted(flat_heatmaps, key=lambda x: x["wsi_id"])
            heatmap_df = pd.DataFrame(flat_heatmaps)

    return metrics, conf_mat_df, epoch_loss, heatmap_df
