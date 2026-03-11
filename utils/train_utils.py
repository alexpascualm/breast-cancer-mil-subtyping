import torch
import torch.nn as nn
import os
import pandas as pd
import random
import numpy as np
import sys

from tqdm import tqdm

from models.patch_encoders.VIT_model import PatchClassifierVIT
from utils.metrics_utils import compute_breast_metrics

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


def get_model(patch_encoder_type, mil_aggregation_mode, unfreeze_backbone, num_classes):
    """
    Instancia el encoder de parches seleccionado.

    Args:
        patch_encoder_type (str): 'VIT' (único soportado actualmente).
        mil_aggregation_mode (str): 'max' | 'mean' | 'attentionMIL'.
        unfreeze_backbone (bool): descongelar backbone del encoder.
        num_classes (int): número de clases de salida (del dataset).

    Returns:
        nn.Module
    """
    if patch_encoder_type == "VIT":
        return PatchClassifierVIT(
            num_classes=num_classes,
            mil_aggregation_mode=mil_aggregation_mode,
            unfreeze_backbone=unfreeze_backbone,
        )
    else:
        raise ValueError(f"Encoder desconocido: '{patch_encoder_type}'. Opciones: 'VIT'")


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

    Returns:
        metrics (dict): métricas de la época.
        epoch_loss (float): pérdida media.
        heatmap_df (pd.DataFrame | None): DataFrame con info de heatmap o None.
    """
    assert mode in ("train", "val", "test"), "mode debe ser 'train', 'val' o 'test'."

    model.train() if mode == "train" else model.eval()
    torch.set_grad_enabled(mode == "train")

    all_y_true, all_y_pred, all_y_prob = [], [], []
    heatmap_list = []
    total_loss_sum = 0.0
    total_seen = torch.tensor(0, device=device)

    for batch_info in tqdm(dataloader, desc=mode.upper(),
                           disable=not accelerator.is_local_main_process,
                           file=sys.stdout):

        if inference_heatmap:
            inputs, labels, wsi_ids, coords = batch_info
        else:
            inputs, labels, wsi_ids, _ = batch_info

        labels = labels.to(device)   # (B,) LongTensor

        # ── Forward ───────────────────────────────────────────────────────────
        wsi_logits, patch_probs_list, weights_list = model(inputs)
        # wsi_logits: (B, num_classes)

        # ── Loss ──────────────────────────────────────────────────────────────
        loss = criterion(wsi_logits, labels)

        # ── Backward ──────────────────────────────────────────────────────────
        if mode == "train":
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Acumulación ───────────────────────────────────────────────────────
        batch_sz = len(inputs)
        total_seen     += batch_sz
        total_loss_sum += accelerator.gather(loss.detach() * batch_sz).sum().item()

        probs = torch.softmax(wsi_logits.detach(), dim=1).cpu()   # (B, num_classes)
        preds = probs.argmax(dim=1)                                # (B,)

        all_y_true.append(labels.detach().cpu())
        all_y_pred.append(preds)
        all_y_prob.append(probs)

        # ── Heatmap info ──────────────────────────────────────────────────────
        if inference_heatmap:
            for i in range(batch_sz):
                heatmap_list.append({
                    "wsi_id":      wsi_ids[i],
                    "pred_label":  preds[i].item(),
                    "true_label":  labels[i].item(),
                    "coords":      coords[i],
                    "patch_scores": torch.argmax(patch_probs_list[i], dim=1).cpu().numpy(),
                    "weights":      weights_list[i],
                })

    # ── Métricas ──────────────────────────────────────────────────────────────
    total_seen = accelerator.reduce(total_seen, reduction="sum").item()
    epoch_loss = total_loss_sum / total_seen

    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()
    y_prob = torch.cat(all_y_prob).numpy()

    # Para tareas binarias pasamos la probabilidad de la clase positiva
    y_prob_for_metrics = y_prob[:, 1] if y_prob.shape[1] == 2 else None
    metrics = compute_breast_metrics(task, y_true, y_pred, y_prob_for_metrics)

    heatmap_df = pd.DataFrame(heatmap_list) if inference_heatmap else None

    return metrics, epoch_loss, heatmap_df
