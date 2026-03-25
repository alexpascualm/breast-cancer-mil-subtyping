import torch
import torch.nn as nn
import os
import pandas as pd
import random
import numpy as np
import sys
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

from models.patch_encoders.VIT_model import PatchClassifierVIT
from utils.metrics_utils import compute_breast_metrics
from sklearn.metrics import confusion_matrix

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

        lines = [f"\n  [{name}]  n={len(subset)}"]
        for cls in counts.index:
            lines.append(f"    clase {cls}: {counts[cls]:>4d}  ({props[cls]:.1%})")

        return "\n".join(lines)

    print("=" * 45)
    print(f"SPLITS — tarea: '{label_col}'  seed={seed}")
    print("=" * 45)

    # Acumulador de texto
    report_lines = []
    report_lines.append("=" * 45)
    report_lines.append(f"SPLITS — tarea: '{label_col}'  seed={seed}")
    report_lines.append("=" * 45)

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
        conf_mat_df (pd.DataFrame): matriz de confusión.
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

    metrics = compute_breast_metrics(task, y_true, y_pred, y_prob)
    
    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(model.num_classes))
    conf_mat_df = pd.DataFrame(
                conf_mat,
                index=[f"true_{i}" for i in range(conf_mat.shape[0])],
                columns=[f"pred_{i}" for i in range(conf_mat.shape[1])]
            )

    heatmap_df = pd.DataFrame(heatmap_list) if inference_heatmap else None

    return metrics, conf_mat_df, epoch_loss, heatmap_df
