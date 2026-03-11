import torch
import os
import csv
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)


# ──────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Métricas estándar de clasificación multiclase.
    Si se proporcionan probabilidades, añade AUROC y AUPRC.

    Args:
        y_true: array / Tensor con etiquetas verdaderas.
        y_pred: array / Tensor con etiquetas predichas.
        y_prob: array / Tensor (N, num_classes) con probabilidades (opcional).

    Returns:
        dict con metricas multilabel.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_micro":    precision_score(y_true, y_pred, average="micro",    zero_division=0),
        "precision_macro":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),

        "recall_micro":    recall_score(y_true, y_pred, average="micro",    zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),

        "f1_micro":    f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["auroc_micro"]    = roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro")
            metrics["auroc_macro"]    = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["auroc_weighted"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")

            # AUPRC — necesita y_true en one-hot para multi_class
            n_classes  = y_prob.shape[1]
            y_true_ohe = np.eye(n_classes)[y_true]   # (N, C) one-hot
            metrics["auprc_micro"]    = average_precision_score(y_true_ohe, y_prob, average="micro")
            metrics["auprc_macro"]    = average_precision_score(y_true_ohe, y_prob, average="macro")
            metrics["auprc_weighted"] = average_precision_score(y_true_ohe, y_prob, average="weighted")
        except ValueError as e:
            print(f"  [WARN] AUROC/AUPRC no calculable: {e}", flush=True)
            for k in ("auroc_micro", "auroc_macro", "auroc_weighted",
                      "auprc_micro", "auprc_macro", "auprc_weighted"):
                metrics[k] = float("nan")

    return metrics


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """
    Métricas para clasificación binaria.
    Añade AUROC y AUPRC si se proporcionan probabilidades.

    Args:
        y_true:  array / Tensor con etiquetas 0/1.
        y_pred:  array / Tensor con predicciones 0/1.
        y_prob:  array / Tensor con probabilidad de la clase positiva (opcional).

    Returns:
        dict con métricas binarias.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),

        "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),

        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        except ValueError as e:
            print(f"  [WARN] AUROC/AUPRC no calculable: {e}", flush=True)
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# TASK-LEVEL DISPATCHER
# ──────────────────────────────────────────────────────────────────────────────
def compute_breast_metrics(task, y_true, y_pred, y_prob=None):
    """
    Selecciona la función de métricas adecuada según la tarea.

    Args:
        task (str): 'molecular_subtype' → multiclase (4 clases)
                    'er'|'pr'|'her2'|'ki67' → binaria
        y_true, y_pred: arrays / Tensors.
        y_prob: array / Tensor (N, num_classes) con probabilidades completas.

    Returns:
        dict con las métricas correspondientes.
    """
    if task == "molecular_subtype":
        return compute_metrics(y_true, y_pred, y_prob)
    else:
        # Para binaria pasamos solo la probabilidad de la clase positiva
        y_prob_positive = y_prob[:, 1] if y_prob is not None else None
        return compute_binary_metrics(y_true, y_pred, y_prob_positive)


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def log_epoch_metrics(epoch, split_name, task, metrics, loss, output_path):
    """
    Imprime y guarda en fichero las métricas de una época.
    """
    lines = []
    lines.append(f"\n=== Época {epoch} | {split_name.upper()} | task={task} ===")
    lines.append(f"Loss: {loss:.4f}")
    lines.append(f"  Accuracy            : {metrics['accuracy']:.4f}")

    # ── Precision ─────────────────────────────────────────────────────────────
    if "precision_micro" in metrics:
        lines.append(f"  Precision (micro)   : {metrics['precision_micro']:.4f}")
    lines.append(f"  Precision (macro)   : {metrics['precision_macro']:.4f}")
    lines.append(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")

    # ── Recall ────────────────────────────────────────────────────────────────
    if "recall_micro" in metrics:
        lines.append(f"  Recall (micro)      : {metrics['recall_micro']:.4f}")
    lines.append(f"  Recall (macro)      : {metrics['recall_macro']:.4f}")
    lines.append(f"  Recall (weighted)   : {metrics['recall_weighted']:.4f}")

    # ── F1 ────────────────────────────────────────────────────────────────────
    if "f1_micro" in metrics:
        lines.append(f"  F1 (micro)          : {metrics['f1_micro']:.4f}")
    lines.append(f"  F1 (macro)          : {metrics['f1_macro']:.4f}")
    lines.append(f"  F1 (weighted)       : {metrics['f1_weighted']:.4f}")

    # ── Kappa ─────────────────────────────────────────────────────────────────
    lines.append(f"  Kappa               : {metrics['kappa']:.4f}")

    # ── AUROC ─────────────────────────────────────────────────────────────────
    if "auroc" in metrics:
        # Binaria: una sola curva
        lines.append(f"  AUROC               : {metrics['auroc']:.4f}")
    else:
        for avg in ("micro", "macro", "weighted"):
            k = f"auroc_{avg}"
            if k in metrics:
                lines.append(f"  AUROC ({avg:8s})  : {metrics[k]:.4f}")

    # ── AUPRC ─────────────────────────────────────────────────────────────────
    if "auprc" in metrics:
        lines.append(f"  AUPRC               : {metrics['auprc']:.4f}")
    else:
        for avg in ("micro", "macro", "weighted"):
            k = f"auprc_{avg}"
            if k in metrics:
                lines.append(f"  AUPRC ({avg:8s})  : {metrics[k]:.4f}")

    output = "\n".join(lines)
    print(output, flush=True)
    with open(output_path, "a") as f:
        f.write(output + "\n")


def save_metrics_to_csv(metrics, epoch, task, filepath="metrics.csv"):
    """
    Añade una fila de métricas al CSV de resultados.

    Args:
        metrics (dict): salida de compute_breast_metrics.
        epoch (int): número de época.
        task (str): nombre de la tarea.
        filepath (str): ruta del CSV.
    """
    row = {"epoch": epoch, "task": task, **metrics}
    fieldnames = list(row.keys())

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
