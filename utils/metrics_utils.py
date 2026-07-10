import torch
import os
import csv
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    classification_report
)


# ──────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_multiclass_metrics(y_true, y_pred, y_prob):
    """
    Métricas estándar de clasificación multiclase.
    Si se proporcionan probabilidades, añade AUROC y AUPRC.

    Args:
        y_true: array / Tensor con etiquetas verdaderas.
        y_pred: array / Tensor con etiquetas predichas.
        y_prob: array / Tensor (N, num_classes) con probabilidades.

    Returns:
        dict con metricas multiclass.
    """
    # --- Conversión ---
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "precision_macro":    precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),

        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),

        "f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),

        # Reporte completo por clase
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    # --- Métricas probabilísticas ---
    try:
        metrics["auroc_macro"]    = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["auroc_weighted"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")

        # AUPRC multiclase
        n_classes  = y_prob.shape[1]
        y_true_ohe = np.eye(n_classes)[y_true]

        metrics["auprc_macro"]    = average_precision_score(y_true_ohe, y_prob, average="macro")
        metrics["auprc_weighted"] = average_precision_score(y_true_ohe, y_prob, average="weighted")

    except ValueError as e:
        print(f"[WARN] AUROC/AUPRC no calculable: {e}", flush=True)
        metrics["auroc_macro"] = float("nan")
        metrics["auroc_weighted"] = float("nan")
        metrics["auprc_macro"] = float("nan")
        metrics["auprc_weighted"] = float("nan")

    return metrics


def compute_binary_metrics(y_true, y_pred, y_prob):
    """
    Métricas para clasificación binaria.
    Añade AUROC y AUPRC si se proporcionan probabilidades.

    Args:
        y_true:  array / Tensor con etiquetas 0/1.
        y_pred:  array / Tensor con predicciones 0/1.
        y_prob:  array / Tensor (N, num_classes) con probabilidades

    Returns:
        dict con métricas binarias.
    """
    # --- Conversión a numpy ---
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()


    # --- Métricas base ---
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        # Por clase
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_pos":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pos":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),

        "precision_neg": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_neg":    recall_score(y_true, y_pred, pos_label=0, zero_division=0),  # specificity
        "f1_neg":        f1_score(y_true, y_pred, pos_label=0, zero_division=0),

        # Agregadas
        "precision_macro":    precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":       recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),

        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # --- Métricas con probabilidades ---
    try:
        y_prob_pos = y_prob[:, 1]  # para auroc y auprc_pos
        y_prob_neg = y_prob[:, 0]   # probabilidad directa de la clase negativa

        # AUROC (clase positiva)
        metrics["auroc"] = roc_auc_score(y_true, y_prob_pos)

        # AUPRC para ambas clases
        metrics["auprc_pos"] = average_precision_score(y_true, y_prob_pos)
        metrics["auprc_neg"] = average_precision_score(1 - y_true, y_prob_neg)

        # AUPRC macro / weighted
        y_true_ohe = np.eye(2)[y_true]

        metrics["auprc_macro"] = average_precision_score(y_true_ohe, y_prob, average="macro")
        metrics["auprc_weighted"] = average_precision_score(y_true_ohe, y_prob, average="weighted")

    except ValueError as e:
        print(f"[WARN] AUROC/AUPRC no calculable: {e}", flush=True)
        metrics["auroc"] = float("nan")
        metrics["auprc_pos"] = float("nan")
        metrics["auprc_neg"] = float("nan")
        metrics["auprc_macro"] = float("nan")
        metrics["auprc_weighted"] = float("nan")
            
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# TASK-LEVEL DISPATCHER
# ──────────────────────────────────────────────────────────────────────────────
def compute_breast_metrics(task, y_true, y_pred, y_prob):
    """
    Selecciona la función de métricas adecuada según la tarea.

    Args:
        task (str): 'molecular_subtype' | 'molecular_subtype_from_biomarkers' → multiclase (4 clases)
                    'er'|'pr'|'her2'|'ki67' → binaria
        y_true, y_pred: arrays / Tensors.
        y_prob: array / Tensor (N, num_classes) con probabilidades completas.

    Returns:
        dict con las métricas correspondientes.
    """
    if task in ("molecular_subtype", "molecular_subtype_from_biomarkers"):
        return compute_multiclass_metrics(y_true, y_pred, y_prob)
    else:
        return compute_binary_metrics(y_true, y_pred, y_prob)


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

    # ── Métricas globales ─────────────────────────────────────────────────────
    if "accuracy" in metrics:
        lines.append(f"  Accuracy            : {metrics['accuracy']:.4f}")
    if "balanced_accuracy" in metrics:
        lines.append(f"  Balanced Accuracy   : {metrics['balanced_accuracy']:.4f}")

    # ── Métricas agregadas ────────────────────────────────────────────────────
    if "precision_macro" in metrics:
        lines.append(f"  Precision (macro)   : {metrics['precision_macro']:.4f}")
    if "precision_weighted" in metrics:
        lines.append(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")

    if "recall_macro" in metrics:
        lines.append(f"  Recall (macro)      : {metrics['recall_macro']:.4f}")
    if "recall_weighted" in metrics:
        lines.append(f"  Recall (weighted)   : {metrics['recall_weighted']:.4f}")

    if "f1_macro" in metrics:
        lines.append(f"  F1 (macro)          : {metrics['f1_macro']:.4f}")
    if "f1_weighted" in metrics:
        lines.append(f"  F1 (weighted)       : {metrics['f1_weighted']:.4f}")

    # ── Métricas robustas ─────────────────────────────────────────────────────
    if "mcc" in metrics:
        lines.append(f"  MCC                 : {metrics['mcc']:.4f}")
    if "kappa" in metrics:
        lines.append(f"  Kappa               : {metrics['kappa']:.4f}")

    # ── AUROC ─────────────────────────────────────────────────────────────────
    if "auroc" in metrics:
        lines.append(f"  AUROC               : {metrics['auroc']:.4f}")
    else:
        for avg in ("macro", "weighted"):
            k = f"auroc_{avg}"
            if k in metrics:
                lines.append(f"  AUROC ({avg})       : {metrics[k]:.4f}")

    # ── AUPRC ─────────────────────────────────────────────────────────────────
    # Agregadas Binario/Multiclase
    for avg in ("macro", "weighted"):
        k = f"auprc_{avg}"
        if k in metrics:
            lines.append(f"  AUPRC ({avg})       : {metrics[k]:.4f}")

    
    # ── Métricas por clase (binario) ──────────────────────────────────────────
    if "precision_pos" in metrics:
        lines.append("  --- Clase positiva (1) ---")
        lines.append(f"    Precision         : {metrics['precision_pos']:.4f}")
        lines.append(f"    Recall            : {metrics['recall_pos']:.4f}")
        lines.append(f"    F1                : {metrics['f1_pos']:.4f}")
        if "auprc_pos" in metrics:
            lines.append(f"    AUPRC             : {metrics['auprc_pos']:.4f}")

    if "precision_neg" in metrics:
        lines.append("  --- Clase negativa (0) ---")
        lines.append(f"    Precision         : {metrics['precision_neg']:.4f}")
        lines.append(f"    Recall (Spec.)    : {metrics['recall_neg']:.4f}")
        lines.append(f"    F1                : {metrics['f1_neg']:.4f}")
        if "auprc_neg" in metrics:
            lines.append(f"    AUPRC             : {metrics['auprc_neg']:.4f}")

    
    # ── Classification report (multiclase) ────────────────────────────────────
    if "classification_report" in metrics:
        lines.append("\n  --- Classification Report ---")

        report = metrics["classification_report"]

        excluded_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}

        for label, stats in report.items():
            if label in excluded_keys:
                continue

            if isinstance(stats, dict):  # solo clases
                lines.append(f"    [{label}]")
                for metric_name, value in stats.items():
                    lines.append(f"      {metric_name:10s}: {value:.4f}")

    # ── Output ────────────────────────────────────────────────────────────────
    output = "\n".join(lines)
    print(output, flush=True)

    with open(output_path, "a") as f:
        f.write(output + "\n")


def save_metrics_to_csv(metrics, loss, epoch, task, filepath="metrics.csv"):
    """
    Añade una fila de métricas al CSV de resultados.

    Args:
        metrics (dict): salida de compute_breast_metrics.
        loss (float): pérdida de la época.
        epoch (int): número de época.
        task (str): nombre de la tarea.
        filepath (str): ruta del CSV.
    """
    row = {"epoch": epoch, "task": task, "loss": loss}

    for k, v in metrics.items():
        if k == "classification_report":
            # Guardar como JSON string
            row[k] = json.dumps(v)
        else:
            row[k] = v

    fieldnames = list(row.keys())

    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
