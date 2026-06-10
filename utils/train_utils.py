"""
utils/train_utils.py
--------------------
Utilidades de entrenamiento: semillas, get_model, splits k-fold, run_epoch.
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
    "keep": {
        "keep_model_name": "Astaxanthin/KEEP",
        "patch_size":      224,
        "embed_dim":       768,
    },
    "gigapath": {
        "gigapath_model_name": "prov-gigapath/prov-gigapath",
        "patch_size":          224, #TODO Revisar (HF dice 256 pero luego hacen center crop a 224)
        "embed_dim":           1536,    # ViT-G/14
    },
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
# SPLIT INICIAL DEV / TEST
# ──────────────────────────────────────────────────────────────────────────────

def stratified_dev_test_split(
    df: pd.DataFrame,
    label_col: str = "molecular_subtype",
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple:
    """
    Divide el DataFrame maestro en dev (80 %) y test (20 %) de forma
    estratificada a nivel de grupo de paciente.

    Estrategia
    ----------
    Se usa StratifiedGroupKFold con n_splits = round(1 / test_size) para
    obtener exactamente un fold como test (≈ test_size) y el resto como dev,
    garantizando:
        · Todas las WSI de un mismo paciente caen en el mismo subconjunto.
        · La distribución de clases de `label_col` es similar en dev y test.

    Solo se realiza una iteración del splitter (el primer fold actúa como test).

    Args
    ----
    df        : pd.DataFrame maestro completo (salida de create_file_label_df).
                Debe contener las columnas 'patient_id' y `label_col`.
    label_col : str — columna usada como estrato para la estratificación.
                Se recomienda 'molecular_subtype' por ser la etiqueta multiclase.
    test_size : float — proporción aproximada reservada para test (por defecto 0.20).
    seed      : int — semilla para reproducibilidad.

    Returns
    -------
    dev_df  : pd.DataFrame con el 80 % de los datos (índice reseteado).
    test_df : pd.DataFrame con el 20 % de los datos (índice reseteado).

    Raises
    ------
    ValueError
        Si `label_col` no existe en df o contiene valores -1 (inválidos).
    """
    if label_col not in df.columns:
        raise ValueError(
            f"La columna '{label_col}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Filtrar muestras con etiqueta válida para el estrato
    df_valid = df[df[label_col] != -1].reset_index(drop=True)
    n_invalid = len(df) - len(df_valid)
    if n_invalid > 0:
        print(
            f"[WARN] stratified_dev_test_split: se excluyen {n_invalid} WSI con "
            f"etiqueta -1 en '{label_col}'. Se incluirán en dev por separado.",
            flush=True,
        )

    n_splits = round(1.0 / test_size)   # 0.20 → 5 splits; el primer fold es test (~20 %)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Una sola iteración: primer fold → test, resto → dev
    split_iter = sgkf.split(
        np.arange(len(df_valid)),
        df_valid[label_col].values,
        groups=df_valid["patient_id"].values,
    )
    dev_pos_idx, test_pos_idx = next(split_iter)

    dev_df  = df_valid.iloc[dev_pos_idx].reset_index(drop=True)
    test_df = df_valid.iloc[test_pos_idx].reset_index(drop=True)

    # Añadir al dev las muestras con etiqueta inválida (si las hubiera)
    if n_invalid > 0:
        invalid_df = df[df[label_col] == -1].reset_index(drop=True)
        dev_df = pd.concat([dev_df, invalid_df], ignore_index=True)

    # ── Reporte ───────────────────────────────────────────────────────────────
    total = len(dev_df) + len(test_df)
    print(
        f"\n{'='*55}\n"
        f"SPLIT DEV / TEST  (estrato='{label_col}',  seed={seed})\n"
        f"{'='*55}\n"
        f"  Total : {total:>4d} WSIs\n"
        f"  Dev   : {len(dev_df):>4d} WSIs  ({len(dev_df)/total:.1%})\n"
        f"  Test  : {len(test_df):>4d} WSIs  ({len(test_df)/total:.1%})\n",
        flush=True,
    )

    for split_name, split_df in [("DEV", dev_df), ("TEST", test_df)]:
        counts = split_df[label_col].value_counts().sort_index()
        props  = split_df[label_col].value_counts(normalize=True).sort_index()
        print(f"  [{split_name}]  distribución de '{label_col}':", flush=True)
        for cls in counts.index:
            if cls == -1:
                continue
            print(f"    clase {cls}: {counts[cls]:>4d}  ({props[cls]:.1%})", flush=True)

    return dev_df, test_df


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
        El embed_dim se lee de ENCODER_CONFIGS y se descarta antes de pasarlo
        al encoder para evitar kwargs inesperados.

    skip_encoder=True (modo embeddings precomputados):
        Solo instancia el MIL + clasificador, sin cargar ningún backbone.
        El embed_dim se lee directamente de ENCODER_CONFIGS.
        Requiere usar WSIClassifier.forward_from_embeddings() en lugar de forward().

    Args:
        patch_encoder_type (str) : 'dino' | 'uni' | 'keep' .
        mil_aggregation_mode (str): 'max' | 'mean' | 'abMIL-instance-level' | 'abMIL-bag-level' | 'transMIL'.
        unfreeze_backbone (bool) : descongelar últimos bloques del backbone (ignorado si skip_encoder=True).
        num_classes (int)        : número de clases de salida.
        chunk_size (int)         : mini-batch interno de parches (ignorado si skip_encoder=True).
        skip_encoder (bool)      : si True, no instancia el encoder — Se deben usar con embeddings precomputados.

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
        # Solo MIL + clasificador, sin backbone. Útil cuando los embeddings ya están precomputados en disco y no se quiere cargar el encoder en memoria.
        model = WSIClassifier(
            encoder=None,
            mil_aggregation_mode=mil_aggregation_mode,
            num_classes=num_classes,
            embed_dim=ENCODER_CONFIGS[key]["embed_dim"],
        )
    else:
        # embed_dim se descarta porque el encoder lo asigna internamente
        # y no lo acepta como argumento directo.
        encoder_kwargs = {k: v for k, v in ENCODER_CONFIGS[key].items() if k != "embed_dim"}
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
# K-FOLD SPLITS
# ──────────────────────────────────────────────────────────────────────────────

def make_kfold_splits(df: pd.DataFrame, label_col: str, seed: int, k: int = 4):
    """
    Genera los k arrays de índices base para cross-validation mediante
    k-fold estratificado por grupo de paciente, sin data leakage entre folds.

    La asignación concreta de train/val a partir de estos arrays se realiza
    en `run_kfolds` de train.py, donde cada fold i actúa como validación y
    el resto como entrenamiento.

    La estratificación se aplica a nivel de grupo de paciente con
    StratifiedGroupKFold, garantizando que:
        · Todas las WSI de un mismo paciente caen en el mismo fold.
        · La distribución de clases es similar en todos los folds.

    Args:
        df        : pd.DataFrame filtrado para la tarea (sin etiquetas -1).
                    Debe contener la columna 'patient_id' como string.
        label_col : str  columna de etiquetas (== nombre de la tarea).
        seed      : int  semilla para reproducibilidad.
        k         : int  número de folds (por defecto 4).

    Returns:
        fold_indices     : lista de k arrays con los índices posicionales
                           de cada fold base.
        kfold_report_str : str — resumen de distribuciones por fold.
    """
    if k < 2:
        raise ValueError(
            f"El número de folds debe ser ≥ 2 para tener al menos 1 fold de train "
            f"separado del de validación. Valor recibido: k={k}."
        )

    # ── Generar los k folds base ──────────────────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_indices = [
        test_pos_idx
        for _, test_pos_idx in sgkf.split(
            np.arange(len(df)),
            df[label_col].values,
            groups=df["patient_id"].values,
        )
    ]

    # ── Construir las k iteraciones train / val ───────────────────────────────
    def _proportions_str(name, idx_array):
        subset = df.iloc[idx_array]
        counts = subset[label_col].value_counts().sort_index()
        props  = subset[label_col].value_counts(normalize=True).sort_index()
        lines  = [f"  [{name}]  n={len(subset)}"]
        for cls in counts.index:
            lines.append(f"    clase {cls}: {counts[cls]:>4d}  ({props[cls]:.1%})")
        return "\n".join(lines)

    report_lines = [
        "=" * 55,
        f"K-FOLD SPLITS — tarea: '{label_col}'  k={k}  seed={seed}",
        "=" * 55,
        _proportions_str("TOTAL", np.arange(len(df))),
    ]

    for i in range(k):
        val_idx   = fold_indices[i]
        train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != i])

        report_lines.append(f"\n── Fold {i} ──────────────────────────────────────")
        report_lines.append(_proportions_str(f"TRAIN (folds ≠ {i})", train_idx))
        report_lines.append(_proportions_str(f"VAL   (fold {i})",     val_idx))

    report_str = "\n".join(report_lines)
    print(report_str, flush=True)

    return fold_indices, report_str


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

        precomputed_embeddings (bool): si True, el dataloader devuelve embeddings en vez de imágenes y se usa model.forward_from_embeddings().

    Returns:
        metrics (dict): métricas de la época.
        conf_mat_df (pd.DataFrame): matriz de confusión.
        epoch_loss (float): pérdida media.
    """
    assert mode in ("train", "val", "test")

    if mode == "train":
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(mode == "train"):

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
            inputs, labels, wsi_ids, coords = batch_info
            labels = labels.to(device)

            if mode == "train":
                optimizer.zero_grad(set_to_none=True)
            
            if precomputed_embeddings:
                # inputs es lista de tensores de embeddings — se mueven a device y se procesan con model.forward_from_embeddings()
                inputs_device = [emb.to(device, non_blocking=True) for emb in inputs]
                wsi_logits, patch_logits_list, weights_list = model.forward_from_embeddings(inputs_device)
            else:
                # inputs es lista de tensores de imágenes — se procesan con model.forward() y se mueven a device dentro del encoder.forward
                wsi_logits, patch_logits_list, weights_list = model(inputs)

            loss = criterion(wsi_logits, labels)

            if mode == "train":
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(wsi_logits.detach(), dim=1)
            preds = probs.argmax(dim=1)

            # Loss - Lógica de acumulación reducida para multi-GPU para obtener la pérdida media global al final de la época, sin sesgo por el tamaño del batch del proceso.
            batch_size = labels.size(0)
            loss_value   = loss.detach() * batch_size                                   
            count_value  = torch.tensor(batch_size, dtype=torch.float32, device=device)

            loss_sum_reduced  = accelerator.reduce(loss_value,  reduction="sum")
            count_reduced     = accelerator.reduce(count_value, reduction="sum")

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

        epoch_loss = total_loss_sum / total_seen

        metrics = {}
        conf_mat_df = pd.DataFrame()

        accelerator.wait_for_everyone()  # Asegura que todos han terminado antes de procesar métricas y heatmaps
        if accelerator.is_main_process:
            y_true = torch.cat(all_y_true).numpy()
            y_pred = torch.cat(all_y_pred).numpy()
            y_prob = torch.cat(all_y_prob).numpy()

            metrics = compute_breast_metrics(task, y_true, y_pred, y_prob)

            conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
            conf_mat_df = pd.DataFrame(
                conf_mat,
                index=[f"true_{i}" for i in range(conf_mat.shape[0])],
                columns=[f"pred_{i}" for i in range(conf_mat.shape[1])],
            )

    return metrics, conf_mat_df, epoch_loss
