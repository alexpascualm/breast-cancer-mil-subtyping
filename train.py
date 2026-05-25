import argparse
import os
import copy
import itertools
import shutil
from datetime import datetime
import time
import numpy as np
from huggingface_hub import login

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from accelerate import Accelerator
from accelerate.utils import broadcast
import gc

from data_load import (
    Patches_WSI_Dataset,
    Embeddings_WSI_Dataset,
    create_file_label_df,
    collate_fn,
)
from utils.train_utils import (
    set_multiple_seed,
    get_model,
    compute_class_weights,
    make_kfold_splits,
    run_epoch,
    stratified_dev_test_split,
)
from utils.metrics_utils import log_epoch_metrics, save_metrics_to_csv
from config import bcnb_images_folder_path, bcnb_labels_df_path, bcnb_embeddings_path, results_folder_path


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MIL breast cancer")

    # ── Datos ──────────────────────────────────────────────────────────────────
    parser.add_argument("--images_folder", type=str, default=bcnb_images_folder_path)
    parser.add_argument("--df_path",       type=str, default=bcnb_labels_df_path)
    parser.add_argument("--slide_col",     type=str, default="Patient ID",
                        help="Columna del Excel que identifica cada WSI")
    parser.add_argument("--results_path",  type=str, default=results_folder_path)

    # ── Tarea ─────────────────────────────────────────────────────────────────
    parser.add_argument("--task", type=str, default="molecular_subtype",
                        choices=["molecular_subtype", "er", "pr", "her2", "ki67"],
                        help="Target de clasificación")
    parser.add_argument("--ki67_threshold", type=float, default=0.14,
                        help="Umbral de binarización de Ki67")
    parser.add_argument("--filter_ductal", action="store_true",
                        help="Entrenar solo con carcinomas ductales")

    # ── Modelo ────────────────────────────────────────────────────────────────
    parser.add_argument("--patch_model", type=str, default="dino",
                        choices=["dino", "uni"],
                        help="Extractor de características")
    parser.add_argument("--unfreeze_patch_model_backbone", action="store_true")
    parser.add_argument("--mil_aggregation_mode", type=str, default="abMIL-bag-level",
                        choices=["max-instance-level", "max-bag-level", "mean-instance-level", "mean-bag-level", "abMIL-instance-level", "abMIL-bag-level", "transMIL"])
    parser.add_argument("--patch_resize_size", type=int, default=224,
                        help="Dimensiones de reescalado de los parches")
    parser.add_argument("--precomputed_embeddings_folder", type=str, default=bcnb_embeddings_path,
                        help="Si se especifica, usa embeddings precomputados en lugar "
                             "de procesar imágenes con el encoder en cada época.")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--patience",      type=int,   default=20)
    parser.add_argument("--optimizer",     type=str,   default="AdamW",
                        choices=["SGD", "AdamW"])
    parser.add_argument("--momentum",      type=float, default=0.9,
                        help="Solo para SGD")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-3)
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Aplicar pesos de clase en la función de pérdida")

    # ── K-Fold ────────────────────────────────────────────────────────────────
    parser.add_argument("--kfold", type=int, default=4,
                        help="Número de folds para cross-validation (mínimo 2). ")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--load_on_demand",         action="store_true")
    parser.add_argument("--small_dataset_for_test", action="store_true")
    parser.add_argument("--seed",                   type=int, default=98)
    parser.add_argument("--experiment_nickname",    type=str, default="")
    parser.add_argument("--save_models",            action="store_true",
                        help="Si se activa, guarda checkpoints best_model y "
                             "last_model en disco.")

    parser.add_argument("--resume_training",       action="store_true")
    parser.add_argument("--pretrained_model_path", type=str, default=None)

    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token para descargar modelos privados")

    # ── Modo multi-configuración ───────────────────────────────────────────────
    parser.add_argument("--run_all_configs", action="store_true",
                        help="Ejecuta automáticamente todas las combinaciones "
                             "Tarea definidas x encoder x MIL x hiperparámetros  en las listas ALL_X")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIONES PARA --run_all_configs
# Edita estas listas para añadir o quitar tareas, encoders y modos MIL.
# ──────────────────────────────────────────────────────────────────────────────

ALL_TASKS = [
    "er",
    "pr",
    "her2",
    "ki67",
    "molecular_subtype",
]

ALL_ENCODERS = [
    "uni",
    "dino",
]

ALL_MIL_MODES = [
    "max-instance-level",
    "mean-instance-level",
    "max-bag-level",
    "mean-bag-level",
    "abMIL-instance-level",
    "abMIL-bag-level",
    "transMIL",
]

ALL_LEARNING_RATES = [
    1e-3,
    1e-4,
    1e-5,
]

ALL_WEIGHT_DECAYS = [
    1e-3,
    1e-4,
]


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 · SPLIT INICIAL DEV / TEST
# Genera el DataFrame maestro completo y lo divide en dev (80 %) y test (20 %).
# Guarda físicamente ambos splits en results/ antes de continuar.
# ──────────────────────────────────────────────────────────────────────────────

def make_initial_data_split(args) -> tuple:
    """
    Construye el DataFrame maestro y realiza el split estratificado dev/test.

    El split se hace a nivel de 'molecular_subtype' para que la distribución
    de subtipos sea representativa en ambos conjuntos, usando la columna
    'patient_id' como grupo para evitar data-leakage entre pacientes.

    Proporciones
    ------------
    - dev  : 80 % del total de pacientes únicos
    - test : 20 % del total de pacientes únicos

    Persistencia
    ------------
    Ambos DataFrames se guardan en CSV dentro de `results/` antes de retornar:
        results/dev_df.csv
        results/test_df.csv

    Args
    ----
    args : Namespace con los argumentos de línea de comandos.

    Returns
    -------
    dev_df  : pd.DataFrame con las muestras de desarrollo (80 %).
    test_df : pd.DataFrame con las muestras de test (20 %).
    """
    print(f"\n{'='*60}", flush=True)
    print("Preparando datos: split inicial DEV / TEST", flush=True)
    print(f"{'='*60}", flush=True)

    # ── DataFrame maestro ─────────────────────────────────────────────────────
    df = create_file_label_df(
        images_path=args.images_folder,
        data_path=args.df_path,
        slide_col=args.slide_col,
        small_dataset_for_test=args.small_dataset_for_test,
        ki67_threshold=args.ki67_threshold,
    )

    if args.filter_ductal:
        before = len(df)
        df = df[df["tumour_type"] == "Invasive ductal carcinoma"].reset_index(drop=True)
        print(f"Filtro ductal: {before} → {len(df)} WSIs", flush=True)

    df["patient_id"] = df["patient_id"].astype(str)

    print(f"Total WSIs en el DataFrame maestro: {len(df)}", flush=True)

    # ── Split estratificado 80/20 usando molecular_subtype como estrato ───────
    # Se estratifica por molecular_subtype porque es la etiqueta con más clases
    # y garantiza representatividad en los splits de todas las tareas.
    dev_df, test_df = stratified_dev_test_split(
        df,
        label_col="molecular_subtype",
        test_size=0.20,
        seed=args.seed,
    )

    print(
        f"\nSplit completado:\n"
        f"  dev  → {len(dev_df):>4d} WSIs ({len(dev_df)/len(df):.1%})\n"
        f"  test → {len(test_df):>4d} WSIs ({len(test_df)/len(df):.1%})",
        flush=True,
    )

    # ── Guardar copias físicas en results/ ────────────────────────────────────
    os.makedirs(args.results_path, exist_ok=True)
    dev_path  = os.path.join(args.results_path, "dev_df.csv")
    test_path = os.path.join(args.results_path, "test_df.csv")

    dev_df.to_csv(dev_path,  index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nDataFrames guardados en:\n  {dev_path}\n  {test_path}", flush=True)

    return dev_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 · CONFIGURACIÓN DE VALIDACIÓN
# Filtra dev_df para la tarea concreta, aplica make_kfold_splits y carga los datasets base (uno por fold).
# Devuelve fold_indices, el resumen de proporciones y los datasets base.
# NO construye dataloaders, NO lanza entrenamiento.
# ──────────────────────────────────────────────────────────────────────────────

def prepare_task_data(args, dev_df, task: str, patch_model: str) -> tuple:
    """
    Prepara la partición K-Fold y carga los datasets base para una tarea.

    Recibe exclusivamente `dev_df` (nunca el test set), aplica make_kfold_splits
    para generar los índices de validación cruzada y construye un dataset base
    por fold.

    Args
    ----
    args        : Namespace con los argumentos de línea de comandos.
    dev_df      : pd.DataFrame con las muestras de desarrollo.
    task        : str — nombre de la tarea / columna de etiquetas.
    patch_model : str — nombre del encoder; necesario para resolver la carpeta
                  de embeddings precomputados cuando aplica.

    Returns
    -------
    df_task          : pd.DataFrame filtrado para la tarea (solo dev).
    fold_indices     : lista de k np.ndarray con los índices posicionales de
                       cada fold sobre df_task.
    kfold_report_str : str — descripción de las proporciones de cada fold.
    base_datasets    : lista de k datasets (uno por fold).
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Configurando K-Fold para tarea: '{task}'", flush=True)
    print(f"{'='*60}", flush=True)

    # Filtrar muestras con etiqueta válida para esta tarea
    df_task = dev_df[dev_df[task] != -1].reset_index(drop=True)
    print(f"WSIs con etiqueta válida para '{task}': {len(df_task)}", flush=True)
    print(f"Distribución:\n{df_task[task].value_counts()}", flush=True)

    # Partición K-Fold estratificada por grupo de paciente
    fold_indices, kfold_report_str = make_kfold_splits(
        df_task,
        label_col=task,
        seed=args.seed,
        k=args.kfold,
    )

    # ── Construir datasets base (uno por fold) ────────────────────────────────
    precomputed = args.precomputed_embeddings_folder is not None
    emb_folder = (
        os.path.join(args.precomputed_embeddings_folder, patch_model)
        if precomputed else None
    )

    def _make_base_dataset(idx):
        """Construye un dataset para los índices posicionales `idx`."""
        if emb_folder:
            return Embeddings_WSI_Dataset(
                df_task, idx, task=task,
                embeddings_folder=emb_folder,
                load_on_demand=args.load_on_demand)
        return Patches_WSI_Dataset(
            df_task, idx, task=task,
            patch_resize_size=args.patch_resize_size,
            load_on_demand=args.load_on_demand)

    k = len(fold_indices)
    t0 = time.time()
    print(f"Cargando {k} datasets base ...", flush=True)
    base_datasets = []
    for fold_i, idx in enumerate(fold_indices):
        ds = _make_base_dataset(idx)
        base_datasets.append(ds)
        print(f"  Fold base {fold_i}: {len(ds)} WSIs", flush=True)
    print(f"Tiempo de carga: {time.time() - t0:.1f}s", flush=True)

    return df_task, fold_indices, kfold_report_str, base_datasets


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 · LÓGICA DE ENTRENAMIENTO
# Opera únicamente sobre folds de desarrollo (train + val).
# No utiliza el test set original.
# Al finalizar guarda best_eval.csv con las métricas de la mejor época de val.
# Estructura de almacenamiento: results/experiments/{experiment_path}/
# ──────────────────────────────────────────────────────────────────────────────

def run_kfolds(
    df_task,
    fold_indices,
    base_datasets,
    accelerator,
    device,
    args,
    kfold_report_str: str,
    task: str,
    patch_model: str,
    mil_aggregation_mode: str,
):
    """
    Ejecuta el entrenamiento con K-Fold cross-validation.

    Opera exclusivamente sobre los folds de desarrollo (train y validación).
    No existe ninguna referencia al test set original en este flujo.

    Recibe `base_datasets` ya construidos (uno por fold) desde `prepare_task_data`.
    En cada iteración solo instancia los DataLoaders a partir de ellos.

    Asignación de folds en cada iteración i (0-indexado):
        - val   ← fold  i
        - train ← todos los folds restantes (k-1 folds)

    Al finalizar todas las iteraciones genera `best_eval.csv` dentro de cada
    subcarpeta de fold con las métricas obtenidas en la mejor época de
    validación (según val_loss mínima).

    Estructura de almacenamiento
    ----------------------------
    results/
      experiments/
        {experiment_path}/         ← carpeta raíz aislada por experimento
          experiment_log.txt
          fold_0/
            train_log.txt
            train_metrics.csv
            val_metrics.csv
            best_eval.csv          ← métricas de val en la mejor época
            [best_model/]          ← solo si --save_models
            [last_model/]          ← solo si --save_models
          fold_1/
            ...

    Args
    ----
    df_task              : pd.DataFrame filtrado para la tarea (solo dev).
    fold_indices         : lista de k np.ndarray con índices posicionales de
                           cada fold sobre df_task.
    base_datasets        : lista de k datasets preconstruidos (uno por fold).
    accelerator          : Accelerator de HuggingFace.
    device               : torch.device.
    args                 : Namespace con todos los argumentos.
    kfold_report_str     : str — resumen de splits (se escribe en el log).
    task                 : str — nombre de la tarea activa.
    patch_model          : str — nombre del encoder activo.
    mil_aggregation_mode : str — modo de agregación MIL activo.
    """
    precomputed = args.precomputed_embeddings_folder is not None
    k = len(fold_indices)

    # ── Carpeta raíz de este experimento ──────────────────────────────────────
    # Almacenamiento aislado: results/experiments/{experiment_path}/
    exp_results_path = ""
    if accelerator.is_main_process:
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        nickname      = f"{args.experiment_nickname}_" if args.experiment_nickname else ""
        freeze_status = "unfrozen" if args.unfreeze_patch_model_backbone else "frozen"
        opt_name      = (
            args.optimizer
            if args.optimizer == "AdamW"
            else f"{args.optimizer}_mom{args.momentum}"
        )
        ductal        = "_ductal"        if args.filter_ductal  else ""
        weighted      = "_weighted_loss" if args.weighted_loss  else ""
        small         = "_small"         if args.small_dataset_for_test else ""

        run_name = (
            f"{nickname}{timestamp}"
            f"_task-{task}{ductal}{weighted}"
            f"_model-{patch_model}_{freeze_status}"
            f"_MIL-{mil_aggregation_mode}"
            f"_{opt_name}_lr{args.learning_rate}_wd{args.weight_decay}"
            f"_bs{args.batch_size}_ep{args.epochs}"
            f"_{k}fold{small}"
        )
        exp_results_path = os.path.join(args.results_path, "experiments", run_name)
        os.makedirs(exp_results_path, exist_ok=True)
        print(f"\nResultados del experimento en: {exp_results_path}", flush=True)

    # ── Log global del experimento ────────────────────────────────────────────
    exp_log_path = os.path.join(exp_results_path, "experiment_log.txt")
    if accelerator.is_main_process:
        with open(exp_log_path, "a") as f:
            f.write(kfold_report_str + "\n\n")

    # ── Iterar sobre los k folds ──────────────────────────────────────────────
    # Cada fold i usa fold_i como validación y el resto como entrenamiento.
    for fold_i in range(k):
        train_fold_js = [j for j in range(k) if j != fold_i]

        # Ensamblar splits combinando datasets base en memoria (sin releer disco)
        ds_val   = base_datasets[fold_i]
        ds_train = ConcatDataset([base_datasets[j] for j in train_fold_js])

        # Índices posicionales del train en df_task (necesarios para weighted_loss)
        train_global_indices = np.concatenate([fold_indices[j] for j in train_fold_js])

        if accelerator.is_main_process:
            print(f"\n{'─'*60}", flush=True)
            print(
                f"  FOLD {fold_i}/{k-1}  "
                f"[task={task}  encoder={patch_model}  MIL={mil_aggregation_mode}]",
                flush=True,
            )
            print(f"  TRAIN: {len(ds_train)} WSIs  |  VAL: {len(ds_val)} WSIs", flush=True)
            print(f"{'─'*60}", flush=True)

        # Subcarpeta aislada por fold dentro del experimento
        fold_results_path = os.path.join(exp_results_path, f"fold_{fold_i}")
        if accelerator.is_main_process:
            os.makedirs(fold_results_path, exist_ok=True)

        log_path        = os.path.join(fold_results_path, "train_log.txt")
        best_model_path = os.path.join(fold_results_path, "best_model")
        last_model_path = os.path.join(fold_results_path, "last_model")

        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                start_fold = f"── Fold {fold_i} "
                end_fold   = f"── Fold {(fold_i + 1) % k} "
                total_part = kfold_report_str.split("── Fold ")[0]
                fold_part  = kfold_report_str.split(start_fold)[1]
                if end_fold in fold_part:
                    fold_part = fold_part.split(end_fold)[0]
                f.write(f"=== Fold {fold_i} ===\n")
                f.write(total_part)
                f.write(start_fold + fold_part + "\n")

        # ── DataLoaders ───────────────────────────────────────────────────────
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        # ── Modelo (instancia fresca por fold) ────────────────────────────────
        num_classes = base_datasets[0].num_classes
        print(
            f"Cargando modelo '{patch_model}' — "
            f"num_classes={num_classes} ...",
            flush=True,
        )
        model = get_model(
            patch_encoder_type=patch_model,
            mil_aggregation_mode=mil_aggregation_mode,
            unfreeze_backbone=args.unfreeze_patch_model_backbone,
            num_classes=num_classes,
            skip_encoder=precomputed,
        ).to(device)

        # ── Pérdida ───────────────────────────────────────────────────────────
        if args.weighted_loss:
            train_labels = df_task.iloc[train_global_indices][task].tolist()
            weights      = compute_class_weights(train_labels, num_classes).to(device)
            criterion    = nn.CrossEntropyLoss(weight=weights)
            if accelerator.is_main_process:
                print("Aplicando pérdida con pesos por clase:", flush=True)
                with open(log_path, "a") as f:
                    f.write("Aplicando pérdida con pesos por clase:\n")
                for cls, w in enumerate(weights.cpu().tolist()):
                    msg = f"  clase {cls}: {w:.4f}  (n={train_labels.count(cls)})"
                    print(msg, flush=True)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
        else:
            criterion = nn.CrossEntropyLoss()

        # ── Optimizador y scheduler ───────────────────────────────────────────
        if args.optimizer == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=10, min_lr=1e-7, threshold=1e-4,
        )

        # ── Accelerate ────────────────────────────────────────────────────────
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

        if args.resume_training:
            if not args.pretrained_model_path:
                raise ValueError("--resume_training requiere --pretrained_model_path")
            accelerator.load_state(args.pretrained_model_path)
            print(f"Reanudando desde: {args.pretrained_model_path}", flush=True)

        # ── Bucle de entrenamiento ────────────────────────────────────────────
        best_val_loss    = float("inf")
        best_val_epoch   = -1
        best_val_metrics = {}
        no_improve       = 0

        for epoch in range(args.epochs):
            t_ep = time.time()
            if accelerator.is_main_process:
                print(f"\n──── Fold {fold_i} · Epoch {epoch} ────", flush=True)

            # Train
            train_metrics, train_conf_mat, train_loss, _ = run_epoch(
                model, train_loader, device, accelerator,
                criterion, optimizer, task=task, mode="train",
                precomputed_embeddings=precomputed,
            )
            if accelerator.is_main_process:
                log_epoch_metrics(epoch, "Train", task, train_metrics, train_loss, log_path)
                save_metrics_to_csv(
                    train_metrics, train_loss, epoch=epoch, task=task,
                    filepath=os.path.join(fold_results_path, "train_metrics.csv"),
                )
                print("\n --- Matriz de confusión en Train ---\n", flush=True)
                print(train_conf_mat)
                with open(log_path, "a") as f:
                    f.write("\n Matriz de confusión en Train:\n")
                    f.write(train_conf_mat.to_string())
                    f.write("\n\n")

            # Val
            val_metrics, val_conf_mat, val_loss, _ = run_epoch(
                model, val_loader, device, accelerator,
                criterion, None, task=task, mode="val",
                precomputed_embeddings=precomputed,
            )
            if accelerator.is_main_process:
                log_epoch_metrics(epoch, "Val", task, val_metrics, val_loss, log_path)
                save_metrics_to_csv(
                    val_metrics, val_loss, epoch=epoch, task=task,
                    filepath=os.path.join(fold_results_path, "val_metrics.csv"),
                )
                print("\n --- Matriz de confusión en Validación ---\n", flush=True)
                print(val_conf_mat)
                with open(log_path, "a") as f:
                    f.write("\n --- Matriz de confusión en Validación ---\n")
                    f.write(val_conf_mat.to_string())
                    f.write("\n\n")

            accelerator.wait_for_everyone()
            val_loss_tensor = torch.tensor(val_loss, device=device)
            broadcast(val_loss_tensor, from_process=0)
            val_loss_sync = val_loss_tensor.item()
            scheduler.step(val_loss_sync)

            # Último checkpoint
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if args.save_models:
                    accelerator.save_state(last_model_path)
                elapsed = time.time() - t_ep
                print(f"Época {epoch} — {elapsed:.1f}s", flush=True)
                with open(log_path, "a") as f:
                    f.write(f"Época {epoch} — {elapsed:.1f}s\n")

            # Best model / Early stopping
            should_stop = torch.tensor(0, device=device)
            if accelerator.is_main_process:
                if val_loss_sync < best_val_loss:
                    best_val_loss    = val_loss_sync
                    best_val_epoch   = epoch
                    best_val_metrics = val_metrics      # guardar métricas de la mejor época
                    no_improve       = 0
                    msg = f"  → Mejor modelo guardado (val_loss={best_val_loss:.4f}) ======================================================"
                    print(msg, flush=True)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
                    if args.save_models:
                        accelerator.save_state(best_model_path)
                        
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        print("Early stopping activado.", flush=True)
                        should_stop.fill_(1)

            broadcast(should_stop, from_process=0)
            if should_stop.item() == 1:
                break

        # ── Guardar best_eval.csv con métricas de la mejor época de validación ─
        if accelerator.is_main_process:
            best_eval_path = os.path.join(fold_results_path, "best_eval.csv")
            save_metrics_to_csv(
                best_val_metrics, best_val_loss,
                epoch=best_val_epoch, task=task,
                filepath=best_eval_path,
            )
            print(
                f"\n  best_eval.csv guardado en: {best_eval_path}\n"
                f"  (mejor época de validación: {best_val_epoch}, "
                f"val_loss={best_val_loss:.4f})",
                flush=True,
            )

        # Liberar modelo y optimizador para el siguiente fold
        accelerator.free_memory()
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    set_multiple_seed(args.seed)

    accelerator = Accelerator()
    device      = accelerator.device
    print(f"Dispositivo: {device}", flush=True)

    # ── PASO 1: split inicial dev / test ──────────────────────────────────────
    dev_df, test_df = make_initial_data_split(args)

    if not args.run_all_configs:
        # ── Modo single: una tarea, un encoder, un MIL, un set de parámetros ────────────────────────
        df_task, fold_indices, kfold_report_str, base_datasets = prepare_task_data(
            args, dev_df, task=args.task, patch_model=args.patch_model,
        )
        run_kfolds(
            df_task              = df_task,
            fold_indices         = fold_indices,
            base_datasets        = base_datasets,
            accelerator          = accelerator,
            device               = device,
            args                 = args,
            kfold_report_str     = kfold_report_str,
            task                 = args.task,
            patch_model          = args.patch_model,
            mil_aggregation_mode = args.mil_aggregation_mode,
        )

    else:
        # ── Modo multi-configuración: todas las combinaciones tarea × encoder × MIL x parámetros ─
        # Orden de iteración: tarea → encoder → MIL → parámetros.
        encoder_mil_combos = list(itertools.product(ALL_ENCODERS, ALL_MIL_MODES))
        lr_wd_combos       = list(itertools.product(ALL_LEARNING_RATES, ALL_WEIGHT_DECAYS))
        total = len(ALL_TASKS) * len(encoder_mil_combos) * len(lr_wd_combos)
        print(
            f"\nModo run_all_configs: {len(ALL_TASKS)} tarea(s) × "
            f"{len(encoder_mil_combos)} combinaciones encoder×MIL × "
            f"{len(lr_wd_combos)} combinaciones lr×wd = "
            f"{total} experimentos",
            flush=True,
        )

        precomputed = args.precomputed_embeddings_folder is not None
        exp_counter = 0

        for task in ALL_TASKS:

            if precomputed:
                # ── Embeddings precomputados: recargar por tarea × encoder ───
                for patch_model in ALL_ENCODERS:
                    df_task, fold_indices, kfold_report_str, base_datasets = prepare_task_data(
                        args, dev_df, task=task, patch_model=patch_model,
                    )

                    for mil_mode in ALL_MIL_MODES:
                        for lr, wd in lr_wd_combos:
                            exp_counter += 1
                            print(
                                f"\n{'='*60}\n"
                                f"Experimento {exp_counter}/{total}: "
                                f"task={task}  encoder={patch_model}  MIL={mil_mode}  "
                                f"lr={lr}  wd={wd}\n"
                                f"{'='*60}",
                                flush=True,
                            )

                            exp_args                      = copy.deepcopy(args)
                            exp_args.task                 = task
                            exp_args.patch_model          = patch_model
                            exp_args.mil_aggregation_mode = mil_mode
                            exp_args.learning_rate        = lr
                            exp_args.weight_decay         = wd

                            run_kfolds(
                                df_task              = df_task,
                                fold_indices         = fold_indices,
                                base_datasets        = base_datasets,
                                accelerator          = accelerator,
                                device               = device,
                                args                 = exp_args,
                                kfold_report_str     = kfold_report_str,
                                task                 = task,
                                patch_model          = patch_model,
                                mil_aggregation_mode = mil_mode,
                            )

                            torch.cuda.empty_cache()
                            gc.collect()

            else:
                # ── Modo imagen: recargar solo cuando cambia la tarea ─────────
                df_task, fold_indices, kfold_report_str, base_datasets = prepare_task_data(
                    args, dev_df, task=task, patch_model=ALL_ENCODERS[0],
                )

                for patch_model, mil_mode in encoder_mil_combos:
                    for lr, wd in lr_wd_combos:
                        exp_counter += 1
                        print(
                            f"\n{'='*60}\n"
                            f"Experimento {exp_counter}/{total}: "
                            f"task={task}  encoder={patch_model}  MIL={mil_mode}  "
                            f"lr={lr}  wd={wd}\n"
                            f"{'='*60}",
                            flush=True,
                        )

                        exp_args                      = copy.deepcopy(args)
                        exp_args.task                 = task
                        exp_args.patch_model          = patch_model
                        exp_args.mil_aggregation_mode = mil_mode
                        exp_args.learning_rate        = lr
                        exp_args.weight_decay         = wd

                        run_kfolds(
                            df_task              = df_task,
                            fold_indices         = fold_indices,
                            base_datasets        = base_datasets,
                            accelerator          = accelerator,
                            device               = device,
                            args                 = exp_args,
                            kfold_report_str     = kfold_report_str,
                            task                 = task,
                            patch_model          = patch_model,
                            mil_aggregation_mode = mil_mode,
                        )

                        torch.cuda.empty_cache()
                        gc.collect()

        print(
            f"\n{'='*60}\n"
            f"Validación cruzada completada para {exp_counter} experimentos.\n"
            f"Fase de re-entrenamiento sobre dev completo pendiente de implementar.\n"
            f"{'='*60}",
            flush=True,
        )
