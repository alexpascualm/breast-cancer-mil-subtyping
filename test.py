"""
test.py
-------
Entrenamiento definitivo + inferencia en test set.

Flujo por cada tarea:
  1. Lee los resultados de experimentos de la carpeta indicada y selecciona
     la mejor combinación de hiperparámetros por tarea (según `--rank_metric`).
  2. Entrena un modelo final usando el conjunto de development (dev_df) completo.
     durante tantas épocas como la media de las mejores épocas de los folds del experimento ganador.
  3. Infiere sobre el test set y guarda métricas en CSV.
  4. Guarda (opcionalmente) el modelo definitivo en disco.

Estructura de salida:
  <test_results_path>/
    <task>/
      train_metrics.csv   ← métricas de entrenamiento época a época
      test_metrics.csv    ← métricas sobre el test set
      test_log.txt        ← log completo
      [final_model/]      ← si --save_models

Uso mínimo:
  python test.py --source_results_path /workspace/results/my_run

Uso completo de ejemplo:
  python test.py \\
    --source_results_path /workspace/results/my_run \\
    --rank_metric balanced_accuracy \\
    --tasks er pr her2 ki67 molecular_subtype \\
    --save_models \\
    --seed 98
"""

import argparse
import os
import time
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from accelerate import Accelerator
from accelerate.utils import broadcast
from huggingface_hub import login

import pandas as pd

from data_load import (
    Patches_WSI_Dataset,
    Embeddings_WSI_Dataset,
    collate_fn,
)
from utils.train_utils import (
    set_multiple_seed,
    get_model,
    compute_class_weights,
    run_epoch,
)
from utils.metrics_utils import log_epoch_metrics, save_metrics_to_csv
from utils.test_utils import select_best_config, build_final_args
from config import (
    bcnb_images_folder_path,
    bcnb_labels_df_path,
    bcnb_embeddings_path,
)


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento definitivo sobre dev completo + inferencia en test")

    # ── Origen de resultados de búsqueda de hiperparámetros ───────────────────
    parser.add_argument("--source_results_path", type=str, required=True,
        help=("Ruta a la carpeta de resultados de búsqueda de hiperparámetros. "
            "Se buscará en <source_results_path>/experiments/ para seleccionar "
            "la mejor configuración por tarea."))

    # ── Destino de los resultados finales ─────────────────────────────────────
    parser.add_argument("--test_results_path", type=str, default=None,
        help=("Carpeta raíz donde se guardarán los resultados finales. "
            "Por defecto: <source_results_path>/test_results"))

    # ── Selección de tareas ───────────────────────────────────────────────────
    parser.add_argument("--tasks", type=str, nargs="+",
        default=["er", "pr", "her2", "ki67", "molecular_subtype"],
        choices=["er", "pr", "her2", "ki67", "molecular_subtype"],
        help="Tareas para las que entrenar y evaluar el modelo definitivo.")

    # ── Métrica de ranking para selección de hiperparámetros ─────────────────
    parser.add_argument("--rank_metric", type=str, default="balanced_accuracy",
        help=("Métrica usada para seleccionar la mejor configuración de hiperparámetros por tarea (leída y promediada de los best_eval.csv de cada fold de cada experimento). "))

    parser.add_argument("--rank_ascending", action="store_true",
        help="Si se activa, menor valor de rank_metric es mejor (útil para 'loss').")

    # ── Datos ─────────────────────────────────────────────────────────────────
    parser.add_argument("--images_folder", type=str, default=bcnb_images_folder_path)
    parser.add_argument("--df_path", type=str, default=bcnb_labels_df_path)
    parser.add_argument("--slide_col", type=str, default="Patient ID")
    parser.add_argument("--precomputed_embeddings_folder", type=str, default=bcnb_embeddings_path,
                        help="Carpeta de embeddings precomputados. Si se especifica, se usarán en lugar de procesar imágenes con el encoder en cada época.")

    # ── Split dev/test ────────────────────────────────────────────────────────
    # test.py CARGA los splits previamente guardados por train.py.
    parser.add_argument("--dev_df_path", type=str, default=None,
                        help=("Ruta al CSV dev_df.csv generado por train.py. "
                        "Si no se especifica se busca en <source_results_path>/dev_df.csv."))

    parser.add_argument("--test_df_path", type=str, default=None,
                        help=("Ruta al CSV test_df.csv generado por train.py. "
                        "Si no se especifica se busca en <source_results_path>/test_df.csv."))

    # ── Entrenamiento definitivo ──────────────────────────────────────────────
    parser.add_argument("--epochs_override", type=int, default=None,
        help=("Si se especifica, sobreescribe el número de épocas calculado automáticamente a partir de la media de mejores épocas de los folds."))

    parser.add_argument("--batch_size", type=int, default=None,
                        help="Sobreescribe el batch_size de la config ganadora.")

    parser.add_argument("--load_on_demand", action="store_true")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--save_models", action="store_true",
                        help="Guarda el modelo definitivo en disco.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token para modelos privados.")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE SPLITS DEV / TEST
# ──────────────────────────────────────────────────────────────────────────────

def load_splits(args) -> tuple:
    """
    Carga dev_df y test_df desde los CSV guardados por train.py.

    Si no se especifican rutas, busca por defecto en <source_results_path>/dev_df.csv y test_df.csv.

    Returns
    -------
    dev_df, test_df : pd.DataFrame
    """
    dev_path  = args.dev_df_path  or os.path.join(args.source_results_path, "dev_df.csv")
    test_path = args.test_df_path or os.path.join(args.source_results_path, "test_df.csv")

    if os.path.isfile(dev_path) and os.path.isfile(test_path):
        print(f"\nCargando splits existentes:\n  dev  → {dev_path}\n  test → {test_path}", flush=True)
        dev_df  = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)
        print(f"  dev  : {len(dev_df)} WSIs\n  test : {len(test_df)} WSIs", flush=True)
        return dev_df, test_df
    else:
        error_msg = (
            f"Splits dev/test no encontrados en:\n  dev  → {dev_path}\n  test → {test_path}\n"
            "Parando ejecución. Asegúrate de que train.py se haya ejecutado previamente y haya guardado los splits"
        )
        raise ValueError(error_msg)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DEFINITIVO + TEST
# ──────────────────────────────────────────────────────────────────────────────

def run_final_training_and_test(
    task: str,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args,             # Namespace con la config ganadora ya aplicada
    accelerator,
    device,
    task_results_path: str,
):
    """
    Entrena el modelo definitivo sobre todo dev_df y lo evalúa en test_df.

    Estructura de salida en task_results_path/:
        train_metrics.csv
        test_metrics.csv
        test_log.txt
        [final_model/]   ← si args.save_models

    Args
    ----
    task               : str — tarea activa.
    dev_df             : pd.DataFrame — conjunto de development completo.
    test_df            : pd.DataFrame — conjunto de test.
    args               : Namespace — configuración definitiva para esta tarea.
    accelerator        : Accelerator.
    device             : torch.device.
    task_results_path  : str — carpeta de salida para esta tarea.
    """
    precomputed = args.precomputed_embeddings_folder is not None

    os.makedirs(task_results_path, exist_ok=True)
    log_path         = os.path.join(task_results_path, "test_log.txt")
    train_csv_path   = os.path.join(task_results_path, "train_metrics.csv")
    test_csv_path    = os.path.join(task_results_path, "test_metrics.csv")
    final_model_path = os.path.join(task_results_path, "final_model")

    def _log(msg: str):
        if accelerator.is_main_process:
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    _log(f"\n{'='*60}")
    _log(f"  TAREA: {task}")
    _log(f"{'='*60}")
    _log(f"  Encoder       : {args.patch_model}")
    _log(f"  MIL           : {args.mil_aggregation_mode}")
    _log(f"  lr / wd       : {args.learning_rate} / {args.weight_decay}")
    _log(f"  optimizer     : {args.optimizer}")
    _log(f"  weighted_loss : {args.weighted_loss}")
    _log(f"  epochs        : {args.epochs}")
    _log(f"  batch_size    : {args.batch_size}")
    _log(f"  precomputed   : {precomputed}")

    # ── Filtrar muestras válidas para la tarea ────────────────────────────────
    df_dev_task  = dev_df[dev_df[task] != -1].reset_index(drop=True)
    df_test_task = test_df[test_df[task] != -1].reset_index(drop=True)

    _log(f"\n  dev  (tarea válida): {len(df_dev_task)} WSIs")
    _log(f"  test (tarea válida): {len(df_test_task)} WSIs")

    # ── Datasets ──────────────────────────────────────────────────────────────
    emb_folder = (os.path.join(args.precomputed_embeddings_folder, args.patch_model) if precomputed else None)

    all_dev_indices  = np.arange(len(df_dev_task))
    all_test_indices = np.arange(len(df_test_task))

    if emb_folder:
        ds_train = Embeddings_WSI_Dataset(
            df_dev_task, all_dev_indices, task=task,
            embeddings_folder=emb_folder,
            load_on_demand=args.load_on_demand,
        )
        ds_test = Embeddings_WSI_Dataset(
            df_test_task, all_test_indices, task=task,
            embeddings_folder=emb_folder,
            load_on_demand=args.load_on_demand,
        )
    else:
        ds_train = Patches_WSI_Dataset(
            df_dev_task, all_dev_indices, task=task,
            patch_resize_size=getattr(args, "patch_resize_size", 224),
            load_on_demand=args.load_on_demand,
        )
        ds_test = Patches_WSI_Dataset(
            df_test_task, all_test_indices, task=task,
            patch_resize_size=getattr(args, "patch_resize_size", 224),
            load_on_demand=args.load_on_demand,
        )

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    num_classes = ds_train.num_classes
    _log(f"\n  num_classes: {num_classes}")

    model = get_model(
        patch_encoder_type=args.patch_model,
        mil_aggregation_mode=args.mil_aggregation_mode,
        unfreeze_backbone=getattr(args, "unfreeze_patch_model_backbone", False),
        num_classes=num_classes,
        skip_encoder=precomputed,
    ).to(device)

    # ── Pérdida ───────────────────────────────────────────────────────────────
    if args.weighted_loss:
        train_labels = df_dev_task[task].tolist()
        weights      = compute_class_weights(train_labels, num_classes).to(device)
        criterion    = nn.CrossEntropyLoss(weight=weights)
        _log("  Pérdida con pesos por clase aplicada.")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizador ───────────────────────────────────────────────
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=getattr(args, "momentum", 0.9),
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )


    # ── Accelerate ────────────────────────────────────────────────────────────
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader) 

    # ── Bucle de entrenamiento definitivo ─────────────────────────────────────
    _log(f"\n  Iniciando entrenamiento definitivo: {args.epochs} época(s) ...\n")

    for epoch in range(args.epochs):
        t_ep = time.time()
        _log(f"\n──── Epoch {epoch} ────")

        train_metrics, train_conf_mat, train_loss, _ = run_epoch(
            model, train_loader, device, accelerator,
            criterion, optimizer, task=task, mode="train",
            precomputed_embeddings=precomputed,
        )

        if accelerator.is_main_process:
            log_epoch_metrics(epoch, "Train", task, train_metrics, train_loss, log_path)
            save_metrics_to_csv(
                train_metrics, train_loss, epoch=epoch, task=task,
                filepath=train_csv_path,
            )
            _log(f"\n --- Matriz de confusión en Train ---")
            _log(train_conf_mat.to_string())

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elapsed = time.time() - t_ep
            _log(f"Época {epoch} — {elapsed:.1f}s")

    # ── Guardar modelo definitivo ─────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.save_models:
        accelerator.save_state(final_model_path)
        _log(f"\n  Modelo definitivo guardado en: {final_model_path}")

    # ── Inferencia sobre test set ─────────────────────────────────────────────
    _log(f"\n{'─'*60}")
    _log(f"  Inferencia sobre el TEST SET — tarea: {task}")
    _log(f"{'─'*60}")

    test_metrics, test_conf_mat, test_loss, _ = run_epoch(
        model, test_loader, device, accelerator,
        criterion, None, task=task, mode="test",
        precomputed_embeddings=precomputed,
    )

    if accelerator.is_main_process:
        log_epoch_metrics(0, "Test", task, test_metrics, test_loss, log_path)
        save_metrics_to_csv(
            test_metrics, test_loss, epoch=0, task=task,
            filepath=test_csv_path,
        )
        _log(f"\n --- Matriz de confusión en Test ---")
        _log(test_conf_mat.to_string())
        _log(f"\n  test_metrics.csv guardado en: {test_csv_path}")

    # ── Liberar memoria ───────────────────────────────────────────────────────
    accelerator.free_memory()
    del model, optimizer, train_loader, test_loader
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

    # ── Resolver ruta de destino ──────────────────────────────────────────────
    if args.test_results_path is None:
        args.test_results_path = os.path.join(args.source_results_path, "test_results")

    os.makedirs(args.test_results_path, exist_ok=True)

    # ── Ruta de experimentos ──────────────────────────────────────────────────
    experiments_path = os.path.join(args.source_results_path, "experiments")

    # ── Cargar splits dev / test ──────────────────────────────────────────────
    dev_df, test_df = load_splits(args)

    # ── Seleccionar la mejor configuración por tarea ──────────────────────────
    print(
        f"\n{'='*60}\n"
        f"Seleccionando mejor configuración por tarea\n"
        f"  experiments_path : {experiments_path}\n"
        f"  rank_metric      : {args.rank_metric}\n"
        f"  ascending        : {args.rank_ascending}\n"
        f"{'='*60}",
        flush=True,
    )

    best_per_task = select_best_config(
        experiments_path=experiments_path,
        tasks=args.tasks,
        rank_metric=args.rank_metric,
        ascending=args.rank_ascending,
        verbose=True,
    )

    if not best_per_task:
        raise RuntimeError(
            "No se encontró ninguna configuración ganadora. "
            "Revisa --source_results_path y que los experimentos tengan best_eval.csv."
        )

    # ── Entrenamiento definitivo + test para cada tarea ───────────────────────
    for task in args.tasks:
        if task not in best_per_task:
            print(f"\n[WARN] Tarea '{task}' sin configuración ganadora — se omite.", flush=True)
            continue

        best = best_per_task[task]

        # Construir args definitivos para esta tarea
        final_args = build_final_args(base_args=args, task=task, best=best)

        # Sobreescribir parámetros si el usuario lo pidió explícitamente
        if args.epochs_override is not None:
            final_args.epochs = args.epochs_override
        if args.batch_size is not None:
            final_args.batch_size = args.batch_size

        # Atributos necesarios que pueden no estar en base_args
        if not hasattr(final_args, "patch_resize_size"):
            final_args.patch_resize_size = 224
        if not hasattr(final_args, "unfreeze_patch_model_backbone"):
            final_args.unfreeze_patch_model_backbone = best["meta"]["freeze"] == "unfrozen"

        task_results_path = os.path.join(args.test_results_path, task)

        print(
            f"\n{'='*60}\n"
            f"  Entrenando modelo definitivo para '{task}'\n"
            f"  Épocas : {final_args.epochs}\n"
            f"  Salida : {task_results_path}\n"
            f"{'='*60}",
            flush=True,
        )

        run_final_training_and_test(
            task=task,
            dev_df=dev_df,
            test_df=test_df,
            args=final_args,
            accelerator=accelerator,
            device=device,
            task_results_path=task_results_path,
        )

    print(
        f"\n{'='*60}\n"
        f"  test.py completado.\n"
        f"  Resultados en: {args.test_results_path}\n"
        f"{'='*60}",
        flush=True,
    )