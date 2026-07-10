"""
test.py
-------
Entrenamiento definitivo con dev set + inferencia en test set.

Flujo por cada tarea:
  1. Lee los experimentos de <source_results_path>/experiments/ y analiza las configuraciones probadas por tarea ordenadas por --rank_metric.
  2. Guarda best_config.json con la configuración seleccionada, la mejor en general, la mejor en cada encoder x mil y el ranking global.
  3. Entrena el modelo final (con la mejor configuración o el mejor encoder x MIL indicado) sobre dev completo durante:
         epochs = round(best_epoch_mean) + 1
        (best_epoch es el índice 0-based de la mejor época, media entre folds).
  4. Infiere sobre el test set y guarda métricas en CSV.

Estructura de salida:
  <test_results_path>/
    <task>/
      best_config.json
      train_metrics.csv
      test_metrics.csv
      test_log.txt
      [final_model/]   ← si --save_models

Uso mínimo:
  python test.py --source_results_path /workspace/results/my_run

Selección manual de encoder x MIL por tarea:
  python test.py \\
    --source_results_path /workspace/results/my_run \\
    --encoder_mil_override er:gigapath_abMIL-bag-level \\
    --encoder_mil_override her2:gigapath_transMIL
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
from utils.test_utils import (
    select_best_config,
    build_final_args,
    save_best_config_json,
    DEFAULT_METRIC_COLS,
)
from config import (
    bcnb_images_folder_path,
    bcnb_labels_df_path,
    bcnb_embeddings_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento definitivo sobre dev completo + inferencia en test"
    )

    parser.add_argument("--source_results_path", type=str, required=True,
        help="Ruta a la carpeta de resultados de búsqueda de hiperparámetros.")

    parser.add_argument("--test_results_path", type=str, default=None,
        help="Carpeta de salida. Por defecto: <source_results_path>/test_results")

    parser.add_argument("--tasks", type=str, nargs="+",
        default=["er", "pr", "her2", "ki67", "molecular_subtype", "molecular_subtype_from_biomarkers"],
        choices=["er", "pr", "her2", "ki67", "molecular_subtype", "molecular_subtype_from_biomarkers"])

    parser.add_argument("--rank_metric", type=str, default="balanced_accuracy")

    parser.add_argument("--rank_ascending", action="store_true",
        help="Menor valor de rank_metric es mejor (útil para 'loss').")

    parser.add_argument(
        "--encoder_mil_override",
        type=str, action="append", default=[],
        metavar="TASK:MODEL_MIL",
        help=(
            "Fuerza un encoder x MIL concreto para una tarea. "
            "Formato: TASK:MODEL_MIL  (ej. 'er:gigapath_abMIL-bag-level'). "
            "Se puede repetir para varias tareas."
        ),
    )

    parser.add_argument("--images_folder", type=str, default=bcnb_images_folder_path)
    parser.add_argument("--df_path", type=str, default=bcnb_labels_df_path)
    parser.add_argument("--slide_col", type=str, default="Patient ID")
    parser.add_argument("--precomputed_embeddings_folder", type=str,
                        default=bcnb_embeddings_path)

    parser.add_argument("--dev_df_path", type=str, default=None,
        help="CSV dev. Por defecto: <source_results_path>/dev_df.csv")
    parser.add_argument("--test_df_path", type=str, default=None,
        help="CSV test. Por defecto: <source_results_path>/test_df.csv")

    parser.add_argument("--epochs_override", type=int, default=None,
        help="Sobreescribe el número de épocas de entrenamiento sobre dev set. Por defecto se calcula como round(best_epoch_mean) + 1.")

    parser.add_argument("--batch_size", type=int, default=None,
        help="Sobreescribe el batch_size de la config ganadora.")

    parser.add_argument("--load_on_demand", action="store_true")
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    return parser.parse_args()


def _parse_encoder_mil_overrides(raw_list: list) -> dict:
    """Convierte ["TASK:MODEL_MIL", ...] → {task: "MODEL_MIL"}."""
    overrides = {}
    for raw in raw_list:
        if ":" not in raw or "_" not in raw.split(":", 1)[1]:
            raise ValueError(
                f"Formato inválido: '{raw}'. Se esperaba 'TASK:MODEL_MIL', "
                "ej. 'er:gigapath_abMIL-bag-level'."
            )
        task, key = raw.split(":", 1)
        overrides[task.strip()] = key.strip()
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(args) -> tuple:
    """Carga dev_df y test_df desde los CSV guardados por train.py."""
    dev_path  = args.dev_df_path  or os.path.join(args.source_results_path, "dev_df.csv")
    test_path = args.test_df_path or os.path.join(args.source_results_path, "test_df.csv")

    if not (os.path.isfile(dev_path) and os.path.isfile(test_path)):
        raise ValueError(
            f"Splits no encontrados:\n  dev  → {dev_path}\n  test → {test_path}\n"
            "Asegúrate de que train.py se haya ejecutado previamente."
        )

    dev_df  = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    print(f"Splits cargados — dev: {len(dev_df)} WSIs  |  test: {len(test_df)} WSIs")
    return dev_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DEFINITIVO + TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_final_training_and_test(
    task: str,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args,
    accelerator,
    device,
    task_results_path: str,
):
    """
    Entrena el modelo definitivo sobre todo dev_df y lo evalúa en test_df.
    """
    precomputed = args.precomputed_embeddings_folder is not None
    os.makedirs(task_results_path, exist_ok=True)

    log_path       = os.path.join(task_results_path, "test_log.txt")
    train_csv_path = os.path.join(task_results_path, "train_metrics.csv")
    test_csv_path  = os.path.join(task_results_path, "test_metrics.csv")
    model_path     = os.path.join(task_results_path, "final_model")

    def _log(msg: str):
        if accelerator.is_main_process:
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

    _log(f"\n{'='*60}\n  TAREA: {task}\n{'='*60}")
    _log(f"  Encoder    : {args.patch_model}")
    _log(f"  MIL        : {args.mil_aggregation_mode}")
    _log(f"  lr / wd    : {args.learning_rate} / {args.weight_decay}")
    _log(f"  optimizer  : {args.optimizer}")
    _log(f"  epochs     : {args.epochs}")
    _log(f"  batch_size : {args.batch_size}")
    _log(f"  precomputed embeddings: {precomputed}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    df_dev_task  = dev_df[dev_df[task] != -1].reset_index(drop=True)
    df_test_task = test_df[test_df[task] != -1].reset_index(drop=True)
    _log(f"\n  dev (válidos): {len(df_dev_task)}  |  test (válidos): {len(df_test_task)}")

    emb_folder = (os.path.join(args.precomputed_embeddings_folder, args.patch_model) if precomputed else None)

    if emb_folder:
        ds_train = Embeddings_WSI_Dataset(
            df_dev_task, np.arange(len(df_dev_task)), task=task,
            embeddings_folder=emb_folder, load_on_demand=args.load_on_demand,
        )
        ds_test = Embeddings_WSI_Dataset(
            df_test_task, np.arange(len(df_test_task)), task=task,
            embeddings_folder=emb_folder, load_on_demand=args.load_on_demand,
        )
    else:
        ds_train = Patches_WSI_Dataset(
            df_dev_task, np.arange(len(df_dev_task)), task=task,
            patch_resize_size=getattr(args, "patch_resize_size", 224),
            load_on_demand=args.load_on_demand,
        )
        ds_test = Patches_WSI_Dataset(
            df_test_task, np.arange(len(df_test_task)), task=task,
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

    weights   = compute_class_weights(df_dev_task[task].tolist(), num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    _log(f"\n  Entrenamiento definitivo: {args.epochs} época(s)\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        _log(f"\n──── Epoch {epoch} ────")

        train_metrics, train_conf_mat, train_loss = run_epoch(
            model, train_loader, device, accelerator,
            criterion, optimizer, task=task, mode="train",
            precomputed_embeddings=precomputed,
        )

        if accelerator.is_main_process:
            log_epoch_metrics(epoch, "Train", task, train_metrics, train_loss, log_path)
            save_metrics_to_csv(train_metrics, train_loss, epoch=epoch, task=task,
                                filepath=train_csv_path)
            _log(train_conf_mat.to_string())
            _log(f"Época {epoch} — {time.time() - t0:.1f}s")

        accelerator.wait_for_everyone()

    # ── Guardar modelo ────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.save_models:
        unwrapped_model = accelerator.unwrap_model(model)
        os.makedirs(model_path, exist_ok=True)
        torch.save(
            unwrapped_model.state_dict(),
            os.path.join(model_path, "model.pt")
        )
        _log(f"\n  Modelo guardado en: {model_path}")

    # ── Inferencia test ───────────────────────────────────────────────────────
    _log(f"\n{'─'*60}\n  TEST SET — tarea: {task}\n{'─'*60}")

    test_metrics, test_conf_mat, test_loss = run_epoch(
        model, test_loader, device, accelerator,
        criterion, None, task=task, mode="test",
        precomputed_embeddings=precomputed,
    )

    if accelerator.is_main_process:
        log_epoch_metrics(0, "Test", task, test_metrics, test_loss, log_path)
        save_metrics_to_csv(test_metrics, test_loss, epoch=0, task=task,
                            filepath=test_csv_path)
        _log(test_conf_mat.to_string())
        _log(f"\n  test_metrics.csv guardado en: {test_csv_path}")

    # ── Liberar memoria ───────────────────────────────────────────────────────
    accelerator.free_memory()
    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    set_multiple_seed(args.seed)

    enc_mil_overrides = _parse_encoder_mil_overrides(args.encoder_mil_override)
    if enc_mil_overrides:
        print("Overrides encoder x MIL:", enc_mil_overrides)

    accelerator = Accelerator()
    device = accelerator.device
    print(f"Dispositivo: {device}")

    if args.test_results_path is None:
        args.test_results_path = os.path.join(args.source_results_path, "test_results")
    os.makedirs(args.test_results_path, exist_ok=True)

    experiments_path = os.path.join(args.source_results_path, "experiments")
    dev_df, test_df = load_splits(args)

    print(f"\nSeleccionando mejor config  |  rank_metric={args.rank_metric}  "
          f"|  ascending={args.rank_ascending}")

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
            "Revisa --source_results_path y que los folds tengan best_eval.csv."
        )

    for task in args.tasks:
        if task not in best_per_task:
            print(f"[WARN] Tarea '{task}' sin configuración ganadora — se omite.")
            continue

        best        = best_per_task[task]
        enc_mil_key = enc_mil_overrides.get(task, None)

        final_args = build_final_args(
            base_args=args, task=task, best=best,
            encoder_mil_override=enc_mil_key,
        )

        if args.epochs_override is not None:
            final_args.epochs = args.epochs_override
        if args.batch_size is not None:
            final_args.batch_size = args.batch_size

        if not hasattr(final_args, "patch_resize_size"):
            final_args.patch_resize_size = 224
        if not hasattr(final_args, "unfreeze_patch_model_backbone"):
            final_args.unfreeze_patch_model_backbone = (
                best["meta"]["freeze"] == "unfrozen"
            )

        task_results_path = os.path.join(args.test_results_path, task)

        if accelerator.is_main_process:
            json_path = save_best_config_json(
                best_per_task=best_per_task,
                task_results_path=task_results_path,
                task=task,
                rank_metric=args.rank_metric,
                selected_enc_mil_key=enc_mil_key,
                metric_cols=DEFAULT_METRIC_COLS,
                ascending=args.rank_ascending,
            )
            print(f"\n  best_config.json → {json_path}")

        print(
            f"\n{'='*60}\n  Entrenando modelo definitivo: {task}\n"
            f"  {final_args.patch_model} / {final_args.mil_aggregation_mode}  "
            f"|  epochs={final_args.epochs}\n{'='*60}"
        )

        run_final_training_and_test(
            task=task, dev_df=dev_df, test_df=test_df,
            args=final_args, accelerator=accelerator, device=device,
            task_results_path=task_results_path,
        )

    print(f"\ntest.py completado. Resultados en: {args.test_results_path}")
