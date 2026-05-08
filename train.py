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
)
from utils.metrics_utils import log_epoch_metrics, save_metrics_to_csv
from config import bcnb_images_folder_path, bcnb_labels_df_path, results_folder_path


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
                        choices=["max", "mean", "abMIL-instance-level", "abMIL-bag-level", "transMIL"])
    parser.add_argument("--patch_resize_size", type=int, default=224,
                        help="Dimensiones de reescalado de los parches")
    parser.add_argument("--precomputed_embeddings_folder", type=str, default=None,
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
    parser.add_argument("--kfold", type=int, default=5,
                        help="Número de folds para cross-validation (mínimo 3). "
                             "En cada iteración i se usa el fold i como test y "
                             "el fold (i+1)%%k como validación.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--load_on_demand",         action="store_true")
    parser.add_argument("--small_dataset_for_test", action="store_true")
    parser.add_argument("--seed",                   type=int, default=98)
    parser.add_argument("--experiment_nickname",    type=str, default="")
    parser.add_argument("--save_models",            action="store_true",
                        help="Si se activa, guarda checkpoints best_model y "
                             "last_model en disco. El best_model se usa para "
                             "la inferencia de test y se elimina al final si "
                             "save_models=False.")

    parser.add_argument("--resume_training",       action="store_true")
    parser.add_argument("--pretrained_model_path", type=str, default=None)

    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token para descargar modelos privados")

    # ── Modo multi-configuración ───────────────────────────────────────────────
    parser.add_argument("--run_all_configs", action="store_true",
                        help="Ejecuta automáticamente todas las combinaciones "
                             "encoder x MIL para cada tarea en ALL_TASKS. "
                             "Los folds y datasets se calculan una sola vez "
                             "por tarea y se reutilizan para cada combinación.")


    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIONES PARA --run_all_configs
# Edita estas listas para añadir o quitar tareas, encoders y modos MIL.
# ──────────────────────────────────────────────────────────────────────────────

ALL_TASKS = [
    "molecular_subtype",
    "er",
    "pr",
    "her2",
    "ki67",
]

ALL_ENCODERS = [
    "uni",
    "dino",
]

ALL_MIL_MODES = [
    "abMIL-instance-level",
    "abMIL-bag-level",
    "transMIL",
]


# ──────────────────────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATOS POR TAREA
# ──────────────────────────────────────────────────────────────────────────────

def prepare_task_data(args, task: str, patch_model: str = None):
    """
    Crea y filtra el DataFrame maestro para `task`, genera los k folds base y carga un dataset por fold.

    Comportamiento según el modo de ejecución:

    - Modo normal (sin embeddings precomputados):
        Los datasets pueden reutilizarse entre distintos encoders porque los datos contienen imágenes y el encoder se aplica dinámicamente durante el forward.

    - Modo embeddings precomputados:
        Los datasets dependen del encoder activo porque cada encoder produce embeddings con distinta dimensionalidad (`embed_dim`). En este caso, los datasets deben reconstruirse cada vez que cambia el encoder para apuntar al subdirectorio correcto: precomputed_embeddings_folder/<encoder>/

    Los folds (`fold_indices`) siguen siendo reutilizables porque dependen únicamente de la tarea y del DataFrame filtrado.

    Args:
        args : Namespace con los argumentos de línea de comandos.
        task : str
            Nombre de la tarea (columna de etiquetas).
        patch_model : str
            Encoder asociado a los embeddings precomputados. Si es None, se usa args.patch_model.

    Returns:
        df_task          : pd.DataFrame filtrado para la tarea.
        base_datasets    : lista de k datasets, donde base_datasets[i]
                        contiene las WSI del fold i.
        fold_indices     : lista de k np.ndarray con los índices posicionales
                        de cada fold sobre df_task.
        kfold_report_str : str resumen de distribuciones por fold.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Preparando datos para tarea: '{task}'", flush=True)
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

    df_task = df[df[task] != -1].reset_index(drop=True)
    print(f"WSIs con etiqueta válida para '{task}': {len(df_task)}", flush=True)
    print(f"Distribución:\n{df_task[task].value_counts()}", flush=True)

    # ── K-Fold splits ─────────────────────────────────────────────────────────
    # folds      → k tuplas (train_idx, val_idx, test_idx) con índices ya combinados
    # fold_indices → k arrays base, uno por fold puro (sin combinar)
    _folds, fold_indices, kfold_report_str = make_kfold_splits(df_task, label_col=task, seed=args.seed, k=args.kfold)

    # ── k datasets base (uno por fold) ────────────────────────────────────────
    patch_model = patch_model if patch_model is not None else args.patch_model
    emb_folder = (
        os.path.join(args.precomputed_embeddings_folder, patch_model)
        if args.precomputed_embeddings_folder
        else None
    )

    DatasetClass = Embeddings_WSI_Dataset if emb_folder else Patches_WSI_Dataset

    def _make_base_dataset(idx):
        """Construye un dataset para los índices posicionales `idx`."""
        if emb_folder:
            return DatasetClass(
                df_task, idx, task=task,
                embeddings_folder=emb_folder,
                load_on_demand=args.load_on_demand)
        return DatasetClass(
            df_task, idx, task=task,
            patch_resize_size=args.patch_resize_size,
            load_on_demand=args.load_on_demand)

    t0 = time.time()
    print(f"Cargando {args.kfold} datasets base ...", flush=True)

    base_datasets = []
    for fold_i, idx in enumerate(fold_indices):
        ds = _make_base_dataset(idx)
        base_datasets.append(ds)
        print(f"  Fold base {fold_i}: {len(ds)} WSIs", flush=True)

    print(f"Tiempo de carga: {time.time() - t0:.1f}s", flush=True)

    return df_task, base_datasets, fold_indices, kfold_report_str


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTO K-FOLD
# Itera sobre los k folds y entrena/evalúa un modelo para cada uno.
# ──────────────────────────────────────────────────────────────────────────────

def run_kfold_experiment(
    df_task,
    base_datasets,
    fold_indices,
    accelerator,
    device,
    args,
    kfold_report_str,
    task: str,
    patch_model: str,
    mil_aggregation_mode: str,
):
    """
    Ejecuta el entrenamiento con k-fold cross-validation para una combinación
    concreta de (tarea, encoder, modo MIL).

    Los k datasets base (uno por fold) se ensamblan en memoria en cada
    iteración i usando ConcatDataset, sin releer datos de disco:
        - ds_test  ← base_datasets[i]
        - ds_val   ← base_datasets[(i+1) % k]
        - ds_train ← ConcatDataset(base_datasets[j] para j ≠ i, (i+1)%k)

    fold_indices se usa exclusivamente para calcular los pesos de clase del
    split train de cada iteración (requiere índices posicionales sobre df_task).

    Por cada fold:
        · Construye DataLoaders sobre los datasets ensamblados.
        · Instancia un modelo fresco.
        · Entrena con early stopping sobre val_loss.
        · Evalúa en test desde el mejor checkpoint de validación.
        · Guarda métricas y logs en fold_i/ dentro del dir del experimento.

    Args:
        df_task              : pd.DataFrame filtrado para la tarea.
        base_datasets        : lista de k datasets base (uno por fold).
        fold_indices         : lista de k np.ndarray con índices posicionales
                               de cada fold base sobre df_task.
        accelerator          : Accelerator de HuggingFace.
        device               : torch.device.
        args                 : Namespace con todos los argumentos.
        kfold_report_str     : str  resumen de splits (se escribe en el log).
        task                 : str  nombre de la tarea activa.
        patch_model          : str  nombre del encoder activo.
        mil_aggregation_mode : str  modo de agregación MIL activo.
    """
    precomputed = args.precomputed_embeddings_folder is not None
    k = len(base_datasets)

    # ── Carpeta raíz de este experimento (encoder × MIL × tarea) ─────────────
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
            f"_{args.kfold}fold{small}"
        )
        exp_results_path = os.path.join(args.results_path, run_name)
        os.makedirs(exp_results_path, exist_ok=True)
        print(f"\nResultados del experimento en: {exp_results_path}", flush=True)

    # ── Log global del experimento ────────────────────────────────────────────
    exp_log_path = os.path.join(exp_results_path, "experiment_log.txt")
    if accelerator.is_main_process:
        with open(exp_log_path, "a") as f:
            f.write(kfold_report_str + "\n\n")

    # ── Iterar sobre los k folds ──────────────────────────────────────────────
    for fold_i in range(k):
        val_fold_i  = (fold_i + 1) % k
        train_fold_js = [j for j in range(k) if j != fold_i and j != val_fold_i]

        # Ensamblar splits combinando datasets base en memoria (sin releer disco)
        ds_test  = base_datasets[fold_i]
        ds_val   = base_datasets[val_fold_i]
        ds_train = ConcatDataset([base_datasets[j] for j in train_fold_js])

        # Índices posicionales del train en df_task (necesarios para weighted_loss)
        train_global_indices = np.concatenate([fold_indices[j] for j in train_fold_js])
        if accelerator.is_main_process:
            print(f"\n{'─'*60}", flush=True)
            print(f"  FOLD {fold_i}/{k-1}  "
                  f"[task={task}  encoder={patch_model}  MIL={mil_aggregation_mode}]",
                  flush=True)
            print(f"{'─'*60}", flush=True)

        # Subcarpeta por fold
        fold_results_path = os.path.join(exp_results_path, f"fold_{fold_i}")
        if accelerator.is_main_process:
            os.makedirs(fold_results_path, exist_ok=True)

        log_path        = os.path.join(fold_results_path, "train_log.txt")
        best_model_path = os.path.join(fold_results_path, "best_model")
        last_model_path = os.path.join(fold_results_path, "last_model")

        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                # Extraer del reporte global la sección correspondiente a este fold y escribirla en el log de este fold
                start_fold = f"── Fold {fold_i} "
                end_fold   = f"── Fold {(fold_i + 1) % k} "

                total_part = kfold_report_str.split("── Fold ")[0]

                fold_part = kfold_report_str.split(start_fold)[1]

                if end_fold in fold_part:
                    fold_part = fold_part.split(end_fold)[0]

                f.write(f"=== Fold {fold_i} ===\n")
                f.write(total_part)
                f.write(start_fold + fold_part)

        # ── DataLoaders ───────────────────────────────────────────────────────
        train_loader = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        test_loader = DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False,
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
            patience=15, min_lr=1e-7, threshold=1e-4,
        )

        # ── Accelerate ────────────────────────────────────────────────────────
        (model, optimizer, scheduler, train_loader, val_loader, test_loader) = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader, test_loader)

        if args.resume_training:
            if not args.pretrained_model_path:
                raise ValueError("--resume_training requiere --pretrained_model_path")
            accelerator.load_state(args.pretrained_model_path)
            print(f"Reanudando desde: {args.pretrained_model_path}", flush=True)

        # ── Bucle de entrenamiento ────────────────────────────────────────────
        best_val_loss  = float("inf")
        best_val_epoch = -1
        no_improve     = 0

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
            if accelerator.is_main_process:
                scheduler.step(val_loss)

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
            if val_loss < best_val_loss:
                best_val_loss, no_improve, best_val_epoch = val_loss, 0, epoch
                if accelerator.is_main_process:
                    accelerator.save_state(best_model_path)
                    msg = f"  → Mejor modelo guardado (val_loss={best_val_loss:.4f})"
                    print(msg, flush=True)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    if accelerator.is_main_process:
                        print("Early stopping activado.", flush=True)
                    break

        # ── Test ──────────────────────────────────────────────────────────────
        if accelerator.is_main_process:
            print(f"\n──── Fold {fold_i} · TEST ────", flush=True)

        accelerator.wait_for_everyone()
        try:
            accelerator.load_state(best_model_path)
        except Exception as e:
            print(f"Error cargando checkpoint: {e}", flush=True)

        test_metrics, test_conf_mat, test_loss, _ = run_epoch(
            model, test_loader, device, accelerator,
            criterion, None, task=task, mode="test",
            precomputed_embeddings=precomputed,
        )
        if accelerator.is_main_process:
            log_epoch_metrics(
                best_val_epoch, "Test", task, test_metrics, test_loss, log_path
            )
            save_metrics_to_csv(
                test_metrics, test_loss, epoch=best_val_epoch, task=task,
                filepath=os.path.join(fold_results_path, "test_metrics.csv"),
            )
            print("\n --- Matriz de confusión en Test ---\n", flush=True)
            print(test_conf_mat)
            with open(log_path, "a") as f:
                f.write("\n --- Matriz de confusión en Test ---\n")
                f.write(test_conf_mat.to_string())
                f.write("\n\n")

        # Limpiar best_model si no se quieren conservar los modelos
        if not args.save_models:
            shutil.rmtree(best_model_path, ignore_errors=True)
            print(
                "save_models=False — best model checkpoint eliminado tras test.",
                flush=True,
            )

        # Liberar modelo y optimizador para el siguiente fold
        accelerator.free_memory()
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
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

    if not args.run_all_configs:
        # ── Modo single: una tarea, un encoder, un MIL ───────────────────────
        df_task, base_datasets, fold_indices, kfold_report_str = prepare_task_data(args, args.task, args.patch_model)
        run_kfold_experiment(
            df_task              = df_task,
            base_datasets        = base_datasets,
            fold_indices         = fold_indices,
            accelerator          = accelerator,
            device               = device,
            args                 = args,
            kfold_report_str     = kfold_report_str,
            task                 = args.task,
            patch_model          = args.patch_model,
            mil_aggregation_mode = args.mil_aggregation_mode,
        )

    else:
        # ── Modo multi-configuración: todas las combinaciones encoder × MIL ──
        encoder_mil_combos = list(itertools.product(ALL_ENCODERS, ALL_MIL_MODES))
        total = len(ALL_TASKS) * len(encoder_mil_combos)
        print(
            f"\nModo run_all_configs: {len(ALL_TASKS)} tarea(s) × "
            f"{len(encoder_mil_combos)} combinaciones encoder×MIL = "
            f"{total} experimentos",
            flush=True,
        )

        exp_counter = 0
        for task in ALL_TASKS:
            # Los folds dependen solo de la tarea, no del encoder
            cached_data_per_encoder = {}

            for patch_model, mil_mode in encoder_mil_combos:
                # ──────────────────────────────────────────────────────────────
                # Si usamos embeddings precomputados:
                #   cada encoder necesita recargar datasets porque embed_dim cambia
                #
                # Si NO usamos embeddings:
                #   podemos reutilizar datasets entre encoders para una misma tarea, porque el encoder se aplica dinámicamente en cada forward.
                # ──────────────────────────────────────────────────────────────
                cache_key = (patch_model if args.precomputed_embeddings_folder is not None else "shared")
                if cache_key not in cached_data_per_encoder:
                    df_task, base_datasets, fold_indices, kfold_report_str = prepare_task_data(args, task, patch_model=patch_model)
                    cached_data_per_encoder[cache_key] = (df_task, base_datasets, fold_indices, kfold_report_str)

                (df_task, base_datasets, fold_indices, kfold_report_str) = cached_data_per_encoder[cache_key]
       
                exp_counter += 1
                print(
                    f"\n{'='*60}\n"
                    f"Experimento {exp_counter}/{total}: "
                    f"task={task}  encoder={patch_model}  MIL={mil_mode}\n"
                    f"{'='*60}",
                    flush=True,
                )

                # Clonar args para no contaminar entre experimentos
                exp_args                      = copy.deepcopy(args)
                exp_args.task                 = task
                exp_args.patch_model          = patch_model
                exp_args.mil_aggregation_mode = mil_mode
                
                run_kfold_experiment(
                    df_task              = df_task,
                    base_datasets        = base_datasets,
                    fold_indices         = fold_indices,
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
