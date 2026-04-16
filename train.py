import argparse
import os
import copy
from datetime import datetime
import time
from huggingface_hub import login

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from accelerate import Accelerator
import gc

from data_load import BreastWSI_Dataset, create_file_label_df, collate_fn
from utils.train_utils import set_multiple_seed, get_model, compute_class_weights, make_splits, run_epoch
from utils.metrics_utils import log_epoch_metrics, save_metrics_to_csv
from config import bcnb_images_folder_path, bcnb_labels_df_path, results_folder_path


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MIL breast cancer")

    # ── Datos ──────────────────────────────────────────────────────────────────
    parser.add_argument("--images_folder",  type=str, default=bcnb_images_folder_path)
    parser.add_argument("--df_path",        type=str, default=bcnb_labels_df_path)
    parser.add_argument("--slide_col",      type=str, default="Patient ID",
                        help="Columna del Excel que identifica cada WSI")
    parser.add_argument("--results_path",   type=str, default=results_folder_path)

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
        choices=["dino", "uni", "conch", "keep"],
        help="Extractor de características: dino | uni | conch | keep")
    parser.add_argument("--unfreeze_patch_model_backbone", action="store_true")
    parser.add_argument("--mil_aggregation_mode",       type=str,  default="attentionMIL",
                        choices=["max", "mean", "attentionMIL", "transMIL"])
    parser.add_argument("--chunk_size",                 type=int,  default=512,
                        help="Tamaño de mini-batch interno de parches (control VRAM)")
    parser.add_argument("--patch_resize_size",                 type=int,  default=224,
                        help="Dimensiones de reescalado de los parches")
    parser.add_argument("--patch_embedding_mode",                 type=int,  default=0,
                        help=(
                            "Modo de obtención del embedding de cada patch a partir de la salida del ViT. "
                            "0: usar el token CLS (representación global aprendida). "
                            "1: mean pooling de los tokens espaciales (promedio de subpatches, sin CLS)."
                        ))

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=20)
    parser.add_argument("--optimizer",     type=str,   default="AdamW",
                        choices=["SGD", "AdamW"])
    parser.add_argument("--momentum",      type=float, default=0.9,
                        help="Solo para SGD")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay",  type=float, default=1e-5)
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Aplicar pesos de clase en la función de pérdida")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("--load_on_demand",         action="store_true")
    parser.add_argument("--small_dataset_for_test", action="store_true")
    parser.add_argument("--seed",                   type=int, default=98)
    parser.add_argument("--experiment_nickname",    type=str, default="")

    parser.add_argument("--resume_training",        action="store_true")
    parser.add_argument("--pretrained_model_path",  type=str, default=None)

    parser.add_argument("--multiple_experiments",      action="store_true")

    parser.add_argument('--hf_token', type=str, default=None,
                        help="HuggingFace token para descargar modelos privados")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN — carga de datos, filtrado y splits para la tarea principal
# ──────────────────────────────────────────────────────────────────────────────
def main(args=None):
    if args is None:
        args = parse_args()
        
    # HuggingFace login para modelos privados
    if args.hf_token:
        login(token=args.hf_token)

    set_multiple_seed(args.seed)

    # ── DataFrame maestro ─────────────────────────────────────────────────────
    print("Construyendo DataFrame ...", flush=True)
    df = create_file_label_df(
        images_path=args.images_folder,
        data_path=args.df_path,
        slide_col=args.slide_col,
        small_dataset_for_test=args.small_dataset_for_test,
        ki67_threshold=args.ki67_threshold,
    )

    # ── Filtro histológico opcional ───────────────────────────────────────────
    if args.filter_ductal:
        before = len(df)
        df = df[df["tumour_type"] == "Invasive ductal carcinoma"].reset_index(drop=True)
        print(f"Filtro ductal: {before} → {len(df)} WSIs", flush=True)

    # ── Conversion de Patient ID a string ─────────────────────────────────────
    df["patient_id"] = df["patient_id"].astype(str)

    # ── Filtrar etiquetas inválidas para la tarea principal ───────────────────
    label_col    = args.task
    df = df[df[label_col] != -1].reset_index(drop=True)
    print(f"WSIs con etiqueta válida para '{args.task}': {len(df)}", flush=True)
    print(f"Distribución:\n{df[label_col].value_counts()}", flush=True)

    # ── Splits ────────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx, split_report_str = make_splits(df, label_col, args.seed)

    # ── Datasets ──────────────────────────────────────────────────────────────
    t0 = time.time()
    print("Cargando datasets ...", flush=True)
    ds_train = BreastWSI_Dataset(df, train_idx, task=args.task, patch_resize_size = args.patch_resize_size,
                                 load_on_demand=args.load_on_demand)
    ds_val   = BreastWSI_Dataset(df, val_idx,   task=args.task,  patch_resize_size = args.patch_resize_size,
                                 load_on_demand=args.load_on_demand)
    ds_test  = BreastWSI_Dataset(df, test_idx,  task=args.task,  patch_resize_size = args.patch_resize_size,
                                 load_on_demand=args.load_on_demand)
    print(f"Tiempo de carga: {time.time() - t0:.1f}s  "
          f"(train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)})", flush=True)

    return ds_train, ds_val, ds_test, df, args, split_report_str


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(ds_train, ds_val, ds_test, df, accelerator, device, args, split_report_str):

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── Modelo ────────────────────────────────────────────────────────────────
    num_classes = ds_train.num_classes
    print(f"Cargando modelo '{args.patch_model}' — num_classes={num_classes} ...", flush=True)
    model = get_model(
        patch_encoder_type=args.patch_model,
        mil_aggregation_mode=args.mil_aggregation_mode,
        unfreeze_backbone=args.unfreeze_patch_model_backbone,
        num_classes=num_classes,
    ).to(device)

    # ── Carpeta de resultados ─────────────────────────────────────────────────
    run_results_path = ""
    if accelerator.is_main_process:
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        nickname         = f"{args.experiment_nickname}_" if args.experiment_nickname else ""
        freeze_status    = "unfrozen" if args.unfreeze_patch_model_backbone else "frozen"
        opt_name         = args.optimizer if args.optimizer == "AdamW" \
                           else f"{args.optimizer}_mom{args.momentum}"
        small            = "_small" if args.small_dataset_for_test else ""
        ductal           = "_ductal" if args.filter_ductal else ""
        weighted_loss    = "_weighted_loss" if args.weighted_loss else ""
        run_name = (
            f"{nickname}{timestamp}_task-{args.task}{ductal}{weighted_loss}"
            f"_model-{args.patch_model}_{freeze_status}"
            f"_MIL-{args.mil_aggregation_mode}"
            f"_{opt_name}_lr{args.learning_rate}_wd{args.weight_decay}"
            f"_bs{args.batch_size}_ep{args.epochs}{small}"
        )
        run_results_path = os.path.join(args.results_path, run_name)
        os.makedirs(run_results_path, exist_ok=True)

    print(f"Resultados en: {run_results_path}", flush=True)
    best_model_path = os.path.join(run_results_path, "best_model")
    last_model_path = os.path.join(run_results_path, "last_model")
    log_path        = os.path.join(run_results_path, "train_log.txt")
    
    # Log splits distributions
    with open(log_path, "a") as f:
        f.write(split_report_str + "\n")

    # ── Pérdida ───────────────────────────────────────────────────────────────
    if args.weighted_loss:
        label_col    = args.task
        train_labels = df.iloc[list(ds_train.indices)][label_col].tolist()
        weights      = compute_class_weights(train_labels, num_classes).to(device)
        criterion    = nn.CrossEntropyLoss(weight=weights)
        print("Aplicando perdida con pesos por clase", flush=True)
        with open(log_path, "a") as f:
                f.write("Aplicando perdida con pesos por clase" + "\n")
        for cls, w in enumerate(weights.cpu().tolist()):
            print(f"  clase {cls}: {w:.4f}  (n={train_labels.count(cls)})", flush=True)
            with open(log_path, "a") as f:
                f.write(f"  clase {cls}: {w:.4f}  (n={train_labels.count(cls)})" + "\n")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizador y scheduler ───────────────────────────────────────────────
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
        patience=10, min_lr=1e-7, threshold=1e-4)

    # ── Accelerate ────────────────────────────────────────────────────────────
    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader, test_loader
    )

    if args.resume_training:
        if not args.pretrained_model_path:
            raise ValueError("--resume_training requiere --pretrained_model_path")
        accelerator.load_state(args.pretrained_model_path)
        print(f"Reanudando desde: {args.pretrained_model_path}", flush=True)

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(args.epochs):
        t_ep = time.time()
        if accelerator.is_main_process:
            print(f"\n──── Epoch {epoch} ────", flush=True)

        # Train
        train_metrics, train_conf_mat, train_loss, _ = run_epoch(
            model, train_loader, device, accelerator,
            criterion, optimizer, task=args.task, mode="train",
        )
        if accelerator.is_main_process:
            log_epoch_metrics(epoch, "Train", args.task, train_metrics, train_loss, log_path)
            save_metrics_to_csv(train_metrics, epoch=epoch, task=args.task,
                                filepath=os.path.join(run_results_path, "train_metrics.csv"))
            print("\n --- Matriz de confusión en Train ---\n", flush = True)
            print(train_conf_mat)
            with open(log_path, "a") as f:
                    f.write("\n Matriz de confusión en Train:\n")
                    f.write(train_conf_mat.to_string())
                    f.write("\n\n")
            

        # Val
        val_metrics, val_conf_mat, val_loss, _ = run_epoch(
            model, val_loader, device, accelerator,
            criterion, None, task=args.task, mode="val",
        )
        if accelerator.is_main_process:
            log_epoch_metrics(epoch, "Val", args.task, val_metrics, val_loss, log_path)
            save_metrics_to_csv(val_metrics, epoch=epoch, task=args.task,
                                filepath=os.path.join(run_results_path, "val_metrics.csv"))
            
            print("\n --- Matriz de confusión en Validación ---\n", flush = True)
            print(val_conf_mat)
            with open(log_path, "a") as f:
                    f.write("\n --- Matriz de confusión en Validación ---\n")
                    f.write(val_conf_mat.to_string())
                    f.write("\n\n")

        accelerator.wait_for_everyone()
        scheduler.step(val_loss)
        
        # Save last model for resume_training option
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.save_state(last_model_path)
            print(f"Época {epoch} — {time.time() - t_ep:.1f}s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"Época {epoch} — {time.time() - t_ep:.1f}s\n")
        
        # Checkpoint / Early Stopping
        if val_loss < best_val_loss:
            best_val_loss, no_improve = val_loss, 0
              
            if accelerator.is_main_process:
                accelerator.save_state(best_model_path)
                print(f"  → Mejor modelo guardado (val_loss={best_val_loss:.4f})", flush=True)
                with open(log_path, "a") as f:
                    f.write(f" Mejor modelo guardado (val_loss={best_val_loss:.4f})\n")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                if accelerator.is_main_process:
                    print("Early stopping activado.", flush=True)
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("\n──── TEST ────", flush=True)

    accelerator.wait_for_everyone()
    try:
        accelerator.load_state(best_model_path)
    except Exception as e:
        print(f"Error cargando checkpoint: {e}", flush=True)

    test_metrics, test_conf_mat, test_loss, _ = run_epoch(
        model, test_loader, device, accelerator,
        criterion, None, task=args.task, mode="test",
    )
    if accelerator.is_main_process:
        log_epoch_metrics("Test", "Test", args.task, test_metrics, test_loss, log_path)
        save_metrics_to_csv(test_metrics, epoch=0, task=args.task,
                            filepath=os.path.join(run_results_path, "test_metrics.csv"))
        print("\n --- Matriz de confusión en Test ---\n", flush = True)
        print(test_conf_mat)
        with open(log_path, "a") as f:
                    f.write("\n --- Matriz de confusión en Test ---\n")
                    f.write(test_conf_mat.to_string())
                    f.write("\n\n")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if not args.multiple_experiments:
        accelerator = Accelerator()
        device      = accelerator.device
        print(f"Dispositivo: {device}", flush=True)
        ds_train, ds_val, ds_test, df, args, split_report_str = main(args)
        run_experiment(ds_train, ds_val, ds_test, df, accelerator, device, args, split_report_str)
    else:
        print("Lanzando grid de experimentos ...", flush=True)
        experiments = [
            dict(task="er"),
            dict(task="pr"),
            dict(task="her2"),
            dict(task="ki67"),
            dict(task="molecular_subtype"),
        ]
        for exp in experiments:
            exp_args = copy.deepcopy(args)
            for k, v in exp.items():
                setattr(exp_args, k, v)
            exp_ds_train, exp_ds_val, exp_ds_test, exp_df, _, split_report_str = main(exp_args)
            exp_accelerator = Accelerator()
            exp_device      = exp_accelerator.device
            print(f"Dispositivo: {exp_device}", flush=True)
            run_experiment(exp_ds_train, exp_ds_val, exp_ds_test, exp_df, exp_accelerator, exp_device, exp_args, split_report_str)
            exp_accelerator.free_memory()
            del exp_ds_train, exp_ds_val, exp_ds_test, exp_df
            del exp_accelerator
            torch.cuda.empty_cache()
            gc.collect()
