"""
utils/test_utils.py
--------------------
Utilidades para el script de entrenamiento definitivo y evaluación en test.

"""

import os
import re
import math
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# REGEX de nombres de carpeta de experimento
# Formato generado por train.py::run_kfolds:
#   [nickname_]YYYYMMDD_HHMMSS_task-TASK[_weighted_loss]
#   _model-MODEL_frozen|unfrozen_MIL-MILMODE_OPT_lrLR_wdWD_bsBS_epEP_KFOLDfold[_small]
# ──────────────────────────────────────────────────────────────────────────────

RUN_RE = re.compile(
    r"""
    (?:(?P<nickname>.+?)_)?           # nickname opcional
    (?P<timestamp>\d{8}_\d{6})_      # timestamp
    task-(?P<task>.+?)                # task  (puede tener guiones)
    (?P<weighted>_weighted_loss)?     # flag weighted loss
    _model-(?P<model>[^_]+)_          # patch model
    (?P<freeze>frozen|unfrozen)_      # frozen / unfrozen
    MIL-(?P<mil>[^_]+)_              # MIL mode
    (?P<optimizer>AdamW|SGD(?:_mom[\d.]+)?)_ # optimizer
    lr(?P<lr>[\d.e+-]+)_             # learning rate
    wd(?P<wd>[\d.e+-]+)_             # weight decay
    bs(?P<bs>\d+)_                   # batch size
    ep(?P<ep>\d+)                     # epochs
    _(?P<kfold>\d+)fold               # kfold
    (?P<small>_small)?                # flag small
    """,
    re.VERBOSE,
)

FOLD_RE = re.compile(r"^fold_(\d+)$")


# ──────────────────────────────────────────────────────────────────────────────
# PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_run_name(folder: str) -> Optional[dict]:
    """
    Extrae metadatos del nombre de la carpeta de experimento.

    Returns
    -------
    dict con las claves:
        run_name, nickname, timestamp, task, model, freeze, mil,
        optimizer, lr, wd, batch_size, max_epochs,
        weighted_loss, kfold, small
    o None si el nombre no coincide con el patrón esperado.
    """
    m = RUN_RE.search(folder)
    if m is None:
        return None
    d = m.groupdict()
    return dict(
        run_name=folder,
        nickname=d["nickname"] or "",
        timestamp=d["timestamp"],
        task=d["task"],
        model=d["model"],
        freeze=d["freeze"],
        mil=d["mil"],
        optimizer=d["optimizer"],
        lr=float(d["lr"]),
        wd=float(d["wd"]),
        batch_size=int(d["bs"]),
        max_epochs=int(d["ep"]),
        weighted_loss=d["weighted"] is not None,
        kfold=int(d["kfold"]),
        small=d["small"] is not None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE MÉTRICAS DE FOLDS
# ──────────────────────────────────────────────────────────────────────────────

def get_fold_dirs(run_dir: str) -> list:
    """
    Devuelve lista ordenada de subdirectorios fold_* dentro de run_dir.
    Lanza ValueError si no encuentra ninguno.
    """
    folds = [
        os.path.join(run_dir, name)
        for name in sorted(os.listdir(run_dir))
        if FOLD_RE.match(name) and os.path.isdir(os.path.join(run_dir, name))
    ]
    if not folds:
        raise ValueError(f"No se encontraron carpetas fold_* en: {run_dir}")
    return folds


def load_best_eval_metrics(run_dir: str) -> tuple:
    """
    Carga best_eval.csv de cada fold de un experimento.

    best_eval.csv contiene UNA sola fila: las métricas correspondientes a la
    mejor época de validación (mínima val_loss) de ese fold.

    Returns
    -------
    rows    : lista de pd.Series (una por fold).
    n_folds : número de folds encontrados.
    """
    try:
        fold_dirs = get_fold_dirs(run_dir)
    except ValueError:
        return [], 0

    rows = []
    for fd in fold_dirs:
        best_eval_path = os.path.join(fd, "best_eval.csv")
        if not os.path.isfile(best_eval_path):
            continue
        df = pd.read_csv(best_eval_path)
        df = df.drop(columns=["classification_report"], errors="ignore")
        if df.empty:
            continue
        # best_eval.csv tiene una fila; la tomamos directamente
        rows.append(df.iloc[0])

    return rows, len(fold_dirs)


def aggregate_fold_metrics(rows: list) -> dict:
    """
    Agrega las métricas de los folds como media y desviación estándar.

    Args
    ----
    rows : lista de pd.Series (una por fold), salida de load_best_eval_metrics.

    Returns
    -------
    dict con claves  <metric>  y  <metric>_std  para cada columna numérica,
    más 'n_folds_ok' con el número de folds con datos válidos.
    Devuelve {} si rows está vacío.
    """
    if not rows:
        return {}

    df = pd.DataFrame(rows).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    agg = {"n_folds_ok": len(rows)}
    for col in numeric_cols:
        vals = df[col].dropna().values
        if len(vals) == 0:
            agg[col] = float("nan")
            agg[f"{col}_std"] = float("nan")
        else:
            agg[col] = float(np.mean(vals))
            agg[f"{col}_std"] = float(np.std(vals, ddof=0))

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# SELECCIÓN DEL MEJOR EXPERIMENTO POR TAREA
# ──────────────────────────────────────────────────────────────────────────────

def select_best_config(
    experiments_path: str,
    tasks: list,
    rank_metric: str = "balanced_accuracy",
    ascending: bool = False,
    task_filter: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Escanea la carpeta de experimentos y selecciona la mejor configuración
    de hiperparámetros para cada tarea.

    La selección se basa en la media de `rank_metric` entre los k folds,
    tomada de best_eval.csv (mejor época de validación por fold).

    Args
    ----
    experiments_path : str — ruta a la subcarpeta 'experiments/' dentro de
                       results_path (p.ej. '/workspace/results/experiments').
    tasks            : list[str] — lista de tareas a considerar.
    rank_metric      : str  — métrica de ranking (por defecto 'balanced_accuracy').
    ascending        : bool — True si menor es mejor (ej. loss); False si mayor.
    task_filter      : str | None — si se especifica, filtra sólo esa tarea.
    verbose          : bool — imprime resumen de resultados.

    Returns
    -------
    best_per_task : dict con clave = nombre de tarea, valor = dict con:
        {
          'run_name'    : str,   # nombre de la carpeta del experimento ganador
          'run_dir'     : str,   # ruta absoluta
          'meta'        : dict,  # metadatos parseados del nombre de carpeta
          'agg_metrics' : dict,  # métricas agregadas entre folds
          rank_metric   : float, # valor medio de la métrica de ranking
          'best_epoch'  : int,   # media de las mejores épocas entre folds (redondeada)
        }
    """
    if not os.path.isdir(experiments_path):
        raise FileNotFoundError(
            f"La carpeta de experimentos no existe: {experiments_path}"
        )

    candidates = sorted(os.listdir(experiments_path))
    records = []
    skipped = 0

    for folder in candidates:
        run_dir = os.path.join(experiments_path, folder)
        if not os.path.isdir(run_dir):
            continue

        meta = parse_run_name(folder)
        if meta is None:
            skipped += 1
            continue

        # Filtro de tarea
        if task_filter and meta["task"] != task_filter:
            continue
        if meta["task"] not in tasks:
            skipped += 1
            continue

        rows, n_folds = load_best_eval_metrics(run_dir)
        if not rows:
            skipped += 1
            continue

        agg = aggregate_fold_metrics(rows)
        if not agg or math.isnan(agg.get(rank_metric, float("nan"))):
            skipped += 1
            continue

        record = {
            "run_name": folder,
            "run_dir":  run_dir,
            "meta":     meta,
            "agg_metrics": agg,
            rank_metric:   agg[rank_metric],
        }

        # Mejor época media entre folds
        if "epoch" in agg:
            record["best_epoch"] = int(round(agg["epoch"]))

        records.append(record)

    if verbose:
        print(
            f"\n[select_best_config] Experimentos escaneados : {len(candidates)}\n"
            f"  Válidos cargados : {len(records)}\n"
            f"  Omitidos / sin datos : {skipped}\n",
            flush=True,
        )

    if not records:
        raise RuntimeError(
            f"No se encontraron experimentos válidos en: {experiments_path}\n"
            "Comprueba que los folds contienen best_eval.csv y que la métrica "
            f"'{rank_metric}' está presente."
        )

    # ── Seleccionar el mejor por tarea ─────────────────────────────────────────
    best_per_task = {}

    for task in (tasks if not task_filter else [task_filter]):
        task_records = [r for r in records if r["meta"]["task"] == task]
        if not task_records:
            if verbose:
                print(f"  [WARN] No hay experimentos para la tarea '{task}'.", flush=True)
            continue

        sorted_records = sorted(
            task_records,
            key=lambda r: r[rank_metric],
            reverse=not ascending,
        )
        best = sorted_records[0]
        best_per_task[task] = best

        if verbose:
            meta = best["meta"]
            agg  = best["agg_metrics"]
            std  = agg.get(f"{rank_metric}_std", float("nan"))
            print(
                f"\n{'='*60}\n"
                f"  TAREA: {task}\n"
                f"{'='*60}\n"
                f"  Ganador : {best['run_name']}\n"
                f"  {rank_metric} (media±std) : "
                f"{best[rank_metric]:.4f} ± {std:.4f}\n"
                f"  Mejor época media  : {best.get('best_epoch', 'N/A')}\n"
                f"  Encoder            : {meta['model']}\n"
                f"  MIL                : {meta['mil']}\n"
                f"  lr / wd            : {meta['lr']} / {meta['wd']}\n"
                f"  optimizer          : {meta['optimizer']}\n"
                f"  weighted_loss      : {meta['weighted_loss']}\n"
                f"  batch_size         : {meta['batch_size']}\n"
                f"  kfold              : {meta['kfold']}\n",
                flush=True,
            )

    return best_per_task


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE ARGS DEFINITIVOS
# ──────────────────────────────────────────────────────────────────────────────

def build_final_args(base_args: SimpleNamespace, task: str, best: dict) -> SimpleNamespace:
    """
    Construye un Namespace de argumentos para el entrenamiento definitivo de
    una tarea, partiendo de `base_args` y sobreescribiendo con la configuración
    ganadora encontrada por `select_best_config`.

    El número de épocas de entrenamiento definitivo se fija como la media de las
    mejores épocas entre los folds (best['best_epoch']).  Si no está disponible,
    se usa base_args.epochs como fallback.

    Args
    ----
    base_args : SimpleNamespace — argumentos base (salida de argparse).
    task      : str             — nombre de la tarea.
    best      : dict            — entrada del dict devuelto por select_best_config.

    Returns
    -------
    SimpleNamespace con los argumentos finales listos para el entrenamiento
    definitivo.
    """
    import copy
    args = copy.deepcopy(base_args)

    meta = best["meta"]

    args.task                 = task
    args.patch_model          = meta["model"]
    args.mil_aggregation_mode = meta["mil"]
    args.learning_rate        = meta["lr"]
    args.weight_decay         = meta["wd"]
    args.batch_size           = meta["batch_size"]
    args.weighted_loss        = meta["weighted_loss"]

    # Número de épocas definitivas = media de las mejores épocas por fold
    # Se añade 1 porque los índices de época empiezan en 0
    if "best_epoch" in best:
        args.epochs = best["best_epoch"] + 1
    # else: se respeta base_args.epochs

    # Optimizer: SGD lleva momentum (se mantiene el del base_args ya que no
    # está en el nombre de carpeta con la resolución necesaria)
    args.optimizer = meta["optimizer"] if not meta["optimizer"].startswith("SGD") else "SGD"

    return args