"""
utils/test_utils.py
--------------------
Utilidades compartidas entre test.py y el notebook de exploración.

Lógica de selección de mejor configuración por tarea
-----------------------------------------------------
Cada fold guarda best_eval.csv con UNA sola fila: las métricas de la época
cuya val_loss fue mínima. La columna 'epoch' es el índice 0-based de esa época.

Al agregar entre folds se obtiene la MEDIA de esas mejores épocas.
Para el entrenamiento definitivo:
    epochs_final = round(best_epoch_mean) + 1

Criterio de selección / ranking
--------------------------------
Se ordena por rank_metric redondeado a 2 decimales (centésimas). En caso de
empate a centésimas, gana la configuración con menor std en rank_metric.
Si también hay empate en el std, se usa el valor exacto de rank_metric como
desempate final.

Estructura del JSON generado por save_best_config_json
-------------------------------------------------------
{
  "task":                str,
  "rank_metric":         str,
  "selected":            <entry>,
  "best_global":         <entry>,
  "ranking_encoder_mil": [<entry>],
  "full_ranking":        [<entry>],
}

Cada <entry> tiene: run_name, run_dir, n_folds, rank_metric, rank_value,
rank_value_std, best_epoch, hyperparams, metrics.
"""

import os
import re
import json
import math
import copy
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# REGEX DE NOMBRES DE CARPETA
# Formato: [nickname_]YYYYMMDD_HHMMSS_task-TASK_model-MODEL_frozen|unfrozen
#           _MIL-MILMODE_OPT_lrLR_wdWD_bsBS_epEP_KFOLDfold[_small]
# ─────────────────────────────────────────────────────────────────────────────

RUN_RE = re.compile(
    r"""
    (?:(?P<nickname>.+?)_)?
    (?P<timestamp>\d{8}_\d{6})_
    task-(?P<task>.+?)
    _model-(?P<model>[^_]+)_
    (?P<freeze>frozen|unfrozen)_
    MIL-(?P<mil>[^_]+)_
    (?P<optimizer>AdamW|SGD)_
    lr(?P<lr>[\d.e+-]+)_
    wd(?P<wd>[\d.e+-]+)_
    bs(?P<bs>\d+)_
    ep(?P<ep>\d+)
    _(?P<kfold>\d+)fold
    (?P<small>_small)?
    """,
    re.VERBOSE,
)

FOLD_RE = re.compile(r"^fold_(\d+)$")

DEFAULT_METRIC_COLS = [
    "balanced_accuracy",
    "accuracy",
    "kappa",
    "f1_macro",
    "auprc_macro",
    "auroc",
    "auroc_macro",
]


# ─────────────────────────────────────────────────────────────────────────────
# PARSING DEL NOMBRE DE CARPETA
# ─────────────────────────────────────────────────────────────────────────────

def parse_run_name(folder: str) -> Optional[dict]:
    """Extrae metadatos del nombre de carpeta. Devuelve None si no coincide."""
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
        kfold=int(d["kfold"]),
        small=d["small"] is not None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y AGREGACIÓN DE MÉTRICAS DE FOLDS
# ─────────────────────────────────────────────────────────────────────────────

def get_fold_dirs(run_dir: str) -> list:
    """Devuelve lista ordenada de subdirectorios fold_* dentro de run_dir."""
    folds = [
        os.path.join(run_dir, name)
        for name in sorted(os.listdir(run_dir))
        if FOLD_RE.match(name) and os.path.isdir(os.path.join(run_dir, name))
    ]
    if not folds:
        raise ValueError(f"No se encontraron carpetas fold_* en: {run_dir}")
    return folds


def _load_best_eval(fold_dir: str) -> Optional[pd.Series]:
    """Carga best_eval.csv de un fold y devuelve su única fila como Series."""
    path = os.path.join(fold_dir, "best_eval.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path).drop(columns=["classification_report"], errors="ignore")
    return df.iloc[0] if not df.empty else None


def aggregate_fold_metrics(rows: list) -> dict:
    """
    Agrega métricas de los folds (media y std, ddof=0).

    Args
    ----
    rows : lista de pd.Series, una por fold (de best_eval.csv).

    Returns
    -------
    dict con claves <metric> y <metric>_std para cada columna numérica,
    más 'n_folds_ok'. Devuelve {} si rows está vacío.
    """
    if not rows:
        return {}

    all_cols = set()
    for r in rows:
        all_cols.update(r.index)

    agg = {"n_folds_ok": len(rows)}
    for col in all_cols:
        vals = []
        for r in rows:
            if col in r.index:
                try:
                    vals.append(float(r[col]))
                except (TypeError, ValueError):
                    pass
        if vals:
            agg[col] = float(np.mean(vals))
            agg[f"{col}_std"] = float(np.std(vals, ddof=0))
        else:
            agg[col] = float("nan")
            agg[f"{col}_std"] = float("nan")

    return agg


def load_and_aggregate_run(run_dir: str, rank_metric: str) -> Optional[dict]:
    """
    Carga best_eval.csv de cada fold y devuelve las métricas agregadas.
    Devuelve None si algún fold no tiene best_eval.csv o falta rank_metric.
    """
    try:
        fold_dirs = get_fold_dirs(run_dir)
    except ValueError:
        return None

    rows = []
    for fd in fold_dirs:
        row = _load_best_eval(fd)
        if row is None:
            return None  # best_eval.csv obligatorio en todos los folds
        rows.append(row)

    agg = aggregate_fold_metrics(rows)
    if not agg or math.isnan(agg.get(rank_metric, float("nan"))):
        return None

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# CRITERIO DE ORDENACIÓN / DESEMPATE
# ─────────────────────────────────────────────────────────────────────────────

def _record_sort_key(record: dict, rank_metric: str, ascending: bool) -> tuple:
    """
    Clave de ordenación para records internos (de select_best_config).

    Orden:
      1. rank_metric redondeado a 2 decimales (centésimas).
      2. std de rank_metric (siempre: menor std es mejor, sea ascending o no).
      3. valor exacto de rank_metric (desempate final).

    La tupla resultante se ordena de forma ascendente (sorted normal), de modo
    que el primer elemento sea siempre la mejor configuración.
    """
    val = record[rank_metric]
    std = record["agg_metrics"].get(f"{rank_metric}_std", float("inf"))
    if math.isnan(std):
        std = float("inf")

    rounded = round(val, 2)

    if ascending:
        # Menor valor es mejor → orden ascendente natural.
        return (rounded, std, val)
    else:
        # Mayor valor es mejor → invertimos signo para que sorted() ascendente
        # deje primero al de mayor valor. El std NO se invierte: menor std
        # siempre gana el empate, independientemente de 'ascending'.
        return (-rounded, std, -val)


def _entry_sort_key(entry: dict, ascending: bool) -> tuple:
    """
    Clave de ordenación equivalente a _record_sort_key pero para entries
    ya serializados (dicts con 'rank_value' y 'rank_value_std'), usados al
    construir los rankings del JSON.
    """
    val = entry["rank_value"]
    std = entry.get("rank_value_std", float("inf"))
    if std is None or (isinstance(std, float) and math.isnan(std)):
        std = float("inf")

    rounded = round(val, 2)

    if ascending:
        return (rounded, std, val)
    else:
        return (-rounded, std, -val)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATAFRAME MAESTRO (usado por el notebook)
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_runs_df(
    experiments_path: str,
    rank_metric: str = "balanced_accuracy",
    ascending: bool = False,
    task_filter: Optional[str] = None,
    metric_cols: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Escanea experiments_path y devuelve un DataFrame con todos los experimentos,
    ordenado por rank_metric (mejor primero).

    Columnas: metadatos del run + rank_value + métricas (media ± std) + best_epoch.
    """
    if metric_cols is None:
        metric_cols = DEFAULT_METRIC_COLS

    if not os.path.isdir(experiments_path):
        raise FileNotFoundError(f"Carpeta no encontrada: {experiments_path}")

    records, skipped = [], 0

    for folder in sorted(os.listdir(experiments_path)):
        run_dir = os.path.join(experiments_path, folder)
        if not os.path.isdir(run_dir):
            continue

        meta = parse_run_name(folder)
        if meta is None:
            skipped += 1
            continue

        if task_filter and meta["task"] != task_filter:
            continue

        agg = load_and_aggregate_run(run_dir, rank_metric)
        if agg is None:
            skipped += 1
            continue

        record = {**meta, "rank_value": agg[rank_metric], "n_folds": agg["n_folds_ok"]}

        for col in metric_cols:
            if col in agg:
                record[col] = agg[col]
                record[f"{col}_std"] = agg.get(f"{col}_std", float("nan"))

        if "epoch" in agg:
            record["best_epoch"] = agg["epoch"]

        records.append(record)

    if verbose:
        print(
            f"[collect_all_runs_df] Carpetas escaneadas: {len(os.listdir(experiments_path))}\n"
            f"  Válidos: {len(records)}  |  Omitidos: {skipped}"
        )

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("rank_value", ascending=ascending, na_position="last")
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# SELECCIÓN DEL MEJOR EXPERIMENTO POR TAREA
# ─────────────────────────────────────────────────────────────────────────────

def select_best_config(
    experiments_path: str,
    tasks: list,
    rank_metric: str = "balanced_accuracy",
    ascending: bool = False,
    task_filter: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Escanea experiments_path y selecciona la mejor configuración por tarea.

    Criterio de selección (para el global y para cada encoder x MIL):
      1. rank_metric redondeado a 2 decimales.
      2. En caso de empate a centésimas, menor std de rank_metric.
      3. En caso de empate también en std, valor exacto de rank_metric.

    Returns
    -------
    best_per_task : dict  {task_name -> record}

    Cada record contiene:
        run_name, run_dir, meta, agg_metrics, n_folds, <rank_metric>,
        best_epoch, all_records (lista completa ordenada),
        best_per_encoder_mil (dict {"MODEL_MIL" -> record})
    """
    if not os.path.isdir(experiments_path):
        raise FileNotFoundError(f"Carpeta no encontrada: {experiments_path}")

    records_by_task: dict = {t: [] for t in tasks}
    skipped = 0

    for folder in sorted(os.listdir(experiments_path)):
        run_dir = os.path.join(experiments_path, folder)
        if not os.path.isdir(run_dir):
            continue

        meta = parse_run_name(folder)
        if meta is None:
            skipped += 1
            continue

        if task_filter and meta["task"] != task_filter:
            continue
        if meta["task"] not in tasks:
            skipped += 1
            continue

        agg = load_and_aggregate_run(run_dir, rank_metric)
        if agg is None:
            skipped += 1
            continue

        record = {
            "run_name":    folder,
            "run_dir":     run_dir,
            "meta":        meta,
            "agg_metrics": agg,
            "n_folds":     agg["n_folds_ok"],
            rank_metric:   agg[rank_metric],
        }
        if "epoch" in agg:
            record["best_epoch"] = agg["epoch"]

        records_by_task[meta["task"]].append(record)

    if verbose:
        total = sum(len(v) for v in records_by_task.values())
        print(
            f"[select_best_config] Escaneados: {len(os.listdir(experiments_path))}  "
            f"Válidos: {total}  Omitidos: {skipped}"
        )

    task_list = [task_filter] if task_filter else tasks
    best_per_task = {}

    for task in task_list:
        task_records = records_by_task.get(task, [])
        if not task_records:
            if verbose:
                print(f"  [WARN] Sin experimentos para '{task}'")
            continue

        sorted_records = sorted(
            task_records,
            key=lambda r: _record_sort_key(r, rank_metric, ascending),
        )
        best = sorted_records[0]

        # Mejor por combinación encoder × MIL
        enc_mil_groups: dict = {}
        for r in task_records:
            key = f"{r['meta']['model']}_{r['meta']['mil']}"
            enc_mil_groups.setdefault(key, []).append(r)

        best["best_per_encoder_mil"] = {
            key: sorted(
                group, key=lambda r: _record_sort_key(r, rank_metric, ascending)
            )[0]
            for key, group in enc_mil_groups.items()
        }
        best["all_records"] = sorted_records
        best_per_task[task] = best

        if verbose:
            meta = best["meta"]
            agg  = best["agg_metrics"]
            std  = agg.get(f"{rank_metric}_std", float("nan"))
            print(
                f"\n{'='*60}\n  TAREA: {task}\n{'='*60}\n"
                f"  Ganador : {best['run_name']}\n"
                f"  {rank_metric}: {best[rank_metric]:.4f} ± {std:.4f}\n"
                f"  best_epoch (media): {best.get('best_epoch', 'N/A')}\n"
                f"  Encoder / MIL : {meta['model']} / {meta['mil']}\n"
                f"  lr / wd       : {meta['lr']} / {meta['wd']}\n"
                f"  optimizer     : {meta['optimizer']}\n"
                f"  batch_size    : {meta['batch_size']}\n"
            )
            enc_mil_sorted = sorted(
                best["best_per_encoder_mil"].items(),
                key=lambda kv: _record_sort_key(kv[1], rank_metric, ascending),
            )
            print("  ── Mejor por encoder x MIL ──")
            for key, rec in enc_mil_sorted:
                val = rec[rank_metric]
                std = rec["agg_metrics"].get(f"{rank_metric}_std", float("nan"))
                marker = "  ★" if rec["run_name"] == best["run_name"] else "   "
                print(
                    f"{marker} {key:50s}  "
                    f"{rank_metric}={val:.4f} ± {std:.4f}  "
                    f"epoch={rec.get('best_epoch', '?'):.1f}"
                )

    return best_per_task


# ─────────────────────────────────────────────────────────────────────────────
# GUARDADO EN JSON
# ─────────────────────────────────────────────────────────────────────────────

def _record_to_entry(record: dict, rank_metric: str,
                     metric_cols: Optional[list] = None) -> dict:
    """Convierte un record interno a dict serializable para el JSON."""
    if metric_cols is None:
        metric_cols = DEFAULT_METRIC_COLS
    meta = record["meta"]
    agg  = record["agg_metrics"]
    entry = {
        "run_name":       record["run_name"],
        "run_dir":        record["run_dir"],
        "n_folds":        record["n_folds"],
        "rank_metric":    rank_metric,
        "rank_value":     record[rank_metric],
        "rank_value_std": agg.get(f"{rank_metric}_std", float("nan")),
        "best_epoch":     record.get("best_epoch"),
        "hyperparams": {
            "model":      meta["model"],
            "freeze":     meta["freeze"],
            "mil":        meta["mil"],
            "optimizer":  meta["optimizer"],
            "lr":         meta["lr"],
            "wd":         meta["wd"],
            "batch_size": meta["batch_size"],
            "kfold":      meta["kfold"],
        },
        "metrics": {
            col: agg[col]
            for col in metric_cols
            if col in agg
        },
    }
    # Añadir std de cada métrica
    for col in metric_cols:
        std_key = f"{col}_std"
        if std_key in agg:
            entry["metrics"][std_key] = agg[std_key]
    return entry


def save_best_config_json(
    best_per_task: dict,
    task_results_path: str,
    task: str,
    rank_metric: str,
    selected_enc_mil_key: Optional[str] = None,
    metric_cols: Optional[list] = None,
    ascending: bool = False,
) -> str:
    """
    Guarda <task_results_path>/best_config.json con la selección de la mejor
    configuración para una tarea.

    El ranking (ranking_encoder_mil y full_ranking) se ordena con el mismo
    criterio que select_best_config: rank_metric redondeado a 2 decimales,
    desempate por menor std de rank_metric, y como último desempate el valor
    exacto de rank_metric.

    Estructura del JSON
    --------------------
    {
      "task":                str,
      "rank_metric":         str,
      "selected":            <entry>,   # config usada en test.py
      "best_global":         <entry>,   # mejor global de la tarea
      "ranking_encoder_mil": [<entry>], # mejor por encoder x MIL, ordenado
      "full_ranking":        [<entry>], # Lista completa de experimentos, ordenados
    }
    """
    if metric_cols is None:
        metric_cols = DEFAULT_METRIC_COLS

    os.makedirs(task_results_path, exist_ok=True)
    best        = best_per_task[task]
    enc_mil_map = best["best_per_encoder_mil"]
    all_records = best["all_records"]

    best_global_entry = _record_to_entry(best, rank_metric, metric_cols)

    ranking_enc_mil = sorted(
        [
            {**_record_to_entry(rec, rank_metric, metric_cols),
             "encoder_mil_key": key}
            for key, rec in enc_mil_map.items()
        ],
        key=lambda e: _entry_sort_key(e, ascending),
    )

    full_ranking = sorted(
        [
            {**_record_to_entry(rec, rank_metric, metric_cols),
             "encoder_mil_key": f"{rec['meta']['model']}_{rec['meta']['mil']}"}
            for rec in all_records
        ],
        key=lambda e: _entry_sort_key(e, ascending),
    )

    if selected_enc_mil_key and selected_enc_mil_key in enc_mil_map:
        selected = {
            **_record_to_entry(enc_mil_map[selected_enc_mil_key], rank_metric, metric_cols),
            "encoder_mil_key":  selected_enc_mil_key,
            "selection_mode":   "manual_override",
        }
    else:
        selected = {
            **best_global_entry,
            "encoder_mil_key": f"{best['meta']['model']}_{best['meta']['mil']}",
            "selection_mode":  "auto_best_global",
        }

    output = {
        "task":                task,
        "rank_metric":         rank_metric,
        "selected":            selected,
        "best_global":         best_global_entry,
        "ranking_encoder_mil": ranking_enc_mil,
        "full_ranking":        full_ranking,
    }

    json_path = os.path.join(task_results_path, "best_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return json_path


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE ARGS DEFINITIVOS
# ─────────────────────────────────────────────────────────────────────────────

def build_final_args(
    base_args: SimpleNamespace,
    task: str,
    best: dict,
    encoder_mil_override: Optional[str] = None,
) -> SimpleNamespace:
    """
    Construye los args para el entrenamiento definitivo de una tarea.

    Épocas: epochs_final = round(best_epoch_mean) + 1
    (best_epoch es índice 0-based, media entre folds)
    """
    args = copy.deepcopy(base_args)
    enc_mil_map = best["best_per_encoder_mil"]

    if encoder_mil_override is not None:
        if encoder_mil_override not in enc_mil_map:
            raise ValueError(
                f"Override '{encoder_mil_override}' no encontrado para '{task}'. "
                f"Disponibles: {list(enc_mil_map.keys())}"
            )
        chosen = enc_mil_map[encoder_mil_override]
        print(f"[build_final_args] Tarea '{task}': override '{encoder_mil_override}'")
    else:
        chosen = best

    meta = chosen["meta"]
    args.task                 = task
    args.patch_model          = meta["model"]
    args.mil_aggregation_mode = meta["mil"]
    args.learning_rate        = meta["lr"]
    args.weight_decay         = meta["wd"]
    args.batch_size           = meta["batch_size"]
    args.optimizer            = meta["optimizer"]

    if "best_epoch" in chosen:
        args.epochs = int(round(chosen["best_epoch"])) + 1

    return args