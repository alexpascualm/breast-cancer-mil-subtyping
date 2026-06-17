"""
compute_srg.py
--------------
Calcula el Symmetric Relevance Gain (SRG) de los heatmaps de atención,
WSI por WSI, para cada tarea, sobre todo el conjunto de test_set.

Implementa el procedimiento de "patch flipping" descrito en:
  Idaji, Hense et al., "Beyond Attention Heatmaps: How to Get Better
  Explanations for Multiple Instance Learning Models in Histopathology"
  (2026), Sección 3.3.

Procedimiento por WSI
----------------------
  1. Se calculan los pesos de atención (heatmap) del modelo sobre la bolsa
     completa de parches.
  2. Se ordenan los parches por peso de atención, en orden ascendente
     (menos relevante primero) y descendente (más relevante primero).
  3. Cada orden se particiona en `n_groups` grupos (percentiles) y se
     eliminan progresivamente de la bolsa, midiendo en cada paso la
     probabilidad softmax que el modelo asigna a una clase objetivo.
  4. AUPC(orden) = promedio de esas probabilidades a lo largo de la curva.
  5. SRG = AUPC(ascendente) - AUPC(descendente).

Un SRG mayor indica un heatmap más fiel al mecanismo de decisión del
modelo (no necesariamente "más correcto biológicamente": el SRG mide
fidelidad al modelo, no verdad clínica).

Notas metodológicas importantes
--------------------------------
Clase objetivo (`--target_class`): por defecto se usa la clase PREDICHA
  por el modelo, no la verdadera. Esto es así porque el SRG está diseñado
  para responder "¿el heatmap explica por qué el modelo decidió lo que
  decidió?", no "¿el modelo tiene razón?". Si usas la clase verdadera en
  WSIs mal clasificadas, terminas rastreando la probabilidad de una clase
  que el modelo no eligió, lo cual mezcla fidelidad explicativa con
  corrección del modelo. Si tu interés es más bien de validación
  biológica, usa --target_class true y considera filtrar después por
  `correct == 1` en el CSV de salida.

Baseline aleatorio (`--n_random_repeats`): un valor de SRG aislado no es
  interpretable por sí mismo, porque está en las unidades de salida del
  modelo y no tiene una escala absoluta de "bueno"/"malo". La forma
  mínima de saber si el SRG de la atención es informativo (y no
  indistinguible de un orden aleatorio de parches) es compararlo contra
  un baseline donde el heatmap se sustituye por una permutación aleatoria,
  repetida varias veces y promediada. Por defecto está desactivado (más
  rápido); se recomienda activarlo (>=3) al menos una vez por tarea.

Requisitos
----------
Este script debe ejecutarse desde el mismo directorio que
`generate_attention_maps.py` (o tenerlo en el PYTHONPATH), ya que reutiliza
sus funciones de carga de configuración, modelo y dataset, para garantizar
exactamente la misma convención de inferencia que se usó para generar los
mapas de atención.

Estructura de salida
---------------------
  <output_path>/
    log.txt
    <task>_srg.csv        ← una fila por WSI (resumible / append-safe)
    <task>_summary.json   ← estadísticas a nivel de cohorte

Uso mínimo
----------
  python compute_srg.py --test_results_path /workspace/results/my_run/test_results

Uso con baseline aleatorio (recomendado al menos una vez):
  python compute_srg.py --test_results_path ... --n_random_repeats 5

Advertencia de rendimiento
---------------------------
Por cada WSI se requieren ~2*n_groups + 1 pasadas forward (≈201 con los
valores por defecto), más 2*n_groups+1 adicionales por cada repetición del
baseline aleatorio. Se recomienda encarecidamente usar embeddings
precalculados (--embeddings_folder) en vez de parches crudos: en modo de
parches crudos, cada pasada reprocesa todos los parches restantes a través
del encoder completo, lo cual puede ser extremadamente lento.
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Imports del proyecto
# ──────────────────────────────────────────────────────────────────────────────
from config import (
    bcnb_images_folder_path,
    bcnb_labels_df_path,
    bcnb_embeddings_path,
)
from data_load import collate_fn
from utils.train_utils import set_multiple_seed

try:
    from generate_attention_maps import (
        SUPPORTED_MIL,
        TeeLogger,
        load_trained_model,
        build_dataset,
        run_inference_single_wsi,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar 'generate_attention_maps'. Este script debe "
        "estar en el mismo directorio que generate_attention_maps.py (o "
        "este último debe estar en el PYTHONPATH), ya que reutiliza su "
        "lógica de carga de modelo y construcción de datasets."
    ) from e


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calcula el SRG (patch flipping) de los heatmaps de atención "
            "para todas las WSIs del test_set, WSI por WSI, para cada tarea."
        )
    )

    parser.add_argument(
        "--test_results_path", type=str, required=True,
        help="Ruta a la carpeta de resultados de test.py (contiene subcarpetas por tarea)."
    )
    parser.add_argument(
        "--images_folder", type=str, default=bcnb_images_folder_path,
        help="Ruta a la carpeta con los ficheros .h5 de parches de las WSIs."
    )
    parser.add_argument(
        "--embeddings_folder", type=str, default=bcnb_embeddings_path,
        help=(
            "Ruta raíz de embeddings precomputados. Si se da, se usa "
            "Embeddings_WSI_Dataset (recomendado, mucho más rápido para SRG) "
            "en lugar de Patches_WSI_Dataset."
        )
    )
    parser.add_argument(
        "--df_path", type=str, default=bcnb_labels_df_path,
        help="Ruta al fichero de etiquetas (.xlsx)."
    )
    parser.add_argument(
        "--test_df_path", type=str, default=None,
        help="CSV con el test set. Por defecto: <parent de test_results_path>/test_df.csv"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help=(
            "Carpeta de salida. Por defecto: "
            "<parent de test_results_path>/srg_results"
        )
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=["er", "pr", "her2", "ki67", "molecular_subtype"],
        choices=["er", "pr", "her2", "ki67", "molecular_subtype"],
        help="Tareas para las que calcular el SRG."
    )
    parser.add_argument(
        "--target_class", type=str, default="predicted",
        choices=["predicted", "true"],
        help=(
            "Clase usada como referencia para el ranking y la curva de "
            "perturbación. 'predicted' (recomendado): mide fidelidad al "
            "mecanismo de decisión real del modelo. 'true': mide presencia "
            "de evidencia de la clase correcta, mezclando fidelidad con "
            "corrección del modelo (ver docstring del script)."
        )
    )
    parser.add_argument(
        "--n_groups", type=int, default=100,
        help="Número de grupos/percentiles para la partición de patch flipping (paper usa 100)."
    )
    parser.add_argument(
        "--n_random_repeats", type=int, default=0,
        help=(
            "Repeticiones de permutación aleatoria por WSI para estimar un "
            "baseline aleatorio de SRG (recomendado >=3 al menos una vez, "
            "para saber si el SRG de la atención es informativo o "
            "indistinguible de ruido). 0 = desactivado (más rápido)."
        )
    )
    parser.add_argument(
        "--max_wsis", type=int, default=None,
        help="Límite opcional de WSIs a procesar por tarea (útil para pruebas rápidas)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Si se indica, ignora resultados previos y recalcula todo desde cero."
    )
    parser.add_argument("--seed", type=int, default=98)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL TEST SET
# ──────────────────────────────────────────────────────────────────────────────

def load_test_split(args) -> pd.DataFrame:
    """Carga únicamente test_df desde el CSV guardado por train.py."""
    parent = os.path.dirname(os.path.abspath(args.test_results_path))
    test_path = args.test_df_path or os.path.join(parent, "test_df.csv")

    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"test_df.csv no encontrado en: {test_path}\n"
            "Usa --test_df_path para indicar la ruta correcta."
        )

    test_df = pd.read_csv(test_path)
    print(f"Test set cargado: {len(test_df)} WSIs")
    return test_df


# ──────────────────────────────────────────────────────────────────────────────
# PATCH FLIPPING — PRIMITIVAS
# ──────────────────────────────────────────────────────────────────────────────

def _forward_prob(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
    target_class: int,
    keep_mask: np.ndarray | None,
) -> float:
    """
    Pasa la bolsa (completa o un subconjunto definido por keep_mask) por el
    modelo y devuelve la probabilidad softmax de `target_class`.

    keep_mask=None            -> bolsa completa (sin eliminar parches)
    keep_mask con todo False  -> bolsa vacía -> se pasa un único parche de
                                  ceros (Sección 3.3.1 del paper: "para
                                  m=100 ... pasamos un array de zeros").
    """
    if keep_mask is None:
        feats_subset = features
    else:
        n_keep = int(keep_mask.sum())
        if n_keep == 0:
            feats_subset = torch.zeros(
                (1,) + tuple(features.shape[1:]), dtype=features.dtype
            )
        else:
            feats_subset = features[torch.from_numpy(keep_mask)]

    feats_subset = feats_subset.to(device)

    with torch.no_grad():
        if use_precomputed:
            logits, _, _ = model.forward_from_embeddings([feats_subset])
        else:
            logits, _, _ = model([feats_subset])

    probs = torch.softmax(logits[0], dim=0)
    return float(probs[target_class].item())


def _build_percentile_groups(order_idx: np.ndarray, n_groups: int) -> list:
    """
    Particiona los índices ya ordenados (orden de eliminación: el primer
    índice se elimina primero) en `n_groups` grupos lo más equilibrados
    posible. Si hay menos parches que grupos, se devuelven menos de
    n_groups grupos no vacíos (uno por parche), en vez de fallar.
    """
    chunks = np.array_split(order_idx, n_groups)
    return [c for c in chunks if len(c) > 0]


def _perturbation_curve(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
    groups: list,
    target_class: int,
    prob_full: float,
) -> np.ndarray:
    """
    Construye la curva de perturbación: probabilidad del modelo en función
    de la fracción de parches eliminados, eliminando los grupos de forma
    acumulativa (Ecuación 1 del paper). El primer punto (m=0) es prob_full,
    reutilizado para no recalcularlo.
    """
    curve = np.empty(len(groups) + 1, dtype=np.float64)
    curve[0] = prob_full

    keep_mask = np.ones(features.shape[0], dtype=bool)
    for m, group in enumerate(groups, start=1):
        keep_mask[group] = False
        curve[m] = _forward_prob(
            model, features, device, use_precomputed, target_class, keep_mask
        )
    return curve


def compute_srg_for_wsi(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
    weights: np.ndarray,
    target_class: int,
    n_groups: int,
) -> tuple:
    """
    Calcula el SRG para una WSI, dado su heatmap de atención (`weights`).

    Returns
    -------
    srg, aupc_ascending, aupc_descending, prob_full_bag
    """
    prob_full = _forward_prob(
        model, features, device, use_precomputed, target_class, keep_mask=None
    )

    # Orden de eliminación: descendente = más relevante primero (se quita primero)
    #                        ascendente  = menos relevante primero (se quita primero)
    order_desc = np.argsort(-weights)
    order_asc = np.argsort(weights)

    groups_desc = _build_percentile_groups(order_desc, n_groups)
    groups_asc = _build_percentile_groups(order_asc, n_groups)

    curve_desc = _perturbation_curve(
        model, features, device, use_precomputed, groups_desc, target_class, prob_full
    )
    curve_asc = _perturbation_curve(
        model, features, device, use_precomputed, groups_asc, target_class, prob_full
    )

    aupc_desc = float(curve_desc.mean())
    aupc_asc = float(curve_asc.mean())
    srg = aupc_asc - aupc_desc

    return srg, aupc_asc, aupc_desc, prob_full


def _random_baseline_srg(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
    target_class: int,
    n_groups: int,
    n_repeats: int,
    rng: np.random.Generator,
    prob_full: float,
) -> tuple:
    """
    Calcula el SRG promedio (y su desviación) sobre `n_repeats` órdenes de
    parches aleatorios, como referencia de "ruido" para interpretar si el
    SRG real es informativo.
    """
    n_patches = features.shape[0]
    srgs = []

    for _ in range(n_repeats):
        rand_scores = rng.random(n_patches)
        order_desc = np.argsort(-rand_scores)
        order_asc = np.argsort(rand_scores)

        groups_desc = _build_percentile_groups(order_desc, n_groups)
        groups_asc = _build_percentile_groups(order_asc, n_groups)

        curve_desc = _perturbation_curve(
            model, features, device, use_precomputed, groups_desc, target_class, prob_full
        )
        curve_asc = _perturbation_curve(
            model, features, device, use_precomputed, groups_asc, target_class, prob_full
        )
        srgs.append(float(curve_asc.mean()) - float(curve_desc.mean()))

    return float(np.mean(srgs)), float(np.std(srgs))


# ──────────────────────────────────────────────────────────────────────────────
# ESTADÍSTICA: TAMAÑO DE EFECTO DE WILCOXON (Ecs. 22-25 del paper)
# ──────────────────────────────────────────────────────────────────────────────

def _wilcoxon_effect_size(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Tamaño de efecto r = z / sqrt(n) del test de Wilcoxon de rango con
    signo para las diferencias pareadas x - y (mismas fórmulas que el
    paper, Sección E / Ecs. 22-25). Signo positivo => x > y en promedio.
    No depende de scipy (usa la aproximación normal vía math.erf).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n < 1:
        return float("nan"), float("nan")

    abs_d = np.abs(d)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()

    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma if sigma > 0 else 0.0
    r = z / math.sqrt(n)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    return float(r), float(p)


def summarize_task(csv_path: str, task: str, n_random_repeats: int) -> dict:
    """Estadísticas a nivel de cohorte a partir del CSV por-WSI de una tarea."""
    df = pd.read_csv(csv_path)

    summary = {
        "task": task,
        "n_wsis": int(len(df)),
        "srg_mean": float(df["srg"].mean()),
        "srg_std": float(df["srg"].std()),
        "srg_median": float(df["srg"].median()),
        "pct_srg_positive": float((df["srg"] > 0).mean() * 100),
    }

    if n_random_repeats > 0 and "srg_random_mean" in df.columns and df["srg_random_mean"].notna().any():
        valid = df.dropna(subset=["srg_random_mean"])
        summary["srg_random_mean"] = float(valid["srg_random_mean"].mean())
        summary["pct_srg_above_random"] = float(
            (valid["srg"] > valid["srg_random_mean"]).mean() * 100
        )
        r, p = _wilcoxon_effect_size(
            valid["srg"].to_numpy(), valid["srg_random_mean"].to_numpy()
        )
        summary["wilcoxon_effect_size_vs_random"] = r
        summary["wilcoxon_p_value_vs_random"] = p

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE POR TAREA
# ──────────────────────────────────────────────────────────────────────────────

def process_task_srg(
    task: str,
    args,
    test_df: pd.DataFrame,
    device: torch.device,
    output_path: str,
) -> dict | None:
    """Calcula el SRG de todas las WSIs del test_set para una tarea dada."""
    task_results_path = os.path.join(args.test_results_path, task)
    config_path = os.path.join(task_results_path, "best_config.json")

    print(f"\n{'='*65}")
    print(f"  TAREA: {task}")
    print(f"{'='*65}")

    if not os.path.isfile(config_path):
        print(f"  [SKIP] best_config.json no encontrado en: {task_results_path}")
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    hyperparams = cfg["selected"]["hyperparams"]
    enc_name = hyperparams["model"]
    mil_mode = hyperparams["mil"]

    if mil_mode not in SUPPORTED_MIL:
        print(
            f"  [SKIP] MIL '{mil_mode}' no soportado para heatmaps de atención "
            f"(requiere uno de {SUPPORTED_MIL})."
        )
        return None

    print(f"  Encoder: {enc_name}  |  MIL: {mil_mode}")

    # ── Resolver embeddings precomputados ────────────────────────────────
    emb_folder_enc = None
    use_precomputed = False
    if args.embeddings_folder is not None:
        candidate = os.path.join(args.embeddings_folder, enc_name)
        if os.path.isdir(candidate):
            emb_folder_enc = candidate
            use_precomputed = True
        else:
            print(
                f"  [WARN] Carpeta de embeddings no encontrada: {candidate}. "
                "Se usarán parches crudos."
            )

    if not use_precomputed:
        print(
            "  [AVISO] Modo de parches crudos: cada paso de patch-flipping "
            "reprocesa los parches restantes a través del encoder completo "
            "(~200+ pasadas por WSI). Se recomienda encarecidamente usar "
            "--embeddings_folder con embeddings precalculados."
        )

    num_classes = 4 if task == "molecular_subtype" else 2

    # ── Cargar modelo ─────────────────────────────────────────────────────
    try:
        model, _, _ = load_trained_model(
            task_results_path, task, num_classes, use_precomputed, device
        )
    except Exception as e:
        print(f"  [ERROR] Carga del modelo fallida: {e}")
        return None

    # ── Dataset: todas las WSIs del test_set (o subconjunto con --max_wsis) ─
    df_eval = test_df.reset_index(drop=True)
    if args.max_wsis is not None:
        df_eval = df_eval.head(args.max_wsis).reset_index(drop=True)

    try:
        dataset, df_valid = build_dataset(df_eval, task, use_precomputed, emb_folder_enc)
    except Exception as e:
        print(f"  [ERROR] No se pudo construir el dataset para '{task}': {e}")
        del model
        return None

    n = len(dataset)
    if n == 0:
        print(f"  [WARN] Dataset vacío para la tarea '{task}'.")
        del model
        return None
    print(f"  WSIs a procesar: {n}")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # ── Resume: omitir WSIs ya calculadas en una ejecución previa ──────────
    csv_path = os.path.join(output_path, f"{task}_srg.csv")
    processed_stems = set()
    if os.path.isfile(csv_path) and not args.overwrite:
        prev_df = pd.read_csv(csv_path)
        processed_stems = set(prev_df["wsi_id"].astype(str))
        print(f"  [RESUME] {len(processed_stems)} WSIs ya procesadas, se omiten.")
    elif os.path.isfile(csv_path) and args.overwrite:
        os.remove(csv_path)

    fieldnames = [
        "wsi_id", "n_patches", "true_label", "pred_label", "correct",
        "target_class_used", "prob_full_bag", "aupc_ascending",
        "aupc_descending", "srg", "srg_random_mean", "srg_random_std",
        "n_random_repeats",
    ]
    write_header = not os.path.isfile(csv_path)
    rng = np.random.default_rng(args.seed)

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_idx, (features_list, labels_batch, wsi_ids, coords_list) in enumerate(loader):
            features = features_list[0]
            true_label = int(labels_batch[0].item())
            wsi_id = wsi_ids[0]

            if use_precomputed:
                h5_path = df_valid.iloc[batch_idx]["features"]
            else:
                h5_path = wsi_id
            wsi_stem = os.path.splitext(os.path.basename(h5_path))[0]

            if wsi_stem in processed_stems:
                continue

            print(f"\n    WSI {batch_idx + 1}/{n} — {wsi_stem}")

            if true_label == -1:
                print(f"      [SKIP] Etiqueta -1 para la tarea '{task}'.")
                continue

            t0 = time.time()
            try:
                pred_label, probs, weights = run_inference_single_wsi(
                    model, features, device, use_precomputed
                )

                target_class = pred_label if args.target_class == "predicted" else true_label

                srg, aupc_asc, aupc_desc, prob_full = compute_srg_for_wsi(
                    model, features, device, use_precomputed,
                    weights, target_class, args.n_groups,
                )

                srg_rand_mean, srg_rand_std = float("nan"), float("nan")
                if args.n_random_repeats > 0:
                    srg_rand_mean, srg_rand_std = _random_baseline_srg(
                        model, features, device, use_precomputed, target_class,
                        args.n_groups, args.n_random_repeats, rng, prob_full,
                    )
            except Exception as e:
                print(f"      [ERROR] Cálculo de SRG fallido: {e}")
                continue

            elapsed = time.time() - t0
            print(
                f"      pred={pred_label}  true={true_label}  "
                f"target_class={target_class}  n_patches={int(features.shape[0])}"
            )
            print(
                f"      SRG={srg:.4f}  (AUPC_asc={aupc_asc:.4f}, "
                f"AUPC_desc={aupc_desc:.4f})  [{elapsed:.1f}s]"
            )
            if args.n_random_repeats > 0:
                print(f"      SRG_random={srg_rand_mean:.4f} ± {srg_rand_std:.4f}")

            writer.writerow({
                "wsi_id": wsi_stem,
                "n_patches": int(features.shape[0]),
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": int(pred_label == true_label),
                "target_class_used": target_class,
                "prob_full_bag": prob_full,
                "aupc_ascending": aupc_asc,
                "aupc_descending": aupc_desc,
                "srg": srg,
                "srg_random_mean": srg_rand_mean,
                "srg_random_std": srg_rand_std,
                "n_random_repeats": args.n_random_repeats,
            })
            fcsv.flush()

            if (batch_idx + 1) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = summarize_task(csv_path, task, args.n_random_repeats)
    summary_path = os.path.join(output_path, f"{task}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Resultados guardados en: {csv_path}")
    print(f"  Resumen guardado en:     {summary_path}")
    print(
        f"  SRG medio: {summary['srg_mean']:.4f}  "
        f"(positivo en {summary['pct_srg_positive']:.1f}% de las WSIs)"
    )
    if "srg_random_mean" in summary:
        print(
            f"  SRG medio random: {summary['srg_random_mean']:.4f}  |  "
            f"tamaño de efecto vs. random (r): {summary['wilcoxon_effect_size_vs_random']:.3f}  "
            f"(p={summary['wilcoxon_p_value_vs_random']:.4g})"
        )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_multiple_seed(args.seed)

    parent = os.path.dirname(os.path.abspath(args.test_results_path))
    if args.output_path is None:
        args.output_path = os.path.join(parent, "srg_results")
    os.makedirs(args.output_path, exist_ok=True)

    logger = TeeLogger(os.path.join(args.output_path, "log.txt"))
    logger.start()
    try:
        _run(args)
    finally:
        logger.stop()


def _run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:          {device}")
    print(f"Salida:               {args.output_path}")
    print(f"test_results:         {args.test_results_path}")
    print(f"Clase objetivo:       {args.target_class}")
    print(f"Grupos percentiles:   {args.n_groups}")
    print(f"Repeticiones random:  {args.n_random_repeats}")

    test_df = load_test_split(args)

    all_summaries = {}
    for task in args.tasks:
        summary = process_task_srg(task, args, test_df, device, args.output_path)
        if summary is not None:
            all_summaries[task] = summary

    print(f"\n{'='*65}")
    print("  RESUMEN FINAL — SRG medio por tarea")
    print(f"{'='*65}")
    for task, summary in all_summaries.items():
        line = f"  {task:20s}  SRG medio = {summary['srg_mean']:.4f}"
        if "srg_random_mean" in summary:
            line += f"  |  SRG random = {summary['srg_random_mean']:.4f}"
            line += f"  |  r vs random = {summary['wilcoxon_effect_size_vs_random']:.3f}"
        print(line)
    print(f"\n  Resultados completos en: {args.output_path}")


if __name__ == "__main__":
    main()
