"""
generate_heatmaps.py
--------------------------
Genera mapas de calor para WSIs usando modelos entrenados y guardados con el script test.py.

Para cada WSI seleccionada y cada tarea produce tres imágenes:
  1. WSI reconstruida (rescalada a resolución visualizable)
  2. Máscara de calor (parches coloreados por peso)
  3. Fusión de WSI + máscara de calor

Métodos de explicación soportados
----------------------------------
  attention  → Pesos de atención internos del modelo.
               Solo disponible para modelos abMIL-bag-level, abMIL-instance-level y transMIL.

  single     → Pasa cada parche de forma individual por el modelo y usa la probabilidad
               softmax de la clase predicha como score de relevancia.
               Compatible con cualquier arquitectura MIL que soporte forward_from_embeddings
               o forward estándar.

Nuevas funcionalidades
-----------------------
  --use_dev_set         → Activa el procesado del dev_set (por defecto False).
  --compute_srg         → Calcula el SRG (Symmetric Relevance Gain) por WSI.
                          Si está activo, al final se guarda un CSV con todos los SRGs
                          y se realiza un análisis estadístico entre métodos de explicación.
  --explanation_method  → Acepta una lista de métodos: ej. attention single.
                          Se ejecutarán en batería todos los métodos indicados.

Estructura de salida
--------------------
  <output_path>/
    log.txt
    srg_results.csv               ← (si --compute_srg) Una fila por (wsi, task, split, method)
    srg_statistical_analysis.txt  ← (si --compute_srg y >1 método) Análisis estadístico
    dev_set/
      <wsi_stem>/
        <task>_<method>_success_pred0_true0/
          wsi.<fmt>
          heatmap.<fmt>
          overlay.<fmt>
        ...
    test_set/
      ...

Uso mínimo:
  python generate_heatmaps.py --test_results_path /workspace/results/my_run/test_results

Ejemplo con SRG:
  python generate_heatmaps.py --test_results_path /workspace/results/my_run/test_results --compute_srg

Nota
----
Este script comparte todo el scaffolding (splits, dataset, carga de modelo,
iteración de WSIs, forward de la bag) con predict_all_wsis.py a través de
wsi_pipeline_common.py. Solo implementa aquí lo específico: scores de
relevancia por parche, reconstrucción visual, SRG y análisis estadístico.
"""

import argparse
import gc
import itertools
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# Imports del proyecto
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    bcnb_images_folder_path,
    bcnb_labels_df_path,
    bcnb_embeddings_path,
)
from utils.train_utils import set_multiple_seed
from utils.explainability.compute_srg import compute_srg
from wsi_pipeline_common import (
    TeeLogger,
    load_splits,
    build_dataset,
    iterate_wsi_dataset,
    for_each_task_and_method,
    forward_bag,
)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# MIL compatibles con pesos de atención internos (método 'attention')
SUPPORTED_MIL_ATTENTION = [
    "abMIL-bag-level",
    "abMIL-instance-level",
    "transMIL",
]

# MIL compatibles con el método 'single' (cualquiera que soporte forward_from_embeddings o forward estándar)
SUPPORTED_MIL_SINGLE = [
    "max-instance-level",
    "max-bag-level",
    "mean-instance-level",
    "mean-bag-level",
    "abMIL-instance-level",
    "abMIL-bag-level",
    "transMIL",
]

# Métodos de explicación soportados
EXPLANATION_METHODS = ["attention", "single"]

# Número de clases por tarea
TASKS_NUM_CLASSES = {
    "er": 2,
    "pr": 2,
    "her2": 2,
    "ki67": 2,
    "molecular_subtype": 4,
}


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera mapas de calor para WSIs. Soporta dos métodos de explicación:\n"
            "  attention  → pesos de atención internos del modelo.\n"
            "  single     → pesos obtenidos al pasar cada parche por el modelo individualmente.\n\n"
            "Ambos métodos pueden lanzarse en batería con --explanation_method attention single.\n"
            "El flag --compute_srg activa el cálculo del SRG y el análisis estadístico."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--test_results_path", type=str, required=True,
        help="Ruta a la carpeta de resultados de test.py (contiene subcarpetas por tarea).")

    parser.add_argument("--images_folder", type=str, default=bcnb_images_folder_path,
        help="Ruta a la carpeta con los ficheros .h5 de parches de las WSIs.")

    parser.add_argument("--embeddings_folder", type=str, default=bcnb_embeddings_path,
        help="Ruta raíz de embeddings precomputados: <embeddings_folder>/<encoder>/<wsi>.h5")

    parser.add_argument("--df_path", type=str, default=bcnb_labels_df_path,
        help="Ruta al fichero de etiquetas (.xlsx).")

    parser.add_argument("--dev_df_path", type=str, default=None,
        help="CSV con el dev set. Por defecto: <test_results_path>/../dev_df.csv")

    parser.add_argument("--test_df_path", type=str, default=None,
        help="CSV con el test set. Por defecto: <test_results_path>/../test_df.csv")

    parser.add_argument("--output_path", type=str, default=None,
        help="Carpeta de salida. Por defecto: <parent de test_results_path>/heatmaps")
    parser.add_argument("--tasks", type=str, nargs="+",
        default=["er", "pr", "her2", "ki67", "molecular_subtype"],
        choices=["er", "pr", "her2", "ki67", "molecular_subtype"],
        help="Tareas para las que generar mapas de calor.")

    parser.add_argument("--n_images", type=int, default=10,
        help="Número de WSIs a procesar por split para generar imágenes.")

    parser.add_argument("--n_srg", type=int, default=None,
        help="Número de WSIs a usar para calcular el SRG. Si es None se usan todas.")

    parser.add_argument("--max_output_pixels", type=int, default=4_000_000,
        help="Máximo de píxeles de la imagen WSI reconstruida.")

    parser.add_argument("--output_format", type=str, default="jpg", choices=["jpg", "png"])

    parser.add_argument("--overlay_alpha", type=float, default=0.45,
        help="Opacidad de la máscara de calor en la imagen de fusión (0.0-1.0).")

    parser.add_argument("--explanation_method", type=str, nargs="+",
        default=EXPLANATION_METHODS, choices=EXPLANATION_METHODS,
        help="Método(s) para calcular los scores de relevancia por parche.")

    parser.add_argument("--use_dev_set", action="store_true",
        help="Si se activa, procesa también el dev_set además del test_set.")

    parser.add_argument("--compute_srg", action="store_true", default=False,
        help="Si se activa, calcula el SRG de cada WSI y guarda srg_results.csv.")

    parser.add_argument("--srg_n_bins", type=int, default=100)
    
    parser.add_argument("--seed", type=int, default=98)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# COMPATIBILIDAD MIL x MÉTODO
# ──────────────────────────────────────────────────────────────────────────────

def _compat_check(mil_mode: str, method: str) -> None:
    """Lanza ValueError si la arquitectura MIL no soporta el método de explicación."""
    if method == "attention" and mil_mode not in SUPPORTED_MIL_ATTENTION:
        raise ValueError(
            f"Arquitectura MIL '{mil_mode}' no soportada para el método 'attention'. "
            f"Compatibles: {SUPPORTED_MIL_ATTENTION}. Prueba con --explanation_method single."
        )
    if method == "single" and mil_mode not in SUPPORTED_MIL_SINGLE:
        raise ValueError(
            f"Arquitectura MIL '{mil_mode}' no listada en SUPPORTED_MIL_SINGLE. "
            "Añádela si es compatible con forward estándar."
        )


# ──────────────────────────────────────────────────────────────────────────────
# COLORMAP
# ──────────────────────────────────────────────────────────────────────────────

def _heatmap_colormap(weight_normalized: float) -> tuple:
    """Convierte un peso normalizado [0,1] a color RGB (azul→cian→verde→amarillo→rojo)."""
    w = float(np.clip(weight_normalized, 0.0, 1.0))
    if w < 0.25:
        t = w / 0.25
        return (0, int(t * 255), 255)
    elif w < 0.5:
        t = (w - 0.25) / 0.25
        return (0, 255, int((1 - t) * 255))
    elif w < 0.75:
        t = (w - 0.5) / 0.25
        return (int(t * 255), 255, 0)
    else:
        t = (w - 0.75) / 0.25
        return (255, int((1 - t) * 255), 0)


# ──────────────────────────────────────────────────────────────────────────────
# SELECCIÓN DE WSIs POR SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def select_wsis(df_split: pd.DataFrame, n_images: int) -> pd.DataFrame:
    """Primeras n_images filas del split. Fijas para todas las tareas."""
    n = min(n_images, len(df_split))
    selected = df_split.head(n).reset_index(drop=True)
    print(f"  → {n} WSIs seleccionadas para imágenes (de {len(df_split)} disponibles)")
    return selected


def select_wsis_srg(df_split: pd.DataFrame, n_srg: int | None) -> pd.DataFrame:
    """Selecciona las WSIs a usar para calcular el SRG. Si n_srg es None, usa todas."""
    if n_srg is None:
        selected = df_split.reset_index(drop=True)
        print(f"  → {len(selected)} WSIs seleccionadas para SRG (todas las disponibles)")
    else:
        n = min(n_srg, len(df_split))
        selected = df_split.head(n).reset_index(drop=True)
        print(f"  → {n} WSIs seleccionadas para SRG (de {len(df_split)} disponibles)")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# SCORES DE RELEVANCIA POR MÉTODO (usan forward_bag de wsi_pipeline_common)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_attention_weights(model, features, device, use_precomputed) -> tuple:
    """
    Usa los pesos de atención internos del modelo (weights_list[0] del forward).

    Returns
    -------
    pred_label : int
    probs      : ndarray (num_classes,)
    weights    : ndarray (N_patches,)
    """
    out = forward_bag(model, features, device, use_precomputed)
    weights = out.weights_list[0].cpu().numpy()
    return out.pred_label, out.probs, weights


def _compute_single_weights(model, features, device, use_precomputed) -> tuple:
    """
    Cada parche se evalúa como una bolsa de tamaño 1; se usa la probabilidad
    de la clase predicha por la bolsa completa como score de ese parche.

    Complejidad: O(N_patches) forwards individuales.

    Returns
    -------
    pred_label : int     ← clase predicha con la bag completa
    probs      : ndarray (num_classes,)
    weights    : ndarray (N_patches,)  ← scores Single por parche
    """
    N = features.shape[0]
    out = forward_bag(model, features, device, use_precomputed)

    scores = []
    for k in range(N):
        single_patch = features[k : k + 1]
        out_k = forward_bag(model, single_patch, device, use_precomputed)
        scores.append(float(out_k.probs[out.pred_label]))

    weights = np.array(scores, dtype=np.float32)
    return out.pred_label, out.probs, weights


def run_inference_single_wsi(model, features, device, use_precomputed, explanation_method) -> tuple:
    """Dispatcher: calcula los scores de relevancia por parche según el método."""
    if explanation_method == "attention":
        return _compute_attention_weights(model, features, device, use_precomputed)
    elif explanation_method == "single":
        return _compute_single_weights(model, features, device, use_precomputed)
    raise ValueError(f"explanation_method desconocido: '{explanation_method}'.")


# ──────────────────────────────────────────────────────────────────────────────
# RECONSTRUCCIÓN VISUAL DE LA WSI
# ──────────────────────────────────────────────────────────────────────────────

def _read_patch_images_from_h5(h5_path: str) -> list:
    """Lee las imágenes crudas de un H5 de parches como lista de PIL.Image."""
    with h5py.File(h5_path, "r") as h5:
        imgs_raw = h5["imgs"][:]
    return [Image.fromarray(img) for img in imgs_raw]


def reconstruct_wsi(pil_imgs: list, coords: np.ndarray, max_output_pixels: int = 4_000_000) -> tuple:
    """Reconstruye la WSI colocando parches según sus coordenadas absolutas."""
    if not pil_imgs:
        raise ValueError("No hay parches para reconstruir la WSI.")

    patch_size = max(pil_imgs[0].size)
    coords_arr = np.asarray(coords)
    x_min, y_min = coords_arr[:, 0].min(), coords_arr[:, 1].min()
    x_max, y_max = coords_arr[:, 0].max(), coords_arr[:, 1].max()

    canvas_w = int(x_max - x_min + patch_size)
    canvas_h = int(y_max - y_min + patch_size)
    total = canvas_w * canvas_h

    scale = (max_output_pixels / total) ** 0.5 if total > max_output_pixels else 1.0
    out_w = max(1, int(canvas_w * scale))
    out_h = max(1, int(canvas_h * scale))
    patch_size_out = max(1, int(patch_size * scale))

    canvas = Image.new("RGB", (out_w, out_h), color=(240, 240, 240))
    for img_pil, (x, y) in zip(pil_imgs, coords_arr):
        px = min(int((x - x_min) * scale), out_w - patch_size_out)
        py = min(int((y - y_min) * scale), out_h - patch_size_out)
        canvas.paste(img_pil.resize((patch_size_out, patch_size_out), Image.BICUBIC), (px, py))

    return canvas, scale, (x_min, y_min), patch_size_out


def build_heatmap_mask(coords, weights, canvas_size, patch_size_out, scale, origin) -> Image.Image:
    """Máscara RGBA: cada parche coloreado por su score de relevancia."""
    out_w, out_h = canvas_size
    x_min, y_min = origin

    w_min, w_max = weights.min(), weights.max()
    weights_norm = (
        (weights - w_min) / (w_max - w_min) if w_max > w_min
        else np.full_like(weights, 0.5)
    )

    mask = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    for (x, y), w_norm in zip(coords, weights_norm):
        px = min(int((x - x_min) * scale), out_w - patch_size_out)
        py = min(int((y - y_min) * scale), out_h - patch_size_out)
        r, g, b = _heatmap_colormap(w_norm)
        alpha = int(60 + w_norm * 195)
        mask.paste(Image.new("RGBA", (patch_size_out, patch_size_out), (r, g, b, alpha)), (px, py))

    return mask


def build_overlay(wsi_img: Image.Image, mask: Image.Image, alpha: float) -> Image.Image:
    """Fusiona WSI + máscara de calor."""
    r, g, b, a = mask.split()
    mask_adj = Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * alpha))))
    return Image.alpha_composite(wsi_img.convert("RGBA"), mask_adj).convert("RGB")


def save_image(img: Image.Image, path: str, fmt: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "jpg":
        img.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)
    else:
        img.save(path, format="PNG", optimize=True)


# ──────────────────────────────────────────────────────────────────────────────
# SRG
# ──────────────────────────────────────────────────────────────────────────────

def compute_and_collect_srg(model, features, weights, pred_label, device, n_bins) -> float:
    """Llama a compute_srg con los embeddings y los pesos del método de explicación."""
    return compute_srg(
        model=model,
        embeddings=features,
        patch_scores=weights,
        target_class=pred_label,
        n_bins=n_bins,
        device=device,
    )


def statistical_analysis_srg(srg_df: pd.DataFrame, output_path: str):
    """
    Análisis estadístico de los SRG entre métodos de explicación.
      - Estadísticos descriptivos por (task, split, method).
      - 2 métodos → Wilcoxon signed-rank (emparejado por WSI).
      - >2 métodos → Kruskal-Wallis + post-hoc Mann-Whitney con Bonferroni.
    Guarda el resultado en <output_path>/srg_statistical_analysis.txt.
    """
    methods = sorted(srg_df["method"].unique())
    n_methods = len(methods)

    if n_methods < 2:
        print("  [INFO] Solo un método de explicación — análisis estadístico omitido.")
        return

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  ANÁLISIS ESTADÍSTICO DEL SRG ENTRE MÉTODOS DE EXPLICACIÓN")
    report_lines.append("=" * 70)
    report_lines.append(f"\nMétodos comparados: {methods}")
    report_lines.append(f"Total de registros SRG: {len(srg_df)}\n")

    report_lines.append("─" * 70)
    report_lines.append("ESTADÍSTICOS DESCRIPTIVOS  (por task × split × method)")
    report_lines.append("─" * 70)

    desc = (
        srg_df.groupby(["task", "split", "method"])["srg"]
        .agg(["count", "mean", "std", "median",
              lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "min", "max"])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
    )
    report_lines.append(desc.to_string())
    report_lines.append("")

    report_lines.append("─" * 70)
    if n_methods == 2:
        report_lines.append(
            "TEST WILCOXON SIGNED-RANK  (H0: distribuciones iguales, α=0.05)\n"
            "Prueba emparejada por WSI dentro de cada (task, split)."
        )
    else:
        report_lines.append(
            "TEST KRUSKAL-WALLIS  (H0: distribuciones iguales, α=0.05)\n"
            "Seguido de Mann-Whitney con corrección Bonferroni."
        )
    report_lines.append("─" * 70)

    for (task, split), group_df in srg_df.groupby(["task", "split"]):
        report_lines.append(f"\n  [{task}] — split: {split}")

        if n_methods == 2:
            m1, m2 = methods
            g1 = group_df[group_df["method"] == m1].sort_values("wsi_stem")
            g2 = group_df[group_df["method"] == m2].sort_values("wsi_stem")

            common = set(g1["wsi_stem"]) & set(g2["wsi_stem"])
            if len(common) < 5:
                report_lines.append(
                    f"    [SKIP] Menos de 5 WSIs emparejadas ({len(common)}). Test no aplicable."
                )
                continue

            g1c = g1[g1["wsi_stem"].isin(common)].set_index("wsi_stem")["srg"]
            g2c = g2[g2["wsi_stem"].isin(common)].set_index("wsi_stem")["srg"]
            g1c, g2c = g1c.align(g2c, join="inner")

            stat, p = stats.wilcoxon(g1c.values, g2c.values, alternative="two-sided")
            sig = "✓ SIGNIFICATIVO" if p < 0.05 else "✗ no significativo"
            report_lines.append(f"    Wilcoxon: W={stat:.4f}  p={p:.4e}  n={len(g1c)}  → {sig}")
            diff = g1c - g2c
            report_lines.append(
                f"    Diferencia media ({m1} − {m2}): {diff.mean():.4f}  (mediana: {diff.median():.4f})"
            )
            report_lines.append(f"    Medias — {m1}: {g1c.mean():.4f}  |  {m2}: {g2c.mean():.4f}")

        else:
            groups = [group_df[group_df["method"] == m]["srg"].values for m in methods]
            if any(len(g) < 3 for g in groups):
                report_lines.append("    [SKIP] Algún grupo tiene < 3 muestras.")
                continue

            stat_kw, p_kw = stats.kruskal(*groups)
            sig_kw = "✓ SIGNIFICATIVO" if p_kw < 0.05 else "✗ no significativo"
            report_lines.append(f"    Kruskal-Wallis: H={stat_kw:.4f}  p={p_kw:.4e}  → {sig_kw}")

            pairs = list(itertools.combinations(methods, 2))
            alpha_bonf = 0.05 / len(pairs)
            report_lines.append(f"    Post-hoc Mann-Whitney (α Bonferroni = {alpha_bonf:.4e}):")
            for m1, m2 in pairs:
                v1 = group_df[group_df["method"] == m1]["srg"].values
                v2 = group_df[group_df["method"] == m2]["srg"].values
                u, p_mw = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                sig_mw = "✓" if p_mw < alpha_bonf else "✗"
                report_lines.append(f"      {m1} vs {m2}: U={u:.1f}  p={p_mw:.4e}  {sig_mw}")

    report_lines.append("\n" + "=" * 70)

    analysis_path = os.path.join(output_path, "srg_statistical_analysis.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  [SRG] Análisis estadístico guardado: {analysis_path}")
    for line in report_lines:
        print(line)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE POR TAREA, MÉTODO Y SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def process_task_on_split(
    split_name: str,
    df_images: pd.DataFrame,
    df_srg: pd.DataFrame | None,
    task: str,
    model,
    use_precomputed: bool,
    emb_folder_enc: str | None,
    device: torch.device,
    output_split_path: str,
    max_output_pixels: int,
    output_format: str,
    overlay_alpha: float,
    explanation_method: str,
    compute_srg_flag: bool,
    srg_n_bins: int,
) -> list[dict]:
    """
    Procesa las WSIs del split con el modelo de `task`.
      - Genera imágenes para las WSIs en df_images.
      - Calcula SRG para las WSIs en df_srg (si compute_srg_flag=True).
    Ambas selecciones son independientes; se itera sobre la unión.
    """
    fmt = output_format.lower()
    srg_records: list[dict] = []

    print(f"\n  [{split_name}] Tarea '{task}' — método '{explanation_method}'")

    if compute_srg_flag and not use_precomputed:
        print(
            "  [WARN] SRG omitido: compute_srg requiere embeddings precomputados "
            "porque compute_srg() llama a forward_from_embeddings()."
        )
        compute_srg_flag = False

    if compute_srg_flag and df_srg is not None:
        df_all = (
            pd.concat([df_images, df_srg])
            .drop_duplicates(subset=["features"])
            .reset_index(drop=True)
        )
    else:
        df_all = df_images.copy().reset_index(drop=True)

    image_stems = set(
        os.path.splitext(os.path.basename(row["features"]))[0]
        for _, row in df_images.iterrows()
    )
    srg_stems = set()
    if compute_srg_flag and df_srg is not None:
        srg_stems = set(
            os.path.splitext(os.path.basename(row["features"]))[0]
            for _, row in df_srg.iterrows()
        )

    print(f"  [{split_name}] Construyendo dataset...")
    try:
        dataset, df_valid = build_dataset(df_all, task, use_precomputed, emb_folder_enc)
    except Exception as e:
        print(f"  [ERROR] No se pudo construir el dataset para '{task}': {e}")
        return srg_records

    n = len(dataset)
    if n == 0:
        print(f"  [WARN] Dataset vacío para tarea '{task}' en {split_name}.")
        return srg_records

    print(f"  [{split_name}] Tarea '{task}' → {n} WSIs totales")

    samples = iterate_wsi_dataset(dataset, df_valid, use_precomputed)

    for batch_idx, sample in enumerate(samples):
        wsi_stem = sample.wsi_stem
        do_images = wsi_stem in image_stems
        do_srg = compute_srg_flag and (wsi_stem in srg_stems)

        if not do_images and not do_srg:
            continue

        print(
            f"\n    WSI {batch_idx + 1}/{n} — {wsi_stem}  (true_{task}={sample.true_label})"
            f"  [imagenes={do_images}, srg={do_srg}]"
        )

        if sample.true_label == -1:
            print(f"      [SKIP] Etiqueta -1 para tarea '{task}'.")
            continue

        if explanation_method == "single":
            print(f"      [Single] Ejecutando {sample.features.shape[0]} forwards individuales...")

        try:
            pred_label, probs, weights = run_inference_single_wsi(
                model=model,
                features=sample.features,
                device=device,
                use_precomputed=use_precomputed,
                explanation_method=explanation_method,
            )
        except Exception as e:
            print(f"      [ERROR] Inferencia fallida: {e}")
            continue

        outcome = "success" if pred_label == sample.true_label else "failure"
        print(f"      pred={pred_label}  true={sample.true_label}  outcome={outcome}")
        print(f"      probs={[f'{p:.3f}' for p in probs]}")
        print(
            f"      scores ({explanation_method}) — "
            f"min={weights.min():.4f}  max={weights.max():.4f}  mean={weights.mean():.4f}"
        )

        # ── SRG ──────────────────────────────────────────────────────────────
        if do_srg:
            try:
                srg_value = compute_and_collect_srg(
                    model=model,
                    features=sample.features.to(device),
                    weights=weights,
                    pred_label=pred_label,
                    device=device,
                    n_bins=srg_n_bins,
                )
                print(f"      SRG ({explanation_method}): {srg_value:.4f}")
                srg_records.append({
                    "wsi_stem": wsi_stem,
                    "task": task,
                    "split": split_name,
                    "method": explanation_method,
                    "srg": srg_value,
                    "pred_label": pred_label,
                    "true_label": sample.true_label,
                    "outcome": outcome,
                })
            except Exception as e:
                print(f"      [ERROR] SRG fallido: {e}")

        # ── Imágenes (solo para las WSIs seleccionadas) ────────────────────────
        if not do_images:
            continue

        coords = sample.coords
        if coords is None:
            print("      [WARN] coords=None desde el dataset, releyendo del H5...")
            try:
                with h5py.File(sample.h5_path, "r") as h5:
                    coords = h5["coords"][:]
            except Exception as e:
                print(f"      [ERROR] No se pudieron obtener coords: {e}")
                continue

        coords_arr = np.asarray(coords)

        try:
            pil_imgs = _read_patch_images_from_h5(sample.h5_path)
        except Exception as e:
            print(f"      [ERROR] Lectura de imágenes PIL fallida: {e}")
            continue

        if len(pil_imgs) != len(coords_arr):
            print(
                f"      [WARN] Discrepancia parches/coords: "
                f"{len(pil_imgs)} imgs vs {len(coords_arr)} coords. Usando mínimo."
            )
            n_min = min(len(pil_imgs), len(coords_arr))
            pil_imgs = pil_imgs[:n_min]
            coords_arr = coords_arr[:n_min]
            weights_img = weights[:n_min]
        else:
            weights_img = weights

        try:
            wsi_img, scale, origin, patch_size_out = reconstruct_wsi(
                pil_imgs, coords_arr, max_output_pixels
            )
        except Exception as e:
            print(f"      [ERROR] Reconstrucción WSI fallida: {e}")
            continue

        print(f"      Canvas={wsi_img.size}  escala={scale:.4f}  parche_px={patch_size_out}")

        try:
            heatmap_mask = build_heatmap_mask(
                coords_arr, weights_img, wsi_img.size, patch_size_out, scale, origin
            )
        except Exception as e:
            print(f"      [ERROR] Máscara de calor fallida: {e}")
            continue

        try:
            overlay = build_overlay(wsi_img, heatmap_mask, overlay_alpha)
        except Exception as e:
            print(f"      [ERROR] Overlay fallido: {e}")
            continue

        folder_tag = f"{task}_{explanation_method}_{outcome}_pred{pred_label}_true{sample.true_label}"
        out_dir = os.path.join(output_split_path, wsi_stem, folder_tag)
        os.makedirs(out_dir, exist_ok=True)

        save_image(wsi_img, os.path.join(out_dir, f"wsi.{fmt}"), fmt)
        save_image(heatmap_mask.convert("RGB"), os.path.join(out_dir, f"heatmap.{fmt}"), fmt)
        save_image(overlay, os.path.join(out_dir, f"overlay.{fmt}"), fmt)

        print(f"      Guardado: {out_dir}")

        del pil_imgs, wsi_img, heatmap_mask, overlay
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return srg_records


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_multiple_seed(args.seed)

    parent = os.path.dirname(os.path.abspath(args.test_results_path))
    if args.output_path is None:
        args.output_path = os.path.join(parent, "heatmaps")
    os.makedirs(args.output_path, exist_ok=True)

    with TeeLogger(os.path.join(args.output_path, "log.txt")):
        _run(args)


def _run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:            {device}")
    print(f"Salida:                 {args.output_path}")
    print(f"test_results:           {args.test_results_path}")
    print(f"Métodos explicación:    {args.explanation_method}")
    print(f"Procesar dev_set:       {args.use_dev_set}")
    print(f"Calcular SRG:           {args.compute_srg}")
    if args.compute_srg:
        n_srg_str = str(args.n_srg) if args.n_srg is not None else "todas"
        print(f"  WSIs para SRG:        {n_srg_str}")
        print(f"  SRG n_bins:           {args.srg_n_bins}")

    dev_df, test_df = load_splits(
        args.test_results_path, args.use_dev_set, args.dev_df_path, args.test_df_path
    )

    print(f"\nSeleccionando {args.n_images} WSIs fijas por split para generar imágenes...")
    dev_images_selected = None
    if args.use_dev_set and dev_df is not None:
        print("  dev_set:")
        dev_images_selected = select_wsis(dev_df, args.n_images)
    print("  test_set:")
    test_images_selected = select_wsis(test_df, args.n_images)

    dev_srg_selected = None
    test_srg_selected = None
    if args.compute_srg:
        print("\nSeleccionando WSIs para cálculo de SRG...")
        if args.use_dev_set and dev_df is not None:
            print("  dev_set:")
            dev_srg_selected = select_wsis_srg(dev_df, args.n_srg)
        print("  test_set:")
        test_srg_selected = select_wsis_srg(test_df, args.n_srg)

    splits = {"test_set": (test_images_selected, test_srg_selected)}
    if args.use_dev_set and dev_df is not None:
        splits["dev_set"] = (dev_images_selected, dev_srg_selected)

    print(f"\nModo embeddings precomputados: {args.embeddings_folder is not None}")

    all_srg_records: list[dict] = []

    for ctx in for_each_task_and_method(
        test_results_path=args.test_results_path,
        tasks=args.tasks,
        num_classes_fn=lambda t: TASKS_NUM_CLASSES[t],
        device=device,
        embeddings_folder=args.embeddings_folder,
        methods=args.explanation_method,
        compat_check=_compat_check,
    ):
        print(f"\n{'='*65}")
        print(f"  TAREA: {ctx.task}  |  Método: {ctx.method}")
        print(f"{'='*65}")

        if ctx.method == "single" and not ctx.use_precomputed:
            print(
                "  [WARN] Método 'single' sin embeddings precomputados puede ser muy lento. "
                "Se recomienda usar --embeddings_folder."
            )

        for split_name, (df_images, df_srg) in splits.items():
            srg_records = process_task_on_split(
                split_name=split_name,
                df_images=df_images,
                df_srg=df_srg,
                task=ctx.task,
                model=ctx.model,
                use_precomputed=ctx.use_precomputed,
                emb_folder_enc=ctx.emb_folder_enc,
                device=device,
                output_split_path=os.path.join(args.output_path, split_name),
                max_output_pixels=args.max_output_pixels,
                output_format=args.output_format,
                overlay_alpha=args.overlay_alpha,
                explanation_method=ctx.method,
                compute_srg_flag=args.compute_srg,
                srg_n_bins=args.srg_n_bins,
            )
            all_srg_records.extend(srg_records)

    if args.compute_srg and all_srg_records:
        srg_df = pd.DataFrame(all_srg_records)
        srg_csv_path = os.path.join(args.output_path, "srg_results.csv")
        srg_df.to_csv(srg_csv_path, index=False)
        print(f"\n  [SRG] Resultados guardados en: {srg_csv_path}")
        print(f"  [SRG] Total de registros: {len(srg_df)}")
        print(srg_df.groupby(["task", "split", "method"])["srg"].describe().to_string())

        statistical_analysis_srg(srg_df, args.output_path)
    elif args.compute_srg:
        print("\n  [SRG] No se recopilaron registros SRG.")

    print(f"\n{'='*65}")
    print(f"  Mapas generados en:  {args.output_path}")
    print(f"  Log guardado en:     {os.path.join(args.output_path, 'log.txt')}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
