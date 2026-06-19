"""
generate_attention_maps.py
--------------------------
Genera mapas de calor para WSIs usando modelos entrenados y guardados en test.py.
Solo funciona con modelos que proporcionen pesos de atención por parche.

Para cada WSI seleccionada y cada tarea produce tres imágenes:
  1. WSI reconstruida (rescalada a resolución visualizable)
  2. Máscara de calor (parches coloreados por peso)
  3. Fusión de WSI + máscara de calor

Métodos de explicación soportados
----------------------------------
  attention  → Pesos de atención internos del modelo (comportamiento original).
               Solo disponible para modelos abMIL-bag-level, abMIL-instance-level y transMIL.

  single     → Método "Single" de Early et al. / xMIL (paper: "Beyond Attention Heatmaps").
               Pasa cada parche de forma individual por el modelo y usa la probabilidad
               softmax de la clase predicha como score de relevancia.
               Compatible con cualquier arquitectura MIL que soporte forward_from_embeddings
               o forward estándar.

Estructura de salida
--------------------
  <output_path>/
    log.txt
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
  python generate_attention_maps.py --test_results_path /workspace/results/my_run/test_results

Ejemplo con método Single:
  python generate_attention_maps.py --test_results_path /workspace/results/my_run/test_results --explanation_method single
"""

import argparse
import gc
import json
import os
import sys

import h5py
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# Imports del proyecto
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    bcnb_images_folder_path,
    bcnb_labels_df_path,
    bcnb_embeddings_path,
)
from data_load import (
    Patches_WSI_Dataset,
    Embeddings_WSI_Dataset,
    collate_fn,
)
from utils.train_utils import get_model, set_multiple_seed


# ──────────────────────────────────────────────────────────────────────────────
# TEE LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class TeeLogger:
    """
    Redirige sys.stdout para que cada write() vaya tanto a la consola
    original como a un fichero de log.
    """

    def __init__(self, log_path: str):
        self.log_path     = log_path
        self._orig_stdout = None
        self._log_file    = None

    def write(self, message: str):
        self._orig_stdout.write(message)
        self._orig_stdout.flush()
        if self._log_file and not self._log_file.closed:
            self._log_file.write(message)
            self._log_file.flush()

    def flush(self):
        self._orig_stdout.flush()
        if self._log_file and not self._log_file.closed:
            self._log_file.flush()

    def start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._orig_stdout = sys.stdout
        self._log_file    = open(self.log_path, "w", encoding="utf-8")
        sys.stdout        = self
        print(f"[LOG] Escribiendo log en: {self.log_path}")

    def stop(self):
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._log_file and not self._log_file.closed:
            self._log_file.close()


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# MIL compatibles con pesos de atención internos (método 'attention')
SUPPORTED_MIL_ATTENTION = [
    "abMIL-bag-level",
    "abMIL-instance-level",
    "transMIL",
]

# Todos los MIL soportados para el método 'single' (no requiere pesos internos)
SUPPORTED_MIL_SINGLE = [
    "abMIL-bag-level",
    "abMIL-instance-level",
    "transMIL",
    # Añade aquí otras arquitecturas que tengas en tu proyecto
]

EXPLANATION_METHODS = ["attention", "single"]


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera mapas de calor para WSIs. Soporta dos métodos de explicación:\n"
            "  attention  → pesos de atención internos del modelo (original).\n"
            "  single     → método Single de xMIL: forward de un parche a la vez."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Rutas obligatorias
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
            "Embeddings_WSI_Dataset en lugar de Patches_WSI_Dataset. "
            "Se espera subcarpeta por encoder: <embeddings_folder>/<encoder>/<wsi>.h5"
        )
    )
    parser.add_argument(
        "--df_path", type=str, default=bcnb_labels_df_path,
        help="Ruta al fichero de etiquetas (.xlsx)."
    )
    parser.add_argument(
        "--dev_df_path", type=str, default=None,
        help="CSV con el dev set. Por defecto: <test_results_path>/../dev_df.csv"
    )
    parser.add_argument(
        "--test_df_path", type=str, default=None,
        help="CSV con el test set. Por defecto: <test_results_path>/../test_df.csv"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help=(
            "Carpeta de salida. "
            "Por defecto: <parent de test_results_path>/attention_maps "
            "(mismo nivel que test_results/)."
        )
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=["er", "pr", "her2", "ki67", "molecular_subtype"],
        choices=["er", "pr", "her2", "ki67", "molecular_subtype"],
        help="Tareas para las que generar mapas de calor."
    )
    parser.add_argument(
        "--n_images", type=int, default=10,
        help="Número de WSIs a procesar por split. Las mismas N WSIs se usan en todas las tareas."
    )
    parser.add_argument(
        "--max_output_pixels", type=int, default=4_000_000,
        help="Máximo de píxeles de la imagen WSI reconstruida."
    )
    parser.add_argument(
        "--output_format", type=str, default="jpg", choices=["jpg", "png"],
        help="Formato de salida de las imágenes (jpg o png)."
    )
    parser.add_argument(
        "--overlay_alpha", type=float, default=0.45,
        help="Opacidad de la máscara de calor en la imagen de fusión (0.0-1.0)."
    )

    # ── Método de explicación ─────────────────────────────────────────────────
    parser.add_argument(
        "--explanation_method", type=str, default="attention",
        choices=EXPLANATION_METHODS,
        help=(
            "Método para calcular los scores de relevancia por parche:\n"
            "  attention  → pesos de atención internos del modelo (por defecto).\n"
            "               Solo válido para abMIL-bag-level, abMIL-instance-level, transMIL.\n"
            "  single     → método Single de xMIL (Early et al., 2022 / Jamshidi et al., 2026).\n"
            "               Hace un forward independiente de cada parche y usa softmax(pred)[clase]\n"
            "               como score de relevancia. Más lento pero más fiel al modelo."
        )
    )

    # Varios
    parser.add_argument("--slide_col", type=str, default="Patient ID")
    parser.add_argument("--seed", type=int, default=98)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# COLORMAP
# ──────────────────────────────────────────────────────────────────────────────

def _heatmap_colormap(weight_normalized: float) -> tuple:
    """
    Convierte un peso normalizado [0,1] a color RGB usando un gradiente
    azul → cian → verde → amarillo → rojo.

    Args:
        weight_normalized: float en [0, 1]

    Returns:
        (R, G, B) con valores enteros en [0, 255]
    """
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
# CARGA DE SPLITS
# ──────────────────────────────────────────────────────────────────────────────

def load_splits(args) -> tuple:
    """Carga dev_df y test_df desde los CSV guardados por train.py."""
    # Los CSV están un nivel arriba de test_results_path
    parent = os.path.dirname(os.path.abspath(args.test_results_path))

    dev_path  = args.dev_df_path  or os.path.join(parent, "dev_df.csv")
    test_path = args.test_df_path or os.path.join(parent, "test_df.csv")

    if not os.path.isfile(dev_path):
        raise FileNotFoundError(
            f"dev_df.csv no encontrado en: {dev_path}\n"
            "Usa --dev_df_path para indicar la ruta correcta."
        )
    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"test_df.csv no encontrado en: {test_path}\n"
            "Usa --test_df_path para indicar la ruta correcta."
        )

    dev_df  = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    print(f"Splits cargados — dev: {len(dev_df)} WSIs  |  test: {len(test_df)} WSIs")
    return dev_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# SELECCIÓN FIJA DE WSIs POR SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def select_wsis(df_split: pd.DataFrame, n_images: int) -> pd.DataFrame:
    """Primeras n_images filas del split. Fijas para todas las tareas."""
    n        = min(n_images, len(df_split))
    selected = df_split.head(n).reset_index(drop=True)
    print(f"  → {n} WSIs seleccionadas (de {len(df_split)} disponibles)")
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATASET PARA LAS N WSIs FIJAS
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    df_selected: pd.DataFrame,
    task: str,
    use_precomputed: bool,
    emb_folder_enc: str | None,
) -> tuple:
    """
    Construye el dataset adecuado (Patches o Embeddings) para las N WSIs fijas.

    Cuando use_precomputed=True, filtra previamente las filas cuyo fichero
    de embeddings no existe, para evitar que Embeddings_WSI_Dataset falle
    en __init__ al validar los ficheros.

    Returns
    -------
    dataset    : Patches_WSI_Dataset | Embeddings_WSI_Dataset
    df_valid   : pd.DataFrame — subconjunto de df_selected realmente cargado
                 (puede ser menor si faltan embeddings)
    """
    indices = list(range(len(df_selected)))

    if use_precomputed:
        # Filtrar WSIs cuyo .h5 de embeddings no existe
        mask = []
        for i in indices:
            src  = df_selected.iloc[i]["features"]
            stem = os.path.splitext(os.path.basename(src))[0]
            path = os.path.join(emb_folder_enc, f"{stem}.h5")
            exists = os.path.isfile(path)
            if not exists:
                print(f"    [WARN] Embedding no encontrado, se omite WSI: {stem}")
            mask.append(exists)

        df_valid = df_selected[mask].reset_index(drop=True)
        indices  = list(range(len(df_valid)))

        dataset = Embeddings_WSI_Dataset(
            dataframe         = df_valid,
            indices           = indices,
            embeddings_folder = emb_folder_enc,
            task              = task,
            load_on_demand    = True,
        )
    else:
        df_valid = df_selected
        dataset  = Patches_WSI_Dataset(
            dataframe      = df_valid,
            indices        = indices,
            task           = task,
            load_on_demand = True,
        )

    return dataset, df_valid


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO ENTRENADO
# ──────────────────────────────────────────────────────────────────────────────

def load_trained_model(
    task_results_path: str,
    task: str,
    num_classes: int,
    use_precomputed: bool,
    device: torch.device,
    explanation_method: str,
) -> tuple:
    """
    Reconstruye el modelo desde best_config.json y carga los pesos de final_model/.

    Para el método 'attention', verifica que la arquitectura MIL sea compatible.
    Para el método 'single', acepta cualquier arquitectura listada en SUPPORTED_MIL_SINGLE.

    Returns
    -------
    model    : WSIClassifier en modo eval
    enc_name : str
    mil_mode : str
    """
    config_path = os.path.join(task_results_path, "best_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"best_config.json no encontrado en: {task_results_path}\n"
            "Ejecuta test.py antes de generar los mapas de calor."
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    selected    = cfg["selected"]
    hyperparams = selected["hyperparams"]
    enc_name    = hyperparams["model"]
    mil_mode    = hyperparams["mil"]

    # Validar compatibilidad con el método de explicación elegido
    if explanation_method == "attention":
        if mil_mode not in SUPPORTED_MIL_ATTENTION:
            raise ValueError(
                f"[{task}] Arquitectura MIL '{mil_mode}' no soportada para el método 'attention'. "
                f"Arquitecturas compatibles: {SUPPORTED_MIL_ATTENTION}. "
                f"Prueba con --explanation_method single."
            )
    elif explanation_method == "single":
        if mil_mode not in SUPPORTED_MIL_SINGLE:
            raise ValueError(
                f"[{task}] Arquitectura MIL '{mil_mode}' no listada en SUPPORTED_MIL_SINGLE. "
                f"Añádela si es compatible con forward estándar."
            )

    model = get_model(
        patch_encoder_type    = enc_name,
        mil_aggregation_mode  = mil_mode,
        unfreeze_backbone     = False,
        num_classes           = num_classes,
        skip_encoder          = use_precomputed,
    ).to(device)

    model_path = os.path.join(task_results_path, "final_model")
    ckpt       = os.path.join(model_path, "model.pt")

    if os.path.isfile(ckpt):
        state_dict = torch.load(ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"  [WARN] Capas faltantes: {missing}")
        if unexpected:
            print(f"  [WARN] Capas inesperadas: {unexpected}")
        print(f"  [OK] Pesos cargados desde: {ckpt}")
    else:
        print(
            f"  [WARN] model.pt no encontrado en {model_path}. "
            "Usa --save_models en test.py. Continuando con pesos aleatorios."
        )

    model.eval()
    return model, enc_name, mil_mode


# ──────────────────────────────────────────────────────────────────────────────
# SCORES DE RELEVANCIA: MÉTODO ATTENTION
# ──────────────────────────────────────────────────────────────────────────────

def _compute_attention_weights(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
) -> tuple:
    """
    Calcula los scores de relevancia usando los pesos de atención internos del modelo.

    El modelo devuelve (wsi_logits, _, weights_list) donde weights_list[0] es un
    tensor (N_patches,) con los pesos de atención normalizados por softmax.

    Returns
    -------
    pred_label : int
    probs      : ndarray (num_classes,)
    weights    : ndarray (N_patches,)  ← pesos de atención del modelo
    """
    feats = features.to(device)

    with torch.no_grad():
        if use_precomputed:
            wsi_logits, _, weights_list = model.forward_from_embeddings([feats])
        else:
            wsi_logits, _, weights_list = model([feats])

    probs      = torch.softmax(wsi_logits[0], dim=0).cpu().numpy()
    pred_label = int(np.argmax(probs))
    weights    = weights_list[0].cpu().numpy()   # (N,)

    return pred_label, probs, weights


# ──────────────────────────────────────────────────────────────────────────────
# SCORES DE RELEVANCIA: MÉTODO SINGLE (xMIL)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_single_weights(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
) -> tuple:
    """
    Calcula los scores de relevancia usando el método Single de xMIL.

    Implementación directa de Early et al. (2022) / Jamshidi et al. (2026):
      - Para cada parche k, se construye una bag de tamaño 1 con solo ese parche.
      - Se hace forward y se aplica softmax.
      - El score del parche k es softmax(logits)[clase_predicha_globalmente].

    La clase a explicar se determina haciendo primero un forward con la bag completa,
    de forma equivalente a la implementación de xAttentionMIL.explain_perturbation
    con perturbation_method='keep'.

    Complejidad: O(N_patches) forwards individuales. Puede ser lento para WSIs
    con muchos parches; se recomienda usar con embeddings precomputados.

    Returns
    -------
    pred_label : int     ← clase predicha con la bag completa
    probs      : ndarray (num_classes,)  ← probabilidades de la bag completa
    weights    : ndarray (N_patches,)    ← scores Single por parche
    """
    feats = features.to(device)   # (N, D)
    N     = feats.shape[0]

    # ── 1. Forward con la bag completa para determinar la clase a explicar ──────
    with torch.no_grad():
        if use_precomputed:
            wsi_logits_full, _, _ = model.forward_from_embeddings([feats])
        else:
            wsi_logits_full, _, _ = model([feats])

    probs_full = torch.softmax(wsi_logits_full[0], dim=0).cpu()
    pred_label = int(torch.argmax(probs_full).item())
    probs      = probs_full.numpy()

    # ── 2. Forward individual para cada parche (bag de tamaño 1) ──────────────
    scores = []
    for k in range(N):
        single_patch = feats[k : k + 1]   # (1, D) — bag con un solo parche

        with torch.no_grad():
            if use_precomputed:
                logit_k, _, _ = model.forward_from_embeddings([single_patch])
            else:
                logit_k, _, _ = model([single_patch])

        prob_k = torch.softmax(logit_k[0], dim=0).cpu()
        # Score = probabilidad softmax de la clase predicha globalmente
        scores.append(prob_k[pred_label].item())

    weights = np.array(scores, dtype=np.float32)   # (N,)

    return pred_label, probs, weights


# ──────────────────────────────────────────────────────────────────────────────
# DISPATCHER DE MÉTODOS DE EXPLICACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_single_wsi(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
    explanation_method: str,
) -> tuple:
    """
    Calcula los scores de relevancia por parche según el método seleccionado.

    Args
    ----
    model               : modelo en modo eval
    features            : tensor (N_patches, D)
    device              : dispositivo
    use_precomputed     : si True usa forward_from_embeddings
    explanation_method  : "attention" | "single"

    Returns
    -------
    pred_label : int
    probs      : ndarray (num_classes,)
    weights    : ndarray (N_patches,)
    """
    if explanation_method == "attention":
        return _compute_attention_weights(model, features, device, use_precomputed)
    elif explanation_method == "single":
        return _compute_single_weights(model, features, device, use_precomputed)
    else:
        raise ValueError(
            f"explanation_method desconocido: '{explanation_method}'. "
            f"Opciones válidas: {EXPLANATION_METHODS}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# RECONSTRUCCIÓN VISUAL DE LA WSI
# ──────────────────────────────────────────────────────────────────────────────

def _read_patch_images_from_h5(h5_path: str) -> list:
    """Lee las imágenes crudas de un H5 de parches como lista de PIL.Image."""
    with h5py.File(h5_path, "r") as h5:
        imgs_raw = h5["imgs"][:]    # (N, H, W, 3) uint8
    return [Image.fromarray(img) for img in imgs_raw]


def reconstruct_wsi(
    pil_imgs: list,
    coords: np.ndarray,
    max_output_pixels: int = 4_000_000,
) -> tuple:
    """
    Reconstruye la WSI colocando parches según sus coordenadas absolutas.

    Returns
    -------
    canvas         : PIL.Image RGB
    scale          : float
    origin         : (x_min, y_min)
    patch_size_out : int
    """
    if not pil_imgs:
        raise ValueError("No hay parches para reconstruir la WSI.")

    patch_size   = max(pil_imgs[0].size)
    coords_arr   = np.asarray(coords)
    x_min, y_min = coords_arr[:, 0].min(), coords_arr[:, 1].min()
    x_max, y_max = coords_arr[:, 0].max(), coords_arr[:, 1].max()

    canvas_w = int(x_max - x_min + patch_size)
    canvas_h = int(y_max - y_min + patch_size)
    total    = canvas_w * canvas_h

    scale          = (max_output_pixels / total) ** 0.5 if total > max_output_pixels else 1.0
    out_w          = max(1, int(canvas_w * scale))
    out_h          = max(1, int(canvas_h * scale))
    patch_size_out = max(1, int(patch_size * scale))

    canvas = Image.new("RGB", (out_w, out_h), color=(240, 240, 240))
    for img_pil, (x, y) in zip(pil_imgs, coords_arr):
        px = min(int((x - x_min) * scale), out_w - patch_size_out)
        py = min(int((y - y_min) * scale), out_h - patch_size_out)
        canvas.paste(img_pil.resize((patch_size_out, patch_size_out), Image.BICUBIC), (px, py))

    return canvas, scale, (x_min, y_min), patch_size_out


def build_heatmap_mask(
    coords: np.ndarray,
    weights: np.ndarray,
    canvas_size: tuple,
    patch_size_out: int,
    scale: float,
    origin: tuple,
) -> Image.Image:
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
        px    = min(int((x - x_min) * scale), out_w - patch_size_out)
        py    = min(int((y - y_min) * scale), out_h - patch_size_out)
        r, g, b = _heatmap_colormap(w_norm)
        alpha   = int(60 + w_norm * 195)
        mask.paste(
            Image.new("RGBA", (patch_size_out, patch_size_out), (r, g, b, alpha)),
            (px, py),
        )

    return mask


def build_overlay(wsi_img: Image.Image, mask: Image.Image, alpha: float) -> Image.Image:
    """Fusiona WSI + máscara de calor."""
    r, g, b, a = mask.split()
    mask_adj   = Image.merge("RGBA", (r, g, b, a.point(lambda v: int(v * alpha))))
    return Image.alpha_composite(wsi_img.convert("RGBA"), mask_adj).convert("RGB")


def save_image(img: Image.Image, path: str, fmt: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "jpg":
        img.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)
    else:
        img.save(path, format="PNG", optimize=True)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE POR TAREA Y SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def process_task_on_split(
    split_name: str,
    df_selected: pd.DataFrame,
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
):
    """
    Procesa las N WSIs fijas del split con el modelo de `task`.

    Estructura de salida por WSI:
        <output_split_path>/<wsi_stem>/<task>_<method>_<outcome>_pred<P>_true<T>/
            wsi.<fmt>   heatmap.<fmt>   overlay.<fmt>
    """
    fmt = output_format.lower()
    print(f"\n  [{split_name}] Tarea '{task}' — método '{explanation_method}' — "
          f"construyendo dataset...")

    # ── Construir dataset ─────────────────────────────────────────────────────
    try:
        dataset, df_valid = build_dataset(df_selected, task, use_precomputed, emb_folder_enc)
    except Exception as e:
        print(f"  [ERROR] No se pudo construir el dataset para '{task}': {e}")
        return

    n = len(dataset)
    if n == 0:
        print(f"  [WARN] Dataset vacío para tarea '{task}' en {split_name}.")
        return

    print(f"  [{split_name}] Tarea '{task}' → {n} WSIs")

    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    for batch_idx, (features_list, labels_batch, wsi_ids, coords_list) in enumerate(loader):
        features   = features_list[0]
        true_label = labels_batch[0].item()
        wsi_id     = wsi_ids[0]
        coords     = coords_list[0]

        if use_precomputed:
            h5_path = df_valid.iloc[batch_idx]["features"]
        else:
            h5_path = wsi_id

        wsi_stem = os.path.splitext(os.path.basename(h5_path))[0]

        print(f"\n    WSI {batch_idx+1}/{n} — {wsi_stem}  (true_{task}={true_label})")

        # ── Saltar etiqueta inválida ──────────────────────────────────────────
        if true_label == -1:
            print(f"      [SKIP] Etiqueta -1 para tarea '{task}'.")
            continue

        # ── Calcular scores de relevancia según el método elegido ─────────────
        if explanation_method == "single":
            print(f"      [Single] Ejecutando {features.shape[0]} forwards individuales...")

        try:
            pred_label, probs, weights = run_inference_single_wsi(
                model              = model,
                features           = features,
                device             = device,
                use_precomputed    = use_precomputed,
                explanation_method = explanation_method,
            )
        except Exception as e:
            print(f"      [ERROR] Inferencia fallida: {e}")
            continue

        outcome = "success" if pred_label == true_label else "failure"
        print(f"      pred={pred_label}  true={true_label}  outcome={outcome}")
        print(f"      probs={[f'{p:.3f}' for p in probs]}")
        print(f"      scores ({explanation_method}) — "
              f"min={weights.min():.4f}  max={weights.max():.4f}  mean={weights.mean():.4f}")

        # ── Coordenadas ───────────────────────────────────────────────────────
        if coords is None:
            print(f"      [WARN] coords=None desde el dataset, releyendo del H5...")
            try:
                with h5py.File(h5_path, "r") as h5:
                    coords = h5["coords"][:]
            except Exception as e:
                print(f"      [ERROR] No se pudieron obtener coords: {e}")
                continue

        coords_arr = np.asarray(coords)

        # ── Imágenes PIL para reconstrucción visual ───────────────────────────
        try:
            pil_imgs = _read_patch_images_from_h5(h5_path)
        except Exception as e:
            print(f"      [ERROR] Lectura de imágenes PIL fallida: {e}")
            continue

        if len(pil_imgs) != len(coords_arr):
            print(
                f"      [WARN] Discrepancia parches/coords: "
                f"{len(pil_imgs)} imgs vs {len(coords_arr)} coords. "
                f"Usando mínimo."
            )
            n_min      = min(len(pil_imgs), len(coords_arr))
            pil_imgs   = pil_imgs[:n_min]
            coords_arr = coords_arr[:n_min]
            weights    = weights[:n_min]

        # ── Reconstrucción WSI ────────────────────────────────────────────────
        try:
            wsi_img, scale, origin, patch_size_out = reconstruct_wsi(
                pil_imgs, coords_arr, max_output_pixels
            )
        except Exception as e:
            print(f"      [ERROR] Reconstrucción WSI fallida: {e}")
            continue

        print(f"      Canvas={wsi_img.size}  escala={scale:.4f}  parche_px={patch_size_out}")

        # ── Máscara de calor ──────────────────────────────────────────────────
        try:
            heatmap_mask = build_heatmap_mask(
                coords_arr, weights, wsi_img.size, patch_size_out, scale, origin
            )
        except Exception as e:
            print(f"      [ERROR] Máscara de calor fallida: {e}")
            continue

        # ── Overlay ───────────────────────────────────────────────────────────
        try:
            overlay = build_overlay(wsi_img, heatmap_mask, overlay_alpha)
        except Exception as e:
            print(f"      [ERROR] Overlay fallido: {e}")
            continue

        # ── Guardar ───────────────────────────────────────────────────────────
        # El nombre de carpeta incluye el método para que no haya colisiones
        # al ejecutar ambos métodos sobre el mismo experimento.
        folder_tag = f"{task}_{explanation_method}_{outcome}_pred{pred_label}_true{true_label}"
        out_dir    = os.path.join(output_split_path, wsi_stem, folder_tag)
        os.makedirs(out_dir, exist_ok=True)

        save_image(wsi_img,                           os.path.join(out_dir, f"wsi.{fmt}"),     fmt)
        save_image(heatmap_mask.convert("RGB"),       os.path.join(out_dir, f"heatmap.{fmt}"), fmt)
        save_image(overlay,                           os.path.join(out_dir, f"overlay.{fmt}"), fmt)

        print(f"      Guardado: {out_dir}")

        # ── Liberar memoria ───────────────────────────────────────────────────
        del pil_imgs, wsi_img, heatmap_mask, overlay
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_multiple_seed(args.seed)

    parent = os.path.dirname(os.path.abspath(args.test_results_path))
    if args.output_path is None:
        args.output_path = os.path.join(parent, "attention_maps")
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
    print(f"Método explicación:   {args.explanation_method}")

    # ── Splits ────────────────────────────────────────────────────────────────
    dev_df, test_df = load_splits(args)

    print(f"\nSeleccionando {args.n_images} WSIs fijas por split...")
    print("  dev_set:")
    dev_selected  = select_wsis(dev_df,  args.n_images)
    print("  test_set:")
    test_selected = select_wsis(test_df, args.n_images)

    splits = {"dev_set": dev_selected, "test_set": test_selected}

    use_embeddings_root = args.embeddings_folder is not None
    print(f"\nModo embeddings precomputados: {use_embeddings_root}")

    # ── Bucle por tareas ──────────────────────────────────────────────────────
    for task in args.tasks:
        task_results_path = os.path.join(args.test_results_path, task)

        print(f"\n{'='*65}")
        print(f"  TAREA: {task}  |  Método: {args.explanation_method}")
        print(f"{'='*65}")

        config_path = os.path.join(task_results_path, "best_config.json")
        if not os.path.isfile(config_path):
            print(f"  [SKIP] best_config.json no encontrado en: {task_results_path}")
            continue

        with open(config_path) as f:
            cfg = json.load(f)

        hyperparams = cfg["selected"]["hyperparams"]
        enc_name    = hyperparams["model"]
        mil_mode    = hyperparams["mil"]

        # Validar compatibilidad antes de cargar pesos
        if args.explanation_method == "attention":
            if mil_mode not in SUPPORTED_MIL_ATTENTION:
                print(
                    f"  [SKIP] Método 'attention' no soportado para MIL '{mil_mode}'. "
                    f"Compatible con: {SUPPORTED_MIL_ATTENTION}. "
                    f"Usa --explanation_method single."
                )
                continue
        elif args.explanation_method == "single":
            if mil_mode not in SUPPORTED_MIL_SINGLE:
                print(
                    f"  [SKIP] Método 'single' no listado para MIL '{mil_mode}'. "
                    f"Añádela a SUPPORTED_MIL_SINGLE si el modelo tiene forward estándar."
                )
                continue

        print(f"  Encoder: {enc_name}  |  MIL: {mil_mode}")

        # Resolver carpeta de embeddings para este encoder
        emb_folder_enc  = None
        use_precomputed = False

        if use_embeddings_root:
            candidate = os.path.join(args.embeddings_folder, enc_name)
            if os.path.isdir(candidate):
                emb_folder_enc  = candidate
                use_precomputed = True
            else:
                print(
                    f"  [WARN] Carpeta de embeddings no encontrada: {candidate}. "
                    "Se usarán parches directamente."
                )

        # Aviso de rendimiento para Single sin embeddings precomputados
        if args.explanation_method == "single" and not use_precomputed:
            print(
                "  [WARN] Método 'single' sin embeddings precomputados puede ser muy lento: "
                "cada parche requiere extraer features + forward del modelo. "
                "Se recomienda usar --embeddings_folder."
            )

        num_classes = 4 if task == "molecular_subtype" else 2

        # ── Cargar modelo ─────────────────────────────────────────────────────
        try:
            model, _, _ = load_trained_model(
                task_results_path  = task_results_path,
                task               = task,
                num_classes        = num_classes,
                use_precomputed    = use_precomputed,
                device             = device,
                explanation_method = args.explanation_method,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"  [SKIP] {e}")
            continue
        except Exception as e:
            print(f"  [ERROR] Carga del modelo fallida: {e}")
            continue

        # ── Procesar dev y test ───────────────────────────────────────────────
        for split_name, df_selected in splits.items():
            process_task_on_split(
                split_name         = split_name,
                df_selected        = df_selected,
                task               = task,
                model              = model,
                use_precomputed    = use_precomputed,
                emb_folder_enc     = emb_folder_enc,
                device             = device,
                output_split_path  = os.path.join(args.output_path, split_name),
                max_output_pixels  = args.max_output_pixels,
                output_format      = args.output_format,
                overlay_alpha      = args.overlay_alpha,
                explanation_method = args.explanation_method,
            )

        # ── Liberar modelo ────────────────────────────────────────────────────
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n  Modelo '{task}' liberado.")

    print(f"\n{'='*65}")
    print(f"  Mapas generados en:  {args.output_path}")
    print(f"  Log guardado en:     {os.path.join(args.output_path, 'log.txt')}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()