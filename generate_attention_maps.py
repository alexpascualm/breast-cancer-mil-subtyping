"""
generate_attention_maps.py
--------------------------
Genera mapas de atención para WSIs usando modelos entrenados con test.py.

Para cada tarea y cada WSI seleccionada produce tres imágenes:
  1. WSI reconstruida (rescalada a resolución visualizable)
  2. Máscara de atención (parches coloreados por peso de atención)
  3. Fusión de WSI + máscara de atención

Solo funciona con modelos entrenados con abMIL-bag-level. Las demás estrategias
MIL se saltan con un aviso.

Estructura de salida:
  <output_path>/
    dev_set/
      <task>/
        <wsi_id>_wsi.jpg
        <wsi_id>_attention.jpg
        <wsi_id>_overlay.jpg
    test_set/
      <task>/
        ...

Uso mínimo:
  python generate_attention_maps.py --test_results_path /workspace/results/my_run/test_results
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
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Imports del proyecto
# ──────────────────────────────────────────────────────────────────────────────
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
from utils.test_utils import select_best_config, build_final_args

# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera mapas de atención para WSIs con modelos abMIL-bag-level"
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

    # Rutas opcionales
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
        help="Carpeta de salida para los mapas. Por defecto: <test_results_path>/attention_maps"
    )

    # Selección de tareas
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=["er", "pr", "her2", "ki67", "molecular_subtype"],
        choices=["er", "pr", "her2", "ki67", "molecular_subtype"],
        help="Tareas para las que generar mapas de atención."
    )

    # Parámetros de generación
    parser.add_argument(
        "--n_images", type=int, default=10,
        help="Número de imágenes a procesar por split (dev_set y test_set)."
    )
    parser.add_argument(
        "--max_output_pixels", type=int, default=4_000_000,
        help=(
            "Máximo de píxeles en la imagen WSI reconstruida (ancho x alto). "
            "Por defecto 4M (~2000x2000). Ajusta para equilibrar detalle y tamaño."
        )
    )
    parser.add_argument(
        "--output_format", type=str, default="jpg", choices=["jpg", "png"],
        help="Formato de salida de las imágenes (jpg o png)."
    )
    parser.add_argument(
        "--overlay_alpha", type=float, default=0.45,
        help="Opacidad de la máscara de atención en la imagen de fusión (0.0-1.0)."
    )

    # Varios
    parser.add_argument("--slide_col", type=str, default="Patient ID")
    parser.add_argument("--seed", type=int, default=98)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# MIL compatible con mapas de atención (los demás se omiten)
SUPPORTED_MIL = "abMIL-bag-level"

# Colormap para la máscara de atención (rojo = alta atención)
# Se aplica manualmente para evitar dependencia de matplotlib
def _attention_colormap(weight_normalized: float) -> tuple:
    """
    Convierte un peso normalizado [0,1] a color RGB usando un gradiente
    azul → amarillo → rojo (simula 'jet' o 'hot' sin matplotlib).

    Args:
        weight_normalized: float en [0, 1]

    Returns:
        (R, G, B) tuple con valores enteros en [0, 255]
    """
    w = float(np.clip(weight_normalized, 0.0, 1.0))

    # Gradiente: azul frío (0) → cian → verde → amarillo → rojo caliente (1)
    if w < 0.25:
        t = w / 0.25
        r = int(0)
        g = int(t * 255)
        b = int(255)
    elif w < 0.5:
        t = (w - 0.25) / 0.25
        r = int(0)
        g = int(255)
        b = int((1 - t) * 255)
    elif w < 0.75:
        t = (w - 0.5) / 0.25
        r = int(t * 255)
        g = int(255)
        b = int(0)
    else:
        t = (w - 0.75) / 0.25
        r = int(255)
        g = int((1 - t) * 255)
        b = int(0)

    return (r, g, b)

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
# CARGA DEL MODELO ENTRENADO
# ──────────────────────────────────────────────────────────────────────────────

def load_trained_model(task_results_path: str, task: str, num_classes: int,
                       precomputed: bool, device: torch.device):
    """
    Reconstruye el modelo a partir de best_config.json y carga los pesos
    guardados en final_model/ (si existen) o advierte que se usará el modelo
    con pesos aleatorios (útil solo para pruebas de estructura).

    Retorna:
        model    : WSIClassifier listo para inferencia
        enc_name : nombre del encoder (str)
        mil_mode : modo MIL (str)
    """
    config_path = os.path.join(task_results_path, "best_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"best_config.json no encontrado en: {task_results_path}\n"
            "Ejecuta test.py antes de generar los mapas de atención."
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    selected = cfg["selected"]
    hyperparams = selected["hyperparams"]
    enc_name = hyperparams["model"]
    mil_mode = hyperparams["mil"]

    # Solo abMIL-bag-level soportado
    if mil_mode != SUPPORTED_MIL:
        raise ValueError(
            f"[{task}] Estrategia MIL '{mil_mode}' no soportada para mapas de atención. "
            f"Solo se genera mapa para '{SUPPORTED_MIL}'."
        )

    model = get_model(
        patch_encoder_type=enc_name,
        mil_aggregation_mode=mil_mode,
        unfreeze_backbone=False,
        num_classes=num_classes,
        skip_encoder=precomputed,
    ).to(device)

    # Intentar cargar pesos entrenados
    model_path = os.path.join(task_results_path, "final_model")
    if os.path.isdir(model_path):
        # accelerate guarda en format safetensors o pytorch_model.bin
        # intentamos cargar con accelerate load_state
        try:
            from accelerate import Accelerator
            _acc = Accelerator()
            _acc.load_state(model_path)
            print(f"  [OK] Pesos cargados desde: {model_path}")
        except Exception as e:
            # Intento directo con torch.load si el fichero es un state_dict
            bin_path = os.path.join(model_path, "pytorch_model.bin")
            sf_path  = os.path.join(model_path, "model.safetensors")
            loaded = False
            for ckpt in [sf_path, bin_path]:
                if os.path.isfile(ckpt):
                    try:
                        if ckpt.endswith(".safetensors"):
                            from safetensors.torch import load_file
                            state = load_file(ckpt, device=str(device))
                        else:
                            state = torch.load(ckpt, map_location=device)
                        # El state_dict puede venir envuelto en "model"
                        if isinstance(state, dict) and "model" in state:
                            state = state["model"]
                        model.load_state_dict(state, strict=False)
                        print(f"  [OK] Pesos cargados desde: {ckpt}")
                        loaded = True
                        break
                    except Exception as e2:
                        print(f"  [WARN] No se pudieron cargar desde {ckpt}: {e2}")
            if not loaded:
                print(
                    f"  [WARN] No se encontraron pesos en {model_path}. "
                    "Se usa el modelo con pesos inicializados aleatoriamente."
                )
    else:
        print(
            f"  [WARN] Carpeta final_model/ no encontrada en {task_results_path}. "
            "Asegúrate de haber ejecutado test.py con --save_models.\n"
            "  Se continúa con pesos aleatorios (útil solo para pruebas)."
        )

    model.eval()
    return model, enc_name, mil_mode


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCIA PARA HEATMAP — UNA WSI
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_single_wsi(model, wsi_data, device: torch.device, precomputed: bool):
    """
    Ejecuta forward sobre una sola WSI y devuelve los pesos de atención.

    Args:
        model      : WSIClassifier en modo eval
        wsi_data   : dict con claves 'features' (tensor) y 'coords' (ndarray)
        device     : dispositivo
        precomputed: si True, features son embeddings; si no, son imágenes

    Returns:
        pred_label  : int
        probs       : ndarray (num_classes,)
        weights     : ndarray (N_patches,) — pesos de atención normalizados
        coords      : ndarray (N_patches, 2) — coordenadas de cada parche
    """
    feats  = wsi_data["features"].to(device)  # (N, D) o (N, C, H, W)
    coords = wsi_data["coords"]               # ndarray (N, 2) o None

    with torch.no_grad():
        if precomputed:
            wsi_logits, _, weights_list = model.forward_from_embeddings([feats])
        else:
            wsi_logits, _, weights_list = model([feats])

    probs  = torch.softmax(wsi_logits[0], dim=0).cpu().numpy()
    pred_label = int(np.argmax(probs))

    weights = weights_list[0].cpu().numpy()  # (N,)

    return pred_label, probs, weights, coords


# ──────────────────────────────────────────────────────────────────────────────
# RECONSTRUCCIÓN Y GUARDADO DE IMÁGENES
# ──────────────────────────────────────────────────────────────────────────────

def _read_patch_images_from_h5(h5_path: str):
    """
    Lee las imágenes y coordenadas raw desde un fichero H5.

    Returns:
        imgs   : list of PIL.Image
        coords : ndarray (N, 2)
    """
    with h5py.File(h5_path, "r") as h5:
        imgs_raw = h5["imgs"][:]    # (N, H, W, 3) uint8
        coords   = h5["coords"][:] # (N, 2)  [x, y] o [row, col] en pixels del WSI
    pil_imgs = [Image.fromarray(img) for img in imgs_raw]
    return pil_imgs, coords


def reconstruct_wsi(
    pil_imgs: list,
    coords: np.ndarray,
    max_output_pixels: int = 4_000_000,
) -> tuple:
    """
    Reconstruye la imagen WSI colocando los parches según sus coordenadas.

    Args:
        pil_imgs          : lista de PIL.Image (parches)
        coords            : ndarray (N, 2) con coordenadas absolutas en el WSI
                            Se asume formato [x, y] (columna, fila) en píxeles.
        max_output_pixels : límite de píxeles para la imagen de salida

    Returns:
        canvas  : PIL.Image con la WSI reconstruida y rescalada
        scale   : float — factor de escala aplicado
        origin  : (x_min, y_min) — desplazamiento aplicado
        patch_size_out : int — tamaño del parche en la imagen de salida
    """
    if len(pil_imgs) == 0:
        raise ValueError("No hay parches para reconstruir la WSI.")

    patch_w, patch_h = pil_imgs[0].size   # tamaño original del parche
    patch_size = max(patch_w, patch_h)    # asumimos parches cuadrados

    coords_arr = np.array(coords)         # (N, 2)
    x_min, y_min = coords_arr[:, 0].min(), coords_arr[:, 1].min()
    x_max, y_max = coords_arr[:, 0].max(), coords_arr[:, 1].max()

    # Dimensiones del canvas original (en píxeles del WSI)
    canvas_w = int(x_max - x_min + patch_size)
    canvas_h = int(y_max - y_min + patch_size)

    # Calcular factor de escala para no superar max_output_pixels
    total_pixels = canvas_w * canvas_h
    if total_pixels > max_output_pixels:
        scale = (max_output_pixels / total_pixels) ** 0.5
    else:
        scale = 1.0

    out_w = max(1, int(canvas_w * scale))
    out_h = max(1, int(canvas_h * scale))
    patch_size_out = max(1, int(patch_size * scale))

    canvas = Image.new("RGB", (out_w, out_h), color=(240, 240, 240))

    for img_pil, (x, y) in zip(pil_imgs, coords_arr):
        # Coordenadas relativas al origen
        px = int((x - x_min) * scale)
        py = int((y - y_min) * scale)

        # Redimensionar parche al tamaño de salida
        patch_resized = img_pil.resize(
            (patch_size_out, patch_size_out), Image.BICUBIC
        )
        # Proteger contra desbordamiento del canvas
        px = min(px, out_w - patch_size_out)
        py = min(py, out_h - patch_size_out)
        canvas.paste(patch_resized, (px, py))

    return canvas, scale, (x_min, y_min), patch_size_out


def build_attention_mask(
    coords: np.ndarray,
    weights: np.ndarray,
    canvas_size: tuple,
    patch_size_out: int,
    scale: float,
    origin: tuple,
) -> Image.Image:
    """
    Construye la máscara RGBA de atención. Cada parche se colorea según
    su peso de atención (frío = bajo, cálido = alto).

    Args:
        coords         : (N, 2) coordenadas absolutas del WSI
        weights        : (N,) pesos de atención
        canvas_size    : (width, height) de la imagen de salida
        patch_size_out : tamaño del parche en la imagen de salida
        scale          : factor de escala
        origin         : (x_min, y_min) del canvas

    Returns:
        mask : PIL.Image RGBA
    """
    out_w, out_h = canvas_size
    x_min, y_min = origin

    # Normalizar pesos al rango [0, 1]
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights_norm = (weights - w_min) / (w_max - w_min)
    else:
        weights_norm = np.ones_like(weights) * 0.5

    mask = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

    for (x, y), w_norm in zip(coords, weights_norm):
        px = int((x - x_min) * scale)
        py = int((y - y_min) * scale)
        px = min(px, out_w - patch_size_out)
        py = min(py, out_h - patch_size_out)

        r, g, b = _attention_colormap(w_norm)
        # Opacidad proporcional al peso (mínimo 60 para que siempre sea visible)
        alpha = int(60 + w_norm * 195)

        patch_color = Image.new("RGBA", (patch_size_out, patch_size_out), (r, g, b, alpha))
        mask.paste(patch_color, (px, py))

    return mask


def build_overlay(wsi_img: Image.Image, mask: Image.Image, alpha: float = 0.45) -> Image.Image:
    """
    Fusiona la WSI reconstruida con la máscara de atención.

    Args:
        wsi_img : PIL.Image RGB
        mask    : PIL.Image RGBA
        alpha   : peso de la máscara (0 = solo WSI, 1 = solo máscara)

    Returns:
        overlay : PIL.Image RGB
    """
    wsi_rgba = wsi_img.convert("RGBA")

    # Ajustar el canal alfa de la máscara según el parámetro alpha
    r, g, b, a = mask.split()
    a_scaled = a.point(lambda v: int(v * alpha))
    mask_adjusted = Image.merge("RGBA", (r, g, b, a_scaled))

    overlay = Image.alpha_composite(wsi_rgba, mask_adjusted)
    return overlay.convert("RGB")


def save_image(img: Image.Image, path: str, fmt: str = "jpg"):
    """Guarda una PIL.Image en el formato indicado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt.lower() == "jpg":
        img.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)
    else:
        img.save(path, format="PNG", optimize=True)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL POR TAREA Y SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def process_split(
    split_name: str,
    df_split: pd.DataFrame,
    task: str,
    model,
    enc_name: str,
    precomputed: bool,
    embeddings_folder: str | None,
    device: torch.device,
    n_images: int,
    output_task_path: str,
    max_output_pixels: int,
    output_format: str,
    overlay_alpha: float,
):
    """
    Procesa `n_images` WSIs de un split para una tarea y genera los mapas.
    """
    # Filtrar WSIs con etiqueta válida para esta tarea
    df_task = df_split[df_split[task] != -1].reset_index(drop=True)
    if len(df_task) == 0:
        print(f"    [WARN] No hay WSIs válidas para tarea '{task}' en {split_name}.")
        return

    n_to_process = min(n_images, len(df_task))
    print(f"\n  [{split_name}] Procesando {n_to_process} / {len(df_task)} WSIs "
          f"válidas para tarea '{task}'")

    os.makedirs(output_task_path, exist_ok=True)

    for idx in range(n_to_process):
        row = df_task.iloc[idx]
        h5_path    = row["features"]           # ruta al .h5 de parches
        patient_id = str(row.get("patient_id", idx))
        true_label = int(row[task])

        wsi_stem = os.path.splitext(os.path.basename(h5_path))[0]
        print(f"\n    WSI {idx+1}/{n_to_process} — {wsi_stem} "
              f"(label={true_label}, patient={patient_id})")

        # ── Leer parches raw para la reconstrucción visual ──────────────────
        try:
            pil_imgs, coords = _read_patch_images_from_h5(h5_path)
        except Exception as e:
            print(f"      [ERROR] No se pudo leer {h5_path}: {e}")
            continue

        # ── Construir tensor de entrada para inferencia ─────────────────────
        if precomputed:
            emb_path = os.path.join(embeddings_folder, f"{wsi_stem}.h5")
            if not os.path.isfile(emb_path):
                print(f"      [SKIP] Embedding no encontrado: {emb_path}")
                continue
            with h5py.File(emb_path, "r") as h5:
                embeddings = torch.from_numpy(h5["embeddings"][:])
                # coords puede venir del embedding h5 o del parche h5
                if "coords" in h5:
                    coords = h5["coords"][:]
            wsi_data = {"features": embeddings, "coords": coords}
        else:
            # Transformación idéntica a la usada en Patches_WSI_Dataset
            from torchvision import transforms as T
            transform = T.Compose([
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            imgs_tensor = torch.stack([transform(img) for img in pil_imgs])
            wsi_data = {"features": imgs_tensor, "coords": coords}

        # ── Inferencia ──────────────────────────────────────────────────────
        try:
            pred_label, probs, weights, _ = run_inference_single_wsi(
                model, wsi_data, device, precomputed
            )
        except Exception as e:
            print(f"      [ERROR] Inferencia fallida: {e}")
            continue

        pred_str = f"pred{pred_label}_true{true_label}"
        print(f"      pred={pred_label}  true={true_label}  "
              f"probs={[f'{p:.3f}' for p in probs]}")
        print(f"      pesos atención — min={weights.min():.4f}  "
              f"max={weights.max():.4f}  mean={weights.mean():.4f}")

        # ── Reconstrucción WSI ───────────────────────────────────────────────
        try:
            wsi_img, scale, origin, patch_size_out = reconstruct_wsi(
                pil_imgs, coords, max_output_pixels
            )
        except Exception as e:
            print(f"      [ERROR] Reconstrucción WSI fallida: {e}")
            continue

        print(f"      Canvas: {wsi_img.size}  |  escala={scale:.4f}  "
              f"|  parche_px={patch_size_out}")

        # ── Máscara de atención ─────────────────────────────────────────────
        try:
            attn_mask = build_attention_mask(
                coords, weights,
                wsi_img.size, patch_size_out,
                scale, origin,
            )
        except Exception as e:
            print(f"      [ERROR] Máscara de atención fallida: {e}")
            continue

        # ── Overlay ─────────────────────────────────────────────────────────
        try:
            overlay = build_overlay(wsi_img, attn_mask, overlay_alpha)
        except Exception as e:
            print(f"      [ERROR] Overlay fallido: {e}")
            continue

        # ── Guardar imágenes ─────────────────────────────────────────────────
        base = f"{wsi_stem}_{pred_str}"
        fmt  = output_format.lower()

        wsi_path     = os.path.join(output_task_path, f"{base}_wsi.{fmt}")
        attn_path    = os.path.join(output_task_path, f"{base}_attention.{fmt}")
        overlay_path = os.path.join(output_task_path, f"{base}_overlay.{fmt}")

        # La máscara RGBA no va bien en jpg; convertir a RGB antes
        attn_rgb = attn_mask.convert("RGB")

        save_image(wsi_img,  wsi_path,     fmt)
        save_image(attn_rgb, attn_path,    fmt)
        save_image(overlay,  overlay_path, fmt)

        print(f"      Guardado: {os.path.basename(wsi_path)}")
        print(f"               {os.path.basename(attn_path)}")
        print(f"               {os.path.basename(overlay_path)}")

        # Liberar memoria
        del wsi_data, pil_imgs, wsi_img, attn_mask, overlay, attn_rgb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_multiple_seed(args.seed)

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Rutas de salida
    if args.output_path is None:
        args.output_path = os.path.join(args.test_results_path, "attention_maps")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Salida: {args.output_path}")

    # Cargar splits
    dev_df, test_df = load_splits(args)

    # Leer el best_config de las tareas desde los resultados de test.py
    # No necesitamos re-escanear experiments/ porque los best_config.json
    # ya están en las subcarpetas de test_results_path.
    # Construimos el "best" directamente desde best_config.json.

    precomputed = args.embeddings_folder is not None
    print(f"Modo embeddings precomputados: {precomputed}")

    splits = {
        "dev_set":  dev_df,
        "test_set": test_df,
    }

    for task in args.tasks:
        task_results_path = os.path.join(args.test_results_path, task)

        print(f"\n{'='*65}")
        print(f"  TAREA: {task}")
        print(f"{'='*65}")

        # Verificar best_config.json
        config_path = os.path.join(task_results_path, "best_config.json")
        if not os.path.isfile(config_path):
            print(f"  [SKIP] best_config.json no encontrado en: {task_results_path}")
            print("         Ejecuta test.py antes de generar los mapas de atención.")
            continue

        # Leer configuración
        with open(config_path, "r") as f:
            cfg = json.load(f)

        selected    = cfg["selected"]
        hyperparams = selected["hyperparams"]
        enc_name    = hyperparams["model"]
        mil_mode    = hyperparams["mil"]

        # Comprobar si el MIL es compatible
        if mil_mode != SUPPORTED_MIL:
            print(
                f"  [SKIP] MIL '{mil_mode}' no soportado para mapas de atención. "
                f"Solo se procesa '{SUPPORTED_MIL}'."
            )
            continue

        print(f"  Encoder: {enc_name}  |  MIL: {mil_mode}")

        # Número de clases
        num_classes = 4 if task == "molecular_subtype" else 2

        # Embeddings folder específica del encoder
        emb_folder_enc = None
        if precomputed:
            emb_folder_enc = os.path.join(args.embeddings_folder, enc_name)
            if not os.path.isdir(emb_folder_enc):
                print(
                    f"  [WARN] Carpeta de embeddings no encontrada: {emb_folder_enc}. "
                    "Se usarán parches directamente (sin precomputed)."
                )
                emb_folder_enc = None

        use_precomputed = emb_folder_enc is not None

        # Cargar modelo
        try:
            model, enc_name_loaded, mil_mode_loaded = load_trained_model(
                task_results_path, task, num_classes, use_precomputed, device
            )
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue
        except Exception as e:
            print(f"  [ERROR] No se pudo cargar el modelo: {e}")
            continue

        # Procesar dev y test
        for split_name, df_split in splits.items():
            output_task_path = os.path.join(args.output_path, split_name, task)
            process_split(
                split_name=split_name,
                df_split=df_split,
                task=task,
                model=model,
                enc_name=enc_name_loaded,
                precomputed=use_precomputed,
                embeddings_folder=emb_folder_enc,
                device=device,
                n_images=args.n_images,
                output_task_path=output_task_path,
                max_output_pixels=args.max_output_pixels,
                output_format=args.output_format,
                overlay_alpha=args.overlay_alpha,
            )

        # Liberar modelo antes de pasar a la siguiente tarea
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print(f"  Mapas de atención generados en: {args.output_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
