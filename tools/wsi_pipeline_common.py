"""
wsi_pipeline_common.py
------------------------
Utilidades compartidas entre generate_heatmaps.py y predict_all_wsis.py.

Contiene todo lo que es puro "scaffolding" de inferencia MIL sobre WSIs:
  - logging a fichero + consola (TeeLogger)
  - carga de splits dev/test
  - resolución de la carpeta de embeddings por encoder
  - construcción del dataset (parches vs embeddings precomputados)
  - carga de un modelo entrenado desde best_config.json
  - orquestación del bucle (método) x (tarea) x modelo, con carga/liberación
    automática del modelo
  - iteración estándar de un DataLoader de una WSI a la vez
  - forward pass de la bag completa (pred_label, probs), reutilizable tanto
    para predicción simple como para el primer paso de los métodos 'single'
    y 'attention' de explicabilidad

Lo que NO está aquí (a propósito) es la lógica específica de cada script:
  - predict_all_wsis.py: solo necesita (pred_label, probs) → construir el CSV
  - generate_heatmaps.py: necesita además pesos por parche, imágenes, SRG

Cada script importa esto y se queda solo con su parte específica, que es la
que de verdad los diferencia.
"""

import gc
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_load import Patches_WSI_Dataset, Embeddings_WSI_Dataset, collate_fn
from utils.train_utils import get_model


# ──────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────

class TeeLogger:
    """Redirige sys.stdout a consola + fichero de log simultáneamente."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._orig_stdout = None
        self._log_file = None

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
        self._log_file = open(self.log_path, "w", encoding="utf-8")
        sys.stdout = self
        print(f"[LOG] Escribiendo log en: {self.log_path}")

    def stop(self):
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._log_file and not self._log_file.closed:
            self._log_file.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ──────────────────────────────────────────────────────────────────────────
# SPLITS
# ──────────────────────────────────────────────────────────────────────────

def load_splits(
    test_results_path: str,
    use_dev_set: bool,
    dev_df_path: Optional[str] = None,
    test_df_path: Optional[str] = None,
) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """Carga dev_df y test_df desde los CSV guardados por train.py."""
    parent = os.path.dirname(os.path.abspath(test_results_path))

    dev_path = dev_df_path or os.path.join(parent, "dev_df.csv")
    test_path = test_df_path or os.path.join(parent, "test_df.csv")

    dev_df = None
    if use_dev_set:
        if not os.path.isfile(dev_path):
            raise FileNotFoundError(
                f"dev_df.csv no encontrado en: {dev_path}\n"
                "Indica la ruta correcta con --dev_df_path."
            )
        dev_df = pd.read_csv(dev_path)
        print(f"[SPLITS] dev_set:  {len(dev_df)} WSIs cargadas desde {dev_path}")

    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"test_df.csv no encontrado en: {test_path}\n"
            "Indica la ruta correcta con --test_df_path."
        )
    test_df = pd.read_csv(test_path)
    print(f"[SPLITS] test_set: {len(test_df)} WSIs cargadas desde {test_path}")

    return dev_df, test_df


# ──────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────

def resolve_embeddings_folder(
    embeddings_folder: Optional[str], enc_name: str
) -> tuple[Optional[str], bool]:
    """Comprueba si existe <embeddings_folder>/<enc_name> y decide el modo."""
    if embeddings_folder is None:
        return None, False

    candidate = os.path.join(embeddings_folder, enc_name)
    if os.path.isdir(candidate):
        return candidate, True

    print(
        f"  [WARN] Carpeta de embeddings no encontrada: {candidate}. "
        "Se usarán parches directamente."
    )
    return None, False


def build_dataset(
    df: pd.DataFrame,
    task: str,
    use_precomputed: bool,
    emb_folder_enc: Optional[str],
) -> tuple:
    """
    Construye el dataset adecuado (Patches o Embeddings), filtrando de
    antemano las WSIs cuyo fichero de embeddings no exista (si aplica).

    Returns
    -------
    dataset  : Patches_WSI_Dataset | Embeddings_WSI_Dataset
    df_valid : pd.DataFrame — subconjunto de df realmente cargado
    """
    indices = list(range(len(df)))

    if use_precomputed:
        mask = []
        for i in indices:
            src = df.iloc[i]["features"]
            stem = os.path.splitext(os.path.basename(src))[0]
            path = os.path.join(emb_folder_enc, f"{stem}.h5")
            exists = os.path.isfile(path)
            if not exists:
                print(f"    [WARN] Embedding no encontrado, se omite WSI: {stem}")
            mask.append(exists)

        df_valid = df[mask].reset_index(drop=True)
        dataset = Embeddings_WSI_Dataset(
            dataframe=df_valid,
            indices=list(range(len(df_valid))),
            embeddings_folder=emb_folder_enc,
            task=task,
            load_on_demand=True,
        )
    else:
        df_valid = df.reset_index(drop=True)
        dataset = Patches_WSI_Dataset(
            dataframe=df_valid,
            indices=indices,
            task=task,
            load_on_demand=True,
        )

    return dataset, df_valid


@dataclass
class WsiSample:
    wsi_stem: str
    features: torch.Tensor
    true_label: int
    coords: np.ndarray
    h5_path: str


def iterate_wsi_dataset(
    dataset,
    df_valid: pd.DataFrame,
    use_precomputed: bool,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Iterator[WsiSample]:
    """
    Envuelve el DataLoader estándar (batch_size=1, sin shuffle) y devuelve
    un WsiSample por WSI. Este bucle era idéntico en los dos scripts
    originales (infer_split y process_task_on_split); ahora vive en un
    único sitio.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    for batch_idx, (features_list, labels_batch, wsi_ids, coords_list) in enumerate(loader):
        features = features_list[0]
        true_label = int(labels_batch[0].item())
        coords = coords_list[0]
        h5_path = df_valid.iloc[batch_idx]["features"] if use_precomputed else wsi_ids[0]
        wsi_stem = os.path.splitext(os.path.basename(h5_path))[0]

        yield WsiSample(
            wsi_stem=wsi_stem,
            features=features,
            true_label=true_label,
            coords=coords,
            h5_path=h5_path,
        )


# ──────────────────────────────────────────────────────────────────────────
# MODELO
# ──────────────────────────────────────────────────────────────────────────

def load_task_config(task_results_path: str) -> tuple[dict, str, str]:
    """Lee best_config.json y devuelve (cfg, enc_name, mil_mode)."""
    config_path = os.path.join(task_results_path, "best_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"best_config.json no encontrado en: {task_results_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    hyperparams = cfg["selected"]["hyperparams"]
    return cfg, hyperparams["model"], hyperparams["mil"]


def load_trained_model(
    task_results_path: str,
    task: str,
    num_classes: int,
    use_precomputed: bool,
    device: torch.device,
    enc_name: str,
    mil_mode: str,
) -> torch.nn.Module:
    """Reconstruye el modelo y carga los pesos de final_model/model.pt."""
    model = get_model(
        patch_encoder_type=enc_name,
        mil_aggregation_mode=mil_mode,
        unfreeze_backbone=False,
        num_classes=num_classes,
        skip_encoder=use_precomputed,
    ).to(device)

    ckpt = os.path.join(task_results_path, "final_model", "model.pt")
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
            f"  [WARN] model.pt no encontrado en {os.path.dirname(ckpt)}. "
            "Continuando con pesos aleatorios."
        )

    model.eval()
    return model


@dataclass
class TaskContext:
    task: str
    method: Optional[str]          # None si el script no tiene noción de "método"
    model: torch.nn.Module
    enc_name: str
    mil_mode: str
    use_precomputed: bool
    emb_folder_enc: Optional[str]
    num_classes: int


@dataclass
class BagForwardOutput:
    pred_label: int
    probs: np.ndarray
    logits: torch.Tensor
    patch_logits_list: list
    weights_list: list


def for_each_task_and_method(
    test_results_path: str,
    tasks: list[str],
    num_classes_fn: Callable[[str], int],
    device: torch.device,
    embeddings_folder: Optional[str] = None,
    methods: list[Optional[str]] = (None,),
    compat_check: Optional[Callable[[str, Optional[str]], None]] = None,
) -> Iterator[TaskContext]:
    """
    Orquesta el bucle (método) x (tarea): para cada combinación
      - lee best_config.json
      - valida compatibilidad (compat_check, opcional — debe lanzar
        ValueError si no es compatible)
      - resuelve la carpeta de embeddings
      - carga el modelo
      - hace yield de un TaskContext
      - libera el modelo (gc + cuda empty_cache) SIEMPRE, incluso si el
        consumidor lanza una excepción dentro del bloque `for`

    Esto reemplaza el bucle duplicado de generate_heatmaps.main / _run y de
    predict_all_wsis.run_predictions.
    """
    for method in methods:
        for task in tasks:
            task_results_path = os.path.join(test_results_path, task)

            try:
                cfg, enc_name, mil_mode = load_task_config(task_results_path)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            if compat_check is not None:
                try:
                    compat_check(mil_mode, method)
                except ValueError as e:
                    print(f"  [SKIP] {e}")
                    continue

            print(f"  Encoder: {enc_name}  |  MIL: {mil_mode}"
                  + (f"  |  Método: {method}" if method else ""))

            emb_folder_enc, use_precomputed = resolve_embeddings_folder(
                embeddings_folder, enc_name
            )
            num_classes = num_classes_fn(task)

            try:
                model = load_trained_model(
                    task_results_path=task_results_path,
                    task=task,
                    num_classes=num_classes,
                    use_precomputed=use_precomputed,
                    device=device,
                    enc_name=enc_name,
                    mil_mode=mil_mode,
                )
            except Exception as e:
                print(f"  [ERROR] Carga del modelo fallida: {e}")
                continue

            try:
                yield TaskContext(
                    task=task,
                    method=method,
                    model=model,
                    enc_name=enc_name,
                    mil_mode=mil_mode,
                    use_precomputed=use_precomputed,
                    emb_folder_enc=emb_folder_enc,
                    num_classes=num_classes,
                )
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  Modelo '{task}' liberado.")


# ──────────────────────────────────────────────────────────────────────────
# FORWARD DE LA BAG COMPLETA
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_bag(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
) -> BagForwardOutput:
    """
    Forward pass de una bag completa (o de tamaño 1, para el método 'single').

    Es el paso que predict_wsi(), _compute_attention_weights() y el primer
    paso de _compute_single_weights() hacían cada uno por su cuenta con
    código casi idéntico. Devuelve siempre la misma información que expone
    el modelo, más la predicción derivada de los logits de WSI.
    """
    feats = features.to(device)

    if use_precomputed:
        logits, patch_logits_list, weights_list = model.forward_from_embeddings([feats])
    else:
        logits, patch_logits_list, weights_list = model([feats])

    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_label = int(np.argmax(probs))

    return BagForwardOutput(
        pred_label=pred_label,
        probs=probs,
        logits=logits,
        patch_logits_list=patch_logits_list,
        weights_list=weights_list,
    )
