"""
predict_all_wsis.py
--------------------
Carga los modelos entrenados (uno por tarea) y ejecuta inferencia sobre el
conjunto de dev y test completo, generando un CSV con las predicciones.

Estructura del CSV de salida
-----------------------------
Cada fila corresponde a una WSI única. Las columnas son:

  wsi_stem                          ← ID único de la WSI (stem del fichero .h5)
  split                             ← "dev_set" | "test_set"

  Por cada tarea (er, pr, her2, ki67, molecular_subtype):
    <task>_true                     ← etiqueta verdadera (int; -1 si no disponible)
    <task>_pred                     ← clase predicha (int)
    <task>_prob_<c>                 ← probabilidad softmax de cada clase c

Ejemplo de columnas para la tarea 'er' (2 clases):
  er_true | er_pred | er_prob_0 | er_prob_1

Ejemplo para 'molecular_subtype' (4 clases):
  molecular_subtype_true | molecular_subtype_pred |
  molecular_subtype_prob_0 | ... | molecular_subtype_prob_3

Uso mínimo
----------
  python predict_all_wsis.py --test_results_path /workspace/results/my_run/test_results

Con dev set:
  python predict_all_wsis.py \\
      --test_results_path /workspace/results/my_run/test_results \\
      --use_dev_set
"""
  
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
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
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

TASKS_NUM_CLASSES = {
    "er"                : 2,
    "pr"                : 2,
    "her2"              : 2,
    "ki67"              : 2,
    "molecular_subtype" : 4,
}

ALL_TASKS = list(TASKS_NUM_CLASSES.keys())


# ──────────────────────────────────────────────────────────────────────────────
# TEE LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class TeeLogger:
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
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta inferencia sobre el dev y/o test set completo y guarda\n"
            "las predicciones (true label, pred label, probabilidades) en un CSV."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
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
            "Ruta raíz de embeddings precomputados.\n"
            "Se espera subcarpeta por encoder: <embeddings_folder>/<encoder>/<wsi>.h5\n"
            "Si se proporciona y la carpeta existe, se usa Embeddings_WSI_Dataset."
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
            "Carpeta de salida.\n"
            "Por defecto: <parent de test_results_path>/predictions/"
        )
    )
    parser.add_argument(
        "--output_filename", type=str, default="predictions.csv",
        help="Nombre del CSV de salida (por defecto: predictions.csv)."
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=ALL_TASKS,
        choices=ALL_TASKS,
        help="Tareas para las que generar predicciones."
    )
    parser.add_argument(
        "--use_dev_set", action="store_true",
        help="Si se activa, procesa también el dev_set además del test_set."
    )
    parser.add_argument(
        "--slide_col", type=str, default="Patient ID",
        help="Nombre de la columna con el ID de la WSI en el dataframe."
    )
    parser.add_argument(
        "--seed", type=int, default=98,
        help="Semilla aleatoria."
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Número de workers para el DataLoader."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size para el DataLoader (recomendado: 1 para MIL)."
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE SPLITS
# ──────────────────────────────────────────────────────────────────────────────

def load_splits(args) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Carga dev_df y test_df desde los CSV guardados por train.py."""
    parent = os.path.dirname(os.path.abspath(args.test_results_path))

    dev_path  = args.dev_df_path  or os.path.join(parent, "dev_df.csv")
    test_path = args.test_df_path or os.path.join(parent, "test_df.csv")

    dev_df = None
    if args.use_dev_set:
        if not os.path.isfile(dev_path):
            raise FileNotFoundError(
                f"dev_df.csv no encontrado en: {dev_path}\n"
                "Usa --dev_df_path para indicar la ruta correcta."
            )
        dev_df = pd.read_csv(dev_path)
        print(f"[SPLITS] dev_set:  {len(dev_df)} WSIs cargadas desde {dev_path}")

    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"test_df.csv no encontrado en: {test_path}\n"
            "Usa --test_df_path para indicar la ruta correcta."
        )
    test_df = pd.read_csv(test_path)
    print(f"[SPLITS] test_set: {len(test_df)} WSIs cargadas desde {test_path}")

    return dev_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO
# ──────────────────────────────────────────────────────────────────────────────

def load_trained_model(
    task_results_path: str,
    task: str,
    num_classes: int,
    use_precomputed: bool,
    device: torch.device,
) -> tuple:
    """
    Reconstruye el modelo desde best_config.json y carga los pesos.

    Returns
    -------
    model    : WSIClassifier en modo eval
    enc_name : str  ← nombre del encoder
    mil_mode : str  ← modo MIL
    """
    config_path = os.path.join(task_results_path, "best_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"best_config.json no encontrado en: {task_results_path}"
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    hyperparams = cfg["selected"]["hyperparams"]
    enc_name    = hyperparams["model"]
    mil_mode    = hyperparams["mil"]

    model = get_model(
        patch_encoder_type   = enc_name,
        mil_aggregation_mode = mil_mode,
        unfreeze_backbone    = False,
        num_classes          = num_classes,
        skip_encoder         = use_precomputed,
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
    return model, enc_name, mil_mode


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATASET
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    df: pd.DataFrame,
    task: str,
    use_precomputed: bool,
    emb_folder_enc: str | None,
) -> tuple:
    """
    Construye el dataset adecuado (Patches o Embeddings) filtrando WSIs
    cuyo fichero de embeddings no exista.

    Returns
    -------
    dataset  : Patches_WSI_Dataset | Embeddings_WSI_Dataset
    df_valid : pd.DataFrame — subconjunto de df realmente cargado
    """
    indices = list(range(len(df)))

    if use_precomputed:
        mask = []
        for i in indices:
            src    = df.iloc[i]["features"]
            stem   = os.path.splitext(os.path.basename(src))[0]
            path   = os.path.join(emb_folder_enc, f"{stem}.h5")
            exists = os.path.isfile(path)
            if not exists:
                print(f"    [WARN] Embedding no encontrado, se omite WSI: {stem}")
            mask.append(exists)

        df_valid = df[mask].reset_index(drop=True)
        dataset  = Embeddings_WSI_Dataset(
            dataframe         = df_valid,
            indices           = list(range(len(df_valid))),
            embeddings_folder = emb_folder_enc,
            task              = task,
            load_on_demand    = True,
        )
    else:
        df_valid = df.reset_index(drop=True)
        dataset  = Patches_WSI_Dataset(
            dataframe      = df_valid,
            indices        = indices,
            task           = task,
            load_on_demand = True,
        )

    return dataset, df_valid


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCIA DE UNA WSI
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_wsi(
    model,
    features: torch.Tensor,
    device: torch.device,
    use_precomputed: bool,
) -> tuple[int, np.ndarray]:
    """
    Ejecuta el forward pass para una WSI completa.

    Returns
    -------
    pred_label : int
    probs      : ndarray (num_classes,)
    """
    feats = features.to(device)

    if use_precomputed:
        logits, *_ = model.forward_from_embeddings([feats])
    else:
        logits, *_ = model([feats])

    probs      = torch.softmax(logits[0], dim=0).cpu().numpy()
    pred_label = int(np.argmax(probs))

    return pred_label, probs


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCIA SOBRE UN SPLIT COMPLETO PARA UNA TAREA
# ──────────────────────────────────────────────────────────────────────────────

def infer_split(
    df_split: pd.DataFrame,
    task: str,
    num_classes: int,
    model,
    use_precomputed: bool,
    emb_folder_enc: str | None,
    device: torch.device,
    num_workers: int,
    batch_size: int,
) -> dict[str, dict]:
    """
    Ejecuta inferencia sobre todas las WSIs de un split para una tarea dada.

    Returns
    -------
    dict[wsi_stem -> {
        "<task>_true"     : int,
        "<task>_pred"     : int,
        "<task>_prob_<c>" : float,   # una entrada por clase
    }]
    """
    records: dict[str, dict] = {}

    try:
        dataset, df_valid = build_dataset(df_split, task, use_precomputed, emb_folder_enc)
    except Exception as e:
        print(f"    [ERROR] No se pudo construir el dataset para '{task}': {e}")
        return records

    n = len(dataset)
    if n == 0:
        print(f"    [WARN] Dataset vacío para tarea '{task}'.")
        return records

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = num_workers,
    )

    print(f"    Inferencia sobre {n} WSIs...")

    for batch_idx, (features_list, labels_batch, wsi_ids, _coords) in enumerate(loader):
        features   = features_list[0]
        true_label = int(labels_batch[0].item())
        h5_path    = df_valid.iloc[batch_idx]["features"] if use_precomputed else wsi_ids[0]
        wsi_stem   = os.path.splitext(os.path.basename(h5_path))[0]

        if true_label == -1:
            # Sin etiqueta para esta tarea — registramos NaN en predicciones
            entry = {f"{task}_true": -1, f"{task}_pred": None}
            for c in range(num_classes):
                entry[f"{task}_prob_{c}"] = None
            records[wsi_stem] = entry
            continue

        try:
            pred_label, probs = predict_wsi(model, features, device, use_precomputed)
        except Exception as e:
            print(f"    [ERROR] Inferencia fallida para {wsi_stem}: {e}")
            continue

        entry = {
            f"{task}_true": true_label,
            f"{task}_pred": pred_label,
        }
        for c, p in enumerate(probs):
            entry[f"{task}_prob_{c}"] = float(p)

        records[wsi_stem] = entry

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n:
            print(f"      {batch_idx + 1}/{n} procesadas...")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# FUSIÓN DE RESULTADOS EN UN DATAFRAME
# ──────────────────────────────────────────────────────────────────────────────

def merge_task_records(
    all_records: dict[str, dict],
    wsi_stem: str,
    split_name: str,
) -> dict:
    """
    Construye la fila del CSV para una WSI combinando los resultados de todas
    las tareas. Las columnas de tareas no procesadas se rellenan con None.
    """
    row = {"wsi_stem": wsi_stem, "split": split_name}
    if wsi_stem in all_records:
        row.update(all_records[wsi_stem])
    return row


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_predictions(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] Dispositivo:          {device}")
    print(f"[CONFIG] test_results:         {args.test_results_path}")
    print(f"[CONFIG] Salida:               {args.output_path}")
    print(f"[CONFIG] Tareas:               {args.tasks}")
    print(f"[CONFIG] Procesar dev_set:     {args.use_dev_set}")

    # ── Splits ────────────────────────────────────────────────────────────────
    dev_df, test_df = load_splits(args)

    splits: dict[str, pd.DataFrame] = {"test_set": test_df}
    if args.use_dev_set and dev_df is not None:
        splits["dev_set"] = dev_df

    use_embeddings_root = args.embeddings_folder is not None
    print(f"[CONFIG] Embeddings precomputados: {use_embeddings_root}")

    # ── Resultado acumulado: split → {wsi_stem → {col → val}} ─────────────────
    # Estructura:  accumulated[split_name][wsi_stem] = {col: val, ...}
    accumulated: dict[str, dict[str, dict]] = {s: {} for s in splits}

    # ── Bucle por tarea ───────────────────────────────────────────────────────
    for task in args.tasks:
        num_classes       = TASKS_NUM_CLASSES[task]
        task_results_path = os.path.join(args.test_results_path, task)

        print(f"\n{'='*60}")
        print(f"  TAREA: {task}  ({num_classes} clases)")
        print(f"{'='*60}")

        # Verificar que existe best_config.json
        config_path = os.path.join(task_results_path, "best_config.json")
        if not os.path.isfile(config_path):
            print(f"  [SKIP] best_config.json no encontrado en: {task_results_path}")
            continue

        with open(config_path) as f:
            cfg = json.load(f)
        enc_name = cfg["selected"]["hyperparams"]["model"]
        mil_mode = cfg["selected"]["hyperparams"]["mil"]
        print(f"  Encoder: {enc_name}  |  MIL: {mil_mode}")

        # Resolver carpeta de embeddings
        emb_folder_enc  = None
        use_precomputed = False

        if use_embeddings_root:
            candidate = os.path.join(args.embeddings_folder, enc_name)
            if os.path.isdir(candidate):
                emb_folder_enc  = candidate
                use_precomputed = True
                print(f"  Embeddings: {emb_folder_enc}")
            else:
                print(
                    f"  [WARN] Carpeta de embeddings no encontrada: {candidate}. "
                    "Se usarán parches directamente."
                )

        # Cargar modelo
        try:
            model, _, _ = load_trained_model(
                task_results_path = task_results_path,
                task              = task,
                num_classes       = num_classes,
                use_precomputed   = use_precomputed,
                device            = device,
            )
        except (FileNotFoundError, Exception) as e:
            print(f"  [SKIP] Carga del modelo fallida: {e}")
            continue

        # Inferencia por split
        for split_name, df_split in splits.items():
            print(f"\n  [{split_name}] {len(df_split)} WSIs...")

            records = infer_split(
                df_split        = df_split,
                task            = task,
                num_classes     = num_classes,
                model           = model,
                use_precomputed = use_precomputed,
                emb_folder_enc  = emb_folder_enc,
                device          = device,
                num_workers     = args.num_workers,
                batch_size      = args.batch_size,
            )

            # Fusionar en accumulated
            for wsi_stem, task_entry in records.items():
                if wsi_stem not in accumulated[split_name]:
                    accumulated[split_name][wsi_stem] = {}
                accumulated[split_name][wsi_stem].update(task_entry)

            print(f"  [{split_name}] {len(records)} WSIs procesadas para '{task}'.")

        # Liberar modelo
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n  Modelo '{task}' liberado.")

    # ── Construir DataFrame final ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Construyendo CSV final...")
    print(f"{'='*60}")

    rows = []
    for split_name, df_split in splits.items():
        for _, row_df in df_split.iterrows():
            # Obtener el wsi_stem desde la columna 'features'
            features_col = row_df.get("features", None)
            if features_col is not None:
                wsi_stem = os.path.splitext(os.path.basename(str(features_col)))[0]
            elif args.slide_col in row_df:
                wsi_stem = str(row_df[args.slide_col])
            else:
                print(f"  [WARN] No se pudo determinar wsi_stem para una fila de {split_name}.")
                continue

            row = {"wsi_stem": wsi_stem, "split": split_name}

            # Añadir predicciones por tarea (si se procesaron)
            wsi_data = accumulated[split_name].get(wsi_stem, {})
            row.update(wsi_data)

            # Rellenar con None las tareas no procesadas o fallidas
            for task in args.tasks:
                num_classes = TASKS_NUM_CLASSES[task]
                if f"{task}_true" not in row:
                    row[f"{task}_true"] = None
                if f"{task}_pred" not in row:
                    row[f"{task}_pred"] = None
                for c in range(num_classes):
                    if f"{task}_prob_{c}" not in row:
                        row[f"{task}_prob_{c}"] = None

            rows.append(row)

    if not rows:
        print("  [WARN] No se generaron filas. Comprueba los paths y las configuraciones.")
        return

    # Orden de columnas: wsi_stem, split, luego tareas en orden canónico
    col_order = ["wsi_stem", "split"]
    for task in args.tasks:
        num_classes = TASKS_NUM_CLASSES[task]
        col_order.append(f"{task}_true")
        col_order.append(f"{task}_pred")
        for c in range(num_classes):
            col_order.append(f"{task}_prob_{c}")

    df_out = pd.DataFrame(rows)

    # Añadir columnas que falten (puede ocurrir si alguna tarea se saltó)
    for col in col_order:
        if col not in df_out.columns:
            df_out[col] = None

    df_out = df_out[col_order]

    # ── Guardar ───────────────────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)
    csv_path = os.path.join(args.output_path, args.output_filename)
    df_out.to_csv(csv_path, index=False)

    print(f"\n  CSV guardado: {csv_path}")
    print(f"  Shape: {df_out.shape}  ({df_out['split'].value_counts().to_dict()})")

    # Resumen rápido de aciertos por tarea y split
    print("\n  === Resumen de predicciones ===")
    for task in args.tasks:
        true_col = f"{task}_true"
        pred_col = f"{task}_pred"
        if true_col not in df_out.columns or pred_col not in df_out.columns:
            continue
        for split_name in splits:
            mask = (
                (df_out["split"] == split_name)
                & df_out[true_col].notna()
                & df_out[pred_col].notna()
                & (df_out[true_col] != -1)
            )
            sub = df_out[mask]
            if sub.empty:
                continue
            correct = (sub[true_col] == sub[pred_col]).sum()
            acc     = correct / len(sub)
            print(
                f"  [{split_name}] {task}: "
                f"{correct}/{len(sub)} correctas  (acc={acc:.3f})"
            )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_multiple_seed(args.seed)

    parent = os.path.dirname(os.path.abspath(args.test_results_path))
    if args.output_path is None:
        args.output_path = os.path.join(parent, "predictions")
    os.makedirs(args.output_path, exist_ok=True)

    logger = TeeLogger(os.path.join(args.output_path, "predict_log.txt"))
    logger.start()
    try:
        run_predictions(args)
    finally:
        logger.stop()

    print(f"\n[DONE] Log guardado en: {os.path.join(args.output_path, 'predict_log.txt')}")


if __name__ == "__main__":
    main()
