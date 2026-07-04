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

Uso mínimo
----------
  python predict_all_wsis.py --test_results_path /workspace/results/my_run/test_results

Con dev set:
  python predict_all_wsis.py \\
      --test_results_path /workspace/results/my_run/test_results \\
      --use_dev_set

Nota
----
Este script comparte todo el scaffolding (splits, dataset, carga de modelo,
iteración de WSIs, forward de la bag) con generate_heatmaps.py a través de
wsi_pipeline_common.py. Solo implementa aquí lo específico: construir filas
de un CSV a partir de (pred_label, probs) por tarea.
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    bcnb_images_folder_path,
    bcnb_labels_df_path,
    bcnb_embeddings_path,
)
from utils.train_utils import set_multiple_seed
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

TASKS_NUM_CLASSES = {
    "er": 2,
    "pr": 2,
    "her2": 2,
    "ki67": 2,
    "molecular_subtype": 4,
}
ALL_TASKS = list(TASKS_NUM_CLASSES.keys())


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
        help="Carpeta de salida. Por defecto: <parent de test_results_path>/predictions/"
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
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


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
) -> dict[str, dict]:
    """
    Ejecuta inferencia sobre todas las WSIs de un split para una tarea dada.

    Returns
    -------
    dict[wsi_stem -> {"<task>_true": int, "<task>_pred": int, "<task>_prob_<c>": float, ...}]
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

    print(f"    Inferencia sobre {n} WSIs...")

    samples = iterate_wsi_dataset(
        dataset, df_valid, use_precomputed, batch_size=1, num_workers=num_workers
    )

    for i, sample in enumerate(samples):
        if sample.true_label == -1:
            entry = {f"{task}_true": -1, f"{task}_pred": None}
            for c in range(num_classes):
                entry[f"{task}_prob_{c}"] = None
            records[sample.wsi_stem] = entry
            continue

        try:
            pred_label, probs = forward_bag(model, sample.features, device, use_precomputed)
        except Exception as e:
            print(f"    [ERROR] Inferencia fallida para {sample.wsi_stem}: {e}")
            continue

        entry = {f"{task}_true": sample.true_label, f"{task}_pred": pred_label}
        for c, p in enumerate(probs):
            entry[f"{task}_prob_{c}"] = float(p)
        records[sample.wsi_stem] = entry

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"      {i + 1}/{n} procesadas...")

    return records


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

    dev_df, test_df = load_splits(
        args.test_results_path, args.use_dev_set, args.dev_df_path, args.test_df_path
    )

    splits: dict[str, pd.DataFrame] = {"test_set": test_df}
    if args.use_dev_set and dev_df is not None:
        splits["dev_set"] = dev_df

    print(f"[CONFIG] Embeddings precomputados: {args.embeddings_folder is not None}")

    # accumulated[split_name][wsi_stem] = {col: val, ...}
    accumulated: dict[str, dict[str, dict]] = {s: {} for s in splits}

    for ctx in for_each_task_and_method(
        test_results_path=args.test_results_path,
        tasks=args.tasks,
        num_classes_fn=lambda t: TASKS_NUM_CLASSES[t],
        device=device,
        embeddings_folder=args.embeddings_folder,
    ):
        print(f"\n{'='*60}")
        print(f"  TAREA: {ctx.task}  ({ctx.num_classes} clases)")
        print(f"{'='*60}")

        for split_name, df_split in splits.items():
            print(f"\n  [{split_name}] {len(df_split)} WSIs...")

            records = infer_split(
                df_split=df_split,
                task=ctx.task,
                num_classes=ctx.num_classes,
                model=ctx.model,
                use_precomputed=ctx.use_precomputed,
                emb_folder_enc=ctx.emb_folder_enc,
                device=device,
                num_workers=args.num_workers,
            )

            for wsi_stem, task_entry in records.items():
                accumulated[split_name].setdefault(wsi_stem, {}).update(task_entry)

            print(f"  [{split_name}] {len(records)} WSIs procesadas para '{ctx.task}'.")

    # ── Construir DataFrame final ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Construyendo CSV final...")
    print(f"{'='*60}")

    rows = []
    for split_name, df_split in splits.items():
        for _, row_df in df_split.iterrows():
            features_col = row_df.get("features", None)
            if features_col is not None:
                wsi_stem = os.path.splitext(os.path.basename(str(features_col)))[0]
            elif args.slide_col in row_df:
                wsi_stem = str(row_df[args.slide_col])
            else:
                print(f"  [WARN] No se pudo determinar wsi_stem para una fila de {split_name}.")
                continue

            row = {"wsi_stem": wsi_stem, "split": split_name}
            row.update(accumulated[split_name].get(wsi_stem, {}))

            for task in args.tasks:
                num_classes = TASKS_NUM_CLASSES[task]
                row.setdefault(f"{task}_true", None)
                row.setdefault(f"{task}_pred", None)
                for c in range(num_classes):
                    row.setdefault(f"{task}_prob_{c}", None)

            rows.append(row)

    if not rows:
        print("  [WARN] No se generaron filas. Comprueba los paths y las configuraciones.")
        return

    col_order = ["wsi_stem", "split"]
    for task in args.tasks:
        num_classes = TASKS_NUM_CLASSES[task]
        col_order.append(f"{task}_true")
        col_order.append(f"{task}_pred")
        for c in range(num_classes):
            col_order.append(f"{task}_prob_{c}")

    df_out = pd.DataFrame(rows)
    for col in col_order:
        if col not in df_out.columns:
            df_out[col] = None
    df_out = df_out[col_order]

    os.makedirs(args.output_path, exist_ok=True)
    csv_path = os.path.join(args.output_path, args.output_filename)
    df_out.to_csv(csv_path, index=False)

    print(f"\n  CSV guardado: {csv_path}")
    print(f"  Shape: {df_out.shape}  ({df_out['split'].value_counts().to_dict()})")

    print("\n  === Resumen de predicciones ===")
    for task in args.tasks:
        true_col, pred_col = f"{task}_true", f"{task}_pred"
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
            acc = correct / len(sub)
            print(f"  [{split_name}] {task}: {correct}/{len(sub)} correctas  (acc={acc:.3f})")


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

    with TeeLogger(os.path.join(args.output_path, "predict_log.txt")):
        run_predictions(args)

    print(f"\n[DONE] Log guardado en: {os.path.join(args.output_path, 'predict_log.txt')}")


if __name__ == "__main__":
    main()
