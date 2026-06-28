"""
subtype_from_biomarkers.py
---------------------------
Toma como entrada el CSV generado por predict_all_wsis.py y calcula el subtipo
molecular inferido a partir de los biomarcadores IHC, tanto los reales como
los predichos, siguiendo la clasificación surrogate (St. Gallen / IHC):

    ┌──────────────┬──────────┬──────────┬──────────────┬───────────────┐
    │ Subtipo      │ ER       │ PR       │ HER2         │ Proliferación │
    ├──────────────┼──────────┼──────────┼──────────────┼───────────────┤
    │ Luminal A    │ positive │ positive │ negative     │ low (ki67=0)  │
    │ Luminal B    │ positive │ any      │ any          │ high (ki67=1) │
    │ HER2-positive│ negative │ negative │ overexpressed│ -             │
    │ TNBC         │ negative │ negative │ negative     │ -             │
    └──────────────┴──────────┴──────────┴──────────────┴───────────────┘

Nota sobre la lógica de prioridad (orden de evaluación):
  1. ER negativo + PR negativo + HER2 positivo  → HER2-positive
  2. ER negativo + PR negativo + HER2 negativo  → TNBC
  3. ER positivo + ki67 bajo                    → Luminal A
  4. ER positivo + ki67 alto                    → Luminal B
  (cualquier caso con ER positivo y PR negativo o HER2 positivo cae en Luminal B)

Convención de etiquetas en el CSV (binarias, clase 1 = positivo/alto):
  er_pred / er_true         : 0=negativo, 1=positivo
  pr_pred / pr_true         : 0=negativo, 1=positivo
  her2_pred / her2_true     : 0=negativo, 1=positivo/overexpressed
  ki67_pred / ki67_true     : 0=bajo,     1=alto

Salidas
-------
  subtype_analysis.csv
      CSV con las columnas originales más:
        subtype_from_true_biomarkers   ← subtipo inferido de biomarcadores reales
        subtype_from_pred_biomarkers   ← subtipo inferido de biomarcadores predichos

  subtype_report.txt
      - Reporte 1: subtype_from_pred_biomarkers vs subtype_from_true_biomarkers
      - Reporte 2: subtype_from_pred_biomarkers vs molecular_subtype_true (etiqueta original)
      - Análisis de concordancia: subtype_from_true_biomarkers vs molecular_subtype_true

Uso
---
  python subtype_from_biomarkers.py --input_csv /path/to/predictions.csv

  # Con salida personalizada:
  python subtype_from_biomarkers.py \\
      --input_csv /path/to/predictions.csv \\
      --output_dir /path/to/output \\
      --split test_set
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

# Mapeo numérico de subtipos (coincide con el usado en el dataset BCNB)
#   0 = Luminal A
#   1 = Luminal B
#   2 = HER2-positive
#   3 = TNBC
SUBTYPE_NAMES = {
    0: "Luminal A",
    1: "Luminal B",
    2: "HER2-positive",
    3: "TNBC",
}
SUBTYPE_ORDER = [0, 1, 2, 3]
SUBTYPE_LABELS = [SUBTYPE_NAMES[k] for k in SUBTYPE_ORDER]


# ──────────────────────────────────────────────────────────────────────────────
# LÓGICA DE SUBTIPADO IHC SURROGATE
# ──────────────────────────────────────────────────────────────────────────────

def infer_subtype(er: int, pr: int, her2: int, ki67: int) -> int | None:
    """
    Infiere el subtipo molecular a partir de los biomarcadores IHC binarios,
    siguiendo la clasificación surrogate de la Tabla 1.

    Parámetros
    ----------
    er   : 0=negativo, 1=positivo
    pr   : 0=negativo, 1=positivo
    her2 : 0=negativo, 1=overexpressed/positivo
    ki67 : 0=bajo,     1=alto

    Retorna
    -------
    int  : 0=Luminal A, 1=Luminal B, 2=HER2-positive, 3=TNBC
    None : si algún biomarcador es NaN / inválido

    Lógica de prioridad
    -------------------
    La condición de ER negativo se evalúa antes que la de ER positivo.
    Dentro de ER negativo, HER2 determina si es HER2-positive o TNBC.
    Dentro de ER positivo, ki67 determina Luminal A (bajo) o Luminal B (alto).
    """
    # Validar que no haya NaN
    for val in (er, pr, her2, ki67):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None

    er   = int(er)
    pr   = int(pr)
    her2 = int(her2)
    ki67 = int(ki67)

    # ── Rama ER negativo ──────────────────────────────────────────────────────
    if er == 0:
        if pr == 0:
            if her2 == 1:
                return 2   # HER2-positive
            else:
                return 3   # TNBC
        else:
            # ER negativo + PR positivo: caso atípico no cubierto explícitamente
            # en la tabla; se asigna HER2-positive si HER2=1, TNBC si HER2=0
            # (comportamiento conservador: se prioriza HER2 sobre PR en ausencia de ER)
            if her2 == 1:
                return 2
            else:
                return 3

    # ── Rama ER positivo ─────────────────────────────────────────────────────
    # ER=1: Luminal A o Luminal B según ki67
    if ki67 == 0:
        return 0   # Luminal A
    else:
        return 1   # Luminal B


def infer_subtype_row(row: pd.Series, suffix: str) -> int | None:
    """
    Extrae los biomarcadores de una fila del CSV usando el sufijo indicado
    ('true' o 'pred') y devuelve el subtipo inferido.
    """
    try:
        er   = row[f"er_{suffix}"]
        pr   = row[f"pr_{suffix}"]
        her2 = row[f"her2_{suffix}"]
        ki67 = row[f"ki67_{suffix}"]
    except KeyError as e:
        raise KeyError(
            f"Columna no encontrada: {e}. "
            "Asegúrate de que el CSV contiene columnas er/pr/her2/ki67 "
            f"con sufijo '_{suffix}'."
        )
    return infer_subtype(er, pr, her2, ki67)


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT FORMATEADO
# ──────────────────────────────────────────────────────────────────────────────

def fmt_classification_report(
    y_true,
    y_pred,
    title: str,
    target_names: list[str] | None = None,
    labels: list[int] | None = None,
) -> str:
    """
    Genera un bloque de texto con título + classification report + matriz de confusión.
    Filtra automáticamente las filas donde y_true o y_pred sea NaN/None.
    """
    # Filtrar NaN
    mask  = [
        (t is not None and p is not None
         and not (isinstance(t, float) and np.isnan(t))
         and not (isinstance(p, float) and np.isnan(p)))
        for t, p in zip(y_true, y_pred)
    ]
    yt = [int(t) for t, m in zip(y_true, mask) if m]
    yp = [int(p) for p, m in zip(y_pred, mask) if m]

    n_skipped = len(y_true) - len(yt)
    lines = []
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)
    lines.append(f"  N = {len(yt)}  (omitidas por NaN: {n_skipped})\n")

    if len(yt) == 0:
        lines.append("  [Sin datos suficientes para generar el reporte.]")
        return "\n".join(lines)

    # Classification report
    report = classification_report(
        yt, yp,
        labels       = labels or SUBTYPE_ORDER,
        target_names = target_names or SUBTYPE_LABELS,
        zero_division = 0,
    )
    lines.append(report)

    # Matriz de confusión
    cm = confusion_matrix(yt, yp, labels=labels or SUBTYPE_ORDER)
    lines.append("Matriz de confusión (filas=real, columnas=predicho):")
    header = "  " + "  ".join(f"{n:>13}" for n in (target_names or SUBTYPE_LABELS))
    lines.append(header)
    for i, row_cm in enumerate(cm):
        row_label = (target_names or SUBTYPE_LABELS)[i]
        row_str   = "  ".join(f"{v:>13}" for v in row_cm)
        lines.append(f"  {row_label:<14} {row_str}")

    lines.append("")
    return "\n".join(lines)


def fmt_concordance_analysis(df_s: pd.DataFrame, has_original: bool) -> list[str]:
    """
    Genera el bloque de análisis de concordancia entre el subtipo inferido
    desde biomarcadores reales y la etiqueta original molecular_subtype_true.

    Retorna una lista de strings lista para añadir a report_blocks.
    """
    lines = []
    lines.append("─" * 70)
    lines.append("  ANÁLISIS DE CONCORDANCIA: SUBTIPO IHC REAL vs ETIQUETA ORIGINAL")
    lines.append("─" * 70)

    if not has_original:
        lines.append(
            "\n  [INFO] Columna 'molecular_subtype_true' no disponible o sin valores válidos.\n"
            "  Se omite el análisis de concordancia.\n"
        )
        return lines

    mask_valid = (
        df_s["molecular_subtype_true"].notna()
        & (df_s["molecular_subtype_true"] != -1)
        & df_s["subtype_from_true_biomarkers"].notna()
    )
    df_conc = df_s[mask_valid].copy()

    if df_conc.empty:
        lines.append("\n  [INFO] Sin filas válidas para el análisis de concordancia.\n")
        return lines

    n_total = len(df_conc)
    agree_mask = (
        df_conc["molecular_subtype_true"].astype(int)
        == df_conc["subtype_from_true_biomarkers"].astype(int)
    )
    n_agree = agree_mask.sum()
    pct_agree = n_agree / n_total * 100

    lines.append(
        f"\n  Concordancia global (subtipo IHC real == etiqueta original): "
        f"{n_agree}/{n_total} ({pct_agree:.1f}%)\n"
    )

    # Tabla de concordancia por subtipo
    lines.append("  Detalle por subtipo (según etiqueta original):")
    header = f"  {'Subtipo':<15}  {'N':<6}  {'Concordantes':<13}  {'Discordantes':<13}  {'% Conc.'}"
    lines.append(header)
    lines.append("  " + "-" * 62)

    for sub_id in SUBTYPE_ORDER:
        sub_name = SUBTYPE_NAMES[sub_id]
        sub_mask = df_conc["molecular_subtype_true"].astype(int) == sub_id
        sub_df   = df_conc[sub_mask]
        n_sub    = len(sub_df)
        if n_sub == 0:
            continue
        n_conc  = (sub_df["subtype_from_true_biomarkers"].astype(int) == sub_id).sum()
        n_disc  = n_sub - n_conc
        pct_sub = n_conc / n_sub * 100
        lines.append(
            f"  {sub_name:<15}  {n_sub:<6}  {n_conc:<13}  {n_disc:<13}  {pct_sub:.1f}%"
        )

    lines.append("")

    # Tabla de discordancias: para cada subtipo original, a qué subtipo IHC se asigna
    lines.append("  Tabla de confusión de concordancia (filas=etiqueta original, columnas=subtipo IHC real):")
    col_header = "  " + f"{'':15}" + "  ".join(f"{n:>13}" for n in SUBTYPE_LABELS)
    lines.append(col_header)

    y_orig = df_conc["molecular_subtype_true"].astype(int).tolist()
    y_ihc  = df_conc["subtype_from_true_biomarkers"].astype(int).tolist()
    cm     = confusion_matrix(y_orig, y_ihc, labels=SUBTYPE_ORDER)

    for i, row_cm in enumerate(cm):
        row_label = SUBTYPE_LABELS[i]
        row_str   = "  ".join(f"{v:>13}" for v in row_cm)
        lines.append(f"  {row_label:<15} {row_str}")

    lines.append("")

    # Desglose de discordancias por subtipo
    lines.append("  Desglose de discordancias (casos etiqueta original ≠ subtipo IHC real):")
    df_disc = df_conc[~agree_mask]
    if df_disc.empty:
        lines.append("  ¡Sin discordancias!\n")
        return lines

    for sub_id in SUBTYPE_ORDER:
        sub_name = SUBTYPE_NAMES[sub_id]
        sub_mask = df_disc["molecular_subtype_true"].astype(int) == sub_id
        sub_df   = df_disc[sub_mask]
        if sub_df.empty:
            continue
        lines.append(f"\n  Etiqueta original = {sub_name} ({len(sub_df)} discordantes):")
        assigned = (
            sub_df["subtype_from_true_biomarkers"]
            .map(SUBTYPE_NAMES)
            .value_counts()
        )
        for assigned_name, count in assigned.items():
            lines.append(f"    → {assigned_name:<15}: {count} casos")

    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Infiere el subtipo molecular desde biomarcadores IHC (reales y predichos)\n"
            "y genera reportes de clasificación y comparativas."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Ruta al CSV generado por predict_all_wsis.py."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Directorio de salida. Por defecto: mismo directorio que input_csv."
        )
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["dev_set", "test_set"],
        help=(
            "Si se indica, filtra el CSV al split especificado.\n"
            "Por defecto se procesan todos los splits por separado y en conjunto."
        )
    )
    parser.add_argument(
        "--output_csv", type=str, default="subtype_analysis.csv",
        help="Nombre del CSV de salida enriquecido (default: subtype_analysis.csv)."
    )
    parser.add_argument(
        "--output_report", type=str, default="subtype_report.txt",
        help="Nombre del fichero de texto con los reportes (default: subtype_report.txt)."
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Leer CSV ──────────────────────────────────────────────────────────────
    if not os.path.isfile(args.input_csv):
        print(f"[ERROR] CSV no encontrado: {args.input_csv}")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    print(f"[OK] CSV cargado: {args.input_csv}  →  {df.shape[0]} filas, {df.shape[1]} columnas")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_csv))
    os.makedirs(out_dir, exist_ok=True)

    # ── Filtrar split si se pide ───────────────────────────────────────────────
    if args.split:
        df = df[df["split"] == args.split].reset_index(drop=True)
        print(f"[FILTER] Split '{args.split}': {len(df)} WSIs")

    # ── Inferir subtipos desde biomarcadores ──────────────────────────────────
    print("\n[INFO] Infiriendo subtipos desde biomarcadores...")

    df["subtype_from_true_biomarkers"] = df.apply(
        lambda row: infer_subtype_row(row, "true"), axis=1
    )
    df["subtype_from_pred_biomarkers"] = df.apply(
        lambda row: infer_subtype_row(row, "pred"), axis=1
    )

    n_true_ok = df["subtype_from_true_biomarkers"].notna().sum()
    n_pred_ok = df["subtype_from_pred_biomarkers"].notna().sum()
    print(f"  subtype_from_true_biomarkers: {n_true_ok} válidos / {len(df)} totales")
    print(f"  subtype_from_pred_biomarkers: {n_pred_ok} válidos / {len(df)} totales")

    # Distribución de subtipos inferidos
    print("\n  Distribución subtype_from_true_biomarkers:")
    dist_true = (
        df["subtype_from_true_biomarkers"]
        .map(SUBTYPE_NAMES)
        .value_counts()
        .reindex(SUBTYPE_LABELS, fill_value=0)
    )
    for name, count in dist_true.items():
        print(f"    {name:<15}: {count}")

    print("\n  Distribución subtype_from_pred_biomarkers:")
    dist_pred = (
        df["subtype_from_pred_biomarkers"]
        .map(SUBTYPE_NAMES)
        .value_counts()
        .reindex(SUBTYPE_LABELS, fill_value=0)
    )
    for name, count in dist_pred.items():
        print(f"    {name:<15}: {count}")

    # ── Guardar CSV enriquecido ───────────────────────────────────────────────
    csv_out = os.path.join(out_dir, args.output_csv)
    df.to_csv(csv_out, index=False)
    print(f"\n[OK] CSV enriquecido guardado: {csv_out}")

    # ── Construir bloques de texto por split ──────────────────────────────────
    splits_present = sorted(df["split"].unique()) if "split" in df.columns else ["all"]
    report_blocks  = []

    report_blocks.append("=" * 70)
    report_blocks.append("  ANÁLISIS DE SUBTIPO MOLECULAR DESDE BIOMARCADORES IHC")
    report_blocks.append("=" * 70)
    report_blocks.append(
        "\nLógica de subtipado surrogate (Tabla 1):\n"
        "  ER+  ki67 bajo              → Luminal A\n"
        "  ER+  ki67 alto              → Luminal B\n"
        "  ER-  PR-   HER2+            → HER2-positive\n"
        "  ER-  PR-   HER2-            → TNBC\n"
        "\nConvención de labels binarios:\n"
        "  er/pr/her2 : 0=negativo, 1=positivo\n"
        "  ki67       : 0=bajo,     1=alto\n"
    )

    for split_name in splits_present:
        df_s = df[df["split"] == split_name] if "split" in df.columns else df
        n    = len(df_s)

        report_blocks.append(f"\n{'#'*70}")
        report_blocks.append(f"  SPLIT: {split_name}  (N={n})")
        report_blocks.append(f"{'#'*70}")

        has_original = (
            "molecular_subtype_true" in df_s.columns
            and df_s["molecular_subtype_true"].notna().any()
            and (df_s["molecular_subtype_true"] != -1).any()
        )

        # ── Reporte 1: subtipo predicho (biomarcadores predichos) vs
        #              subtipo real (biomarcadores reales) ─────────────────────
        report_blocks.append(
            fmt_classification_report(
                y_true       = df_s["subtype_from_true_biomarkers"].tolist(),
                y_pred       = df_s["subtype_from_pred_biomarkers"].tolist(),
                title        = (
                    "REPORTE 1 — Subtipo predicho (biomarcadores predichos) vs\n"
                    "            Subtipo real (biomarcadores reales)\n"
                    "  → Mide la capacidad del modelo para inferir el subtipo\n"
                    "    correcto a través de la predicción de biomarcadores."
                ),
            )
        )

        # ── Reporte 2: subtipo predicho (biomarcadores predichos) vs
        #              etiqueta original molecular_subtype_true ────────────────
        if has_original:
            mask_valid = (
                df_s["molecular_subtype_true"].notna()
                & (df_s["molecular_subtype_true"] != -1)
            )
            df_comp = df_s[mask_valid]

            report_blocks.append(
                fmt_classification_report(
                    y_true       = df_comp["molecular_subtype_true"].tolist(),
                    y_pred       = df_comp["subtype_from_pred_biomarkers"].tolist(),
                    title        = (
                        "REPORTE 2 — Subtipo predicho (biomarcadores predichos) vs\n"
                        "            Etiqueta original molecular_subtype_true\n"
                        "  → Mide directamente si el subtipo inferido por el modelo\n"
                        "    coincide con la etiqueta clínica de referencia."
                    ),
                )
            )
        else:
            report_blocks.append(
                "\n  [INFO] Columna 'molecular_subtype_true' no disponible o sin valores válidos.\n"
                "  Se omite el Reporte 2.\n"
            )

        # ── Análisis de concordancia: subtipo IHC real vs etiqueta original ───
        report_blocks.extend(fmt_concordance_analysis(df_s, has_original))

        # ── Tabla resumen: distribuciones ─────────────────────────────────────
        report_blocks.append("─" * 70)
        report_blocks.append("  DISTRIBUCIÓN DE SUBTIPOS")
        report_blocks.append("─" * 70)

        rows_dist = []
        for sub_id in SUBTYPE_ORDER:
            sub_name = SUBTYPE_NAMES[sub_id]
            n_true_b = (df_s["subtype_from_true_biomarkers"] == sub_id).sum()
            n_pred_b = (df_s["subtype_from_pred_biomarkers"] == sub_id).sum()
            row_d    = {
                "Subtipo"                        : sub_name,
                "Biomarcadores reales (N)"        : int(n_true_b),
                "Biomarcadores predichos (N)"     : int(n_pred_b),
            }
            if has_original:
                mask_orig = (
                    df_s["molecular_subtype_true"].notna()
                    & (df_s["molecular_subtype_true"] != -1)
                )
                n_orig = (df_s[mask_orig]["molecular_subtype_true"].astype(int) == sub_id).sum()
                row_d["Etiqueta original (N)"] = int(n_orig)
            rows_dist.append(row_d)

        df_dist = pd.DataFrame(rows_dist).set_index("Subtipo")
        report_blocks.append(df_dist.to_string())
        report_blocks.append("")

    # ── Guardar reporte ───────────────────────────────────────────────────────
    report_text = "\n".join(report_blocks)
    report_out  = os.path.join(out_dir, args.output_report)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[OK] Reporte guardado: {report_out}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()