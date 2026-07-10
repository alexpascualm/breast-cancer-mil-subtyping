import re
import numpy as np
# ──────────────────────────────────────────────────────────────────────────────
# LABEL MAPPINGS
# ──────────────────────────────────────────────────────────────────────────────

MOLECULAR_SUBTYPE_MAP = {
    'Luminal A':       0,
    'Luminal B':       1,
    'HER2(+)':         2,
    'Triple negative': 3,
}

# Biomarkers with individual binary classification (one model per biomarker)
# Valid task values: 'er', 'pr', 'her2', 'ki67'
BIOMARKER_COLS = ['er', 'pr', 'her2', 'ki67']
BIOMARKER_BINARY_MAP = {'Positive': 1, 'Negative': 0}

# Ki67 threshold for binarisation (standard clinical cut-offs: 14% or 20%)
# Low proliferation (< threshold) → 0,  High proliferation (>= threshold) → 1
KI67_THRESHOLD = 0.14   # ← change to 0.20 if your centre uses the 20% cut-off

def map_ki67(value, threshold: float = KI67_THRESHOLD) -> int:
    """
    Binarise Ki67 proliferation index.

    Handles formats found in the Excel:
      - Plain float/int:   0.01  →  1%
      - Percentage string: '14%' → 14%
      - Range string:      '30-40%' → midpoint = 35%
      - Inequality:        '>50%' or '＞50%' → 50%  (conservative: use the number itself)

    Returns 0 (low proliferation) or 1 (high). Returns -1 if unparseable.
    """
    try:
        v = str(value).strip()

        # Normalise unicode '＞' / '＜' to ASCII
        v = v.replace('＞', '>').replace('＜', '<')

        # Remove inequality signs ('>50%' → '50%')
        v = re.sub(r'^[<>]=?\s*', '', v)

        # Handle ranges: '30-40%' → midpoint 35
        range_match = re.match(r'^([\d.]+)\s*[-–]\s*([\d.]+)\s*%?$', v)
        if range_match:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            ki67_float = (lo + hi) / 2.0
        else:
            # Plain number or percentage string: '14%' or '0.14' or '14'
            ki67_float = float(v.rstrip('%'))

        # Normalise: if stored as percentage integer (e.g. 14 → 0.14)
        if ki67_float > 1.0:
            ki67_float /= 100.0

        return int(ki67_float >= threshold)

    except (ValueError, TypeError, AttributeError):
        return -1


def map_molecular_subtype(value: str) -> int:
    """
    Maps the clinical 'Molecular subtype' Excel value to its int label.
    Returns -1 (missing/unparseable) if `value` isn't one of
    MOLECULAR_SUBTYPE_MAP's keys, matching the sentinel convention used by
    map_ki67() and the biomarker columns (BIOMARKER_BINARY_MAP.get(..., -1)).
    """
    v = str(value).strip()
    return MOLECULAR_SUBTYPE_MAP.get(v, -1)


def infer_subtype_from_biomarkers(er, pr, her2, ki67, missing_value=-1):
    """
    Infers the surrogate molecular subtype from binarised IHC biomarkers
    (ER/PR/HER2/Ki67):
      ER-  HER2+     → HER2(+)         (2)
      ER-  HER2-     → Triple negative (3)
      ER+  ki67 low  → Luminal A       (0)
      ER+  ki67 high → Luminal B       (1)
    (PR does not change the outcome in this surrogate rule.)

    Single source of truth for this rule: used both to build the training
    dataframe (data_load.py, `missing_value=-1` to match its int sentinel
    convention) and by tools/subtype_from_biomarkers.py's post-hoc analysis
    (`missing_value=None`, to keep its pandas NaN-based filtering).

    Parameters
    ----------
    er, pr, her2 : 0=negative, 1=positive
    ki67         : 0=low, 1=high
    missing_value
        Value returned if any biomarker is missing (None, NaN, or -1).

    Returns
    -------
    One of MOLECULAR_SUBTYPE_MAP's values (0-3), or `missing_value` if any
    biomarker is missing/unparseable.
    """
    def _is_missing(v):
        return v is None or v == -1 or (isinstance(v, float) and np.isnan(v))

    if any(_is_missing(v) for v in (er, pr, her2, ki67)):
        return missing_value

    er, pr, her2, ki67 = int(er), int(pr), int(her2), int(ki67)

    if er == 0:
        return 2 if her2 == 1 else 3   # HER2(+) / Triple negative

    return 1 if ki67 == 1 else 0       # Luminal B / Luminal A