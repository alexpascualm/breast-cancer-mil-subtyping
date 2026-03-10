import re
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
    v = str(value).strip()
    if v not in MOLECULAR_SUBTYPE_MAP:
        raise ValueError(
            f"Unknown molecular subtype '{v}'. "
            f"Expected one of {list(MOLECULAR_SUBTYPE_MAP.keys())}"
        )
    return MOLECULAR_SUBTYPE_MAP[v]
