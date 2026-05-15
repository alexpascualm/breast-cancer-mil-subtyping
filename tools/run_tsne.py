# run_tsne.py
"""
Carga los embeddings .h5 generados por extract_embeddings.py,
aplica t-SNE y guarda un HTML interactivo listo para abrir en el navegador.

Uso básico:
    python run_tsne.py --patch_model dino --color_col ki67

Nivel WSI (un punto por WSI, mean-pooling):
    python run_tsne.py --patch_model dino --color_col grade --level wsi

Nivel patch (un punto por parche, con muestreo opcional):
    python run_tsne.py --patch_model uni --color_col subtype \
        --level patch --sample_per_wsi 50
"""

import argparse
import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_load import create_file_label_df
from config import bcnb_images_folder_path, bcnb_labels_df_path, bcnb_embeddings_path


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE sobre embeddings → HTML interactivo"
    )
    parser.add_argument("--patch_model",       type=str, default="dino",
                        choices=["dino", "uni"])
    parser.add_argument("--embeddings_folder", type=str, default=bcnb_embeddings_path)
    parser.add_argument("--images_folder",     type=str, default=bcnb_images_folder_path)
    parser.add_argument("--df_path",           type=str, default=bcnb_labels_df_path)
    parser.add_argument("--output_html",       type=str, default=None,
                        help="Ruta del HTML. Por defecto: "
                             "/workspace/tools/tsne_results/<patch_model>/tsne_<color_col>_<level>.html")

    # Columna inicial para colorear (opcional)
    parser.add_argument("--color_col",         type=str, default=None,
                        choices=["molecular_subtype", "er", "pr", "her2", "ki67"],
                        help="Columna inicial para colorear (opcional, se puede cambiar en el HTML)")

    # Columnas extra que aparecen en el tooltip al pasar el ratón
    parser.add_argument("--tooltip_cols",      type=str, nargs="*", default=None,
                        help="Columnas adicionales a mostrar en el tooltip "
                             "(además de wsi_id y color_col). None = todas las del df.")

    # Granularidad
    parser.add_argument("--level",             type=str, default="wsi",
                        choices=["patch", "wsi"])
    parser.add_argument("--sample_per_wsi",    type=int, default=None,
                        help="(solo --level patch) Máx. parches a muestrear por WSI")

    # t-SNE / PCA
    parser.add_argument("--pca_components",    type=int,   default=50)
    parser.add_argument("--perplexity",        type=float, default=30.0)
    parser.add_argument("--n_iter",            type=int,   default=1000)
    parser.add_argument("--random_state",      type=int,   default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_embeddings(df, model_dir, level, sample_per_wsi,
                    color_col, tooltip_cols):
    SKIP_COLS = {"features"}
    if tooltip_cols is None:
        tooltip_cols = [c for c in df.columns if c not in SKIP_COLS]
    if color_col and color_col not in tooltip_cols:
        tooltip_cols = [color_col] + tooltip_cols

    all_embs = []
    all_meta = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Cargando embeddings", file=sys.stdout):
        h5_src = row["features"]
        wsi_id = os.path.splitext(os.path.basename(h5_src))[0]
        h5_path = os.path.join(model_dir, f"{wsi_id}.h5")

        if not os.path.exists(h5_path):
            print(f"  [WARN] No encontrado: {h5_path} — saltando", flush=True)
            continue

        try:
            with h5py.File(h5_path, "r") as h5:
                embs   = h5["embeddings"][:]
                coords = h5["coords"][:] if "coords" in h5 else None
        except Exception as e:
            print(f"  [ERROR] {wsi_id}: {e}", flush=True)
            continue

        n = len(embs)

        if level == "wsi":
            embs   = embs.mean(axis=0, keepdims=True)
            coords = None
            n      = 1
        elif sample_per_wsi and n > sample_per_wsi:
            idx    = np.random.choice(n, size=sample_per_wsi, replace=False)
            embs   = embs[idx]
            coords = coords[idx] if coords is not None else None
            n      = sample_per_wsi

        all_embs.append(embs)

        for i in range(n):
            d = {"wsi_id": wsi_id}
            if coords is not None:
                d["coord_x"] = int(coords[i, 0])
                d["coord_y"] = int(coords[i, 1])
            for col in tooltip_cols:
                d[col] = row[col]
            all_meta.append(d)

    embeddings = np.vstack(all_embs).astype(np.float32)
    meta_df    = pd.DataFrame(all_meta)
    return embeddings, meta_df, tooltip_cols


# ---------------------------------------------------------------------------
# Reducción
# ---------------------------------------------------------------------------

def reduce(embeddings, pca_components, perplexity, n_iter, random_state):
    X = embeddings
    if pca_components > 0 and X.shape[1] > pca_components:
        print(f"PCA: {X.shape[1]}D → {pca_components}D …", flush=True)
        pca = PCA(n_components=pca_components, random_state=random_state)
        X   = pca.fit_transform(X)
        print(f"  Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}", flush=True)

    print(f"t-SNE: {X.shape[1]}D → 2D  (n={X.shape[0]}, perplexity={perplexity}) …", flush=True)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=random_state, verbose=1)
    return tsne.fit_transform(X)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

PALETTE = [
    "#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#8ecae6",
    "#a8dadc", "#f1c453", "#c77dff", "#06d6a0", "#fb8500",
    "#ef233c", "#4cc9f0", "#b5179e", "#80b918", "#ffd60a",
    "#ff6b6b", "#48cae4", "#d62828", "#023e8a", "#aacc00",
]


def _safe(v):
    """Convierte cualquier valor a string limpio para HTML/JSON."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return str(v)


def build_html(tsne_coords, meta_df, color_col, tooltip_cols,
               patch_model, level, perplexity, n_iter):

    available_cols = [c for c in tooltip_cols if c in meta_df.columns and c != "wsi_id"]
    initial_col = color_col if (color_col and color_col in available_cols) else available_cols[0]

    points = []
    for i, (x, y) in enumerate(tsne_coords):
        row = meta_df.iloc[i]
        tip = {}
        for col in tooltip_cols:
            if col in meta_df.columns:
                tip[col] = _safe(row[col])
        points.append({"x": float(x), "y": float(y), "tip": tip})

    data_json        = json.dumps(points)
    cols_json        = json.dumps(available_cols)
    initial_col_json = json.dumps(initial_col)
    palette_json     = json.dumps(PALETTE)

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>t-SNE · {patch_model}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  /* ── Dark theme (default) ── */
  :root, [data-theme="dark"] {{
    --bg:         #0d0f14;
    --surface:    #141720;
    --surface2:   #1c2030;
    --border:     #252a38;
    --text:       #c8cdd8;
    --muted:      #555e72;
    --accent:     #4cc9f0;
    --accent2:    #7b5ea7;
    --radius:     6px;
    --shadow:     0 8px 24px rgba(0,0,0,.5);
    --theme-icon: "☀️";
  }}

  /* ── Light theme ── */
  [data-theme="light"] {{
    --bg:         #f0f2f7;
    --surface:    #ffffff;
    --surface2:   #e8ecf4;
    --border:     #d0d5e2;
    --text:       #1a1d27;
    --muted:      #8890a4;
    --accent:     #0077cc;
    --accent2:    #6b3fa0;
    --radius:     6px;
    --shadow:     0 8px 24px rgba(0,0,0,.12);
    --theme-icon: "🌙";
  }}

  /* ── Transition for theme switch ── */
  body, .sidebar, header, #tooltip, .ctrl-btn, .col-select,
  .meta-item, .legend-item, .silhouette-bar-wrap {{
    transition: background-color .25s, border-color .25s, color .25s;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  header {{
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    background: var(--surface);
  }}
  header h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: .04em;
    flex: 1;
  }}
  header .meta-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
  }}

  /* ── Theme toggle button ── */
  #btnTheme {{
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius);
    padding: 4px 10px;
    font-size: 13px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Sans', sans-serif;
    transition: border-color .15s, background .15s;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  #btnTheme:hover {{ border-color: var(--accent); }}
  #btnTheme .theme-label {{
    font-size: 11px;
    color: var(--muted);
  }}

  .workspace {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  .sidebar {{
    width: 235px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    padding: 18px 14px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
    background: var(--surface);
  }}

  .sidebar-section h2 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 9px;
  }}

  .meta-item {{
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    color: var(--muted);
  }}
  .meta-item span:last-child {{
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ── Column selector ── */
  .col-select {{
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius);
    padding: 6px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    outline: none;
    transition: border-color .15s;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23555e72'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 8px center;
    padding-right: 24px;
  }}
  .col-select:focus {{ border-color: var(--accent); }}

  /* ── Silhouette ── */
  .silhouette-row {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 5px;
  }}
  .silhouette-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
  }}
  #silhouette-spinner {{
    font-size: 10px;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
  }}
  .silhouette-bar-wrap {{
    background: var(--surface2);
    border-radius: 99px;
    border: 1px solid var(--border);
    overflow: hidden;
    height: 6px;
    margin-bottom: 6px;
  }}
  .silhouette-bar {{
    height: 100%;
    width: 0%;
    transition: width .5s cubic-bezier(.4,0,.2,1), background-color .4s;
    border-radius: 99px;
  }}
  .silhouette-note {{
    font-size: 10px;
    color: var(--muted);
    line-height: 1.6;
  }}
  .sil-range {{
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--muted);
    margin-top: 2px;
  }}

  /* ── Legend ── */
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 6px;
    border-radius: var(--radius);
    cursor: pointer;
    transition: background .15s;
    font-size: 12px;
    user-select: none;
  }}
  .legend-item:hover {{ background: var(--surface2); }}
  .legend-item.dimmed {{ opacity: .2; }}
  .legend-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* ── Canvas ── */
  .canvas-wrap {{
    flex: 1;
    position: relative;
    overflow: hidden;
    background: var(--bg);
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 100%;
    cursor: crosshair;
  }}

  /* ── Tooltip ── */
  #tooltip {{
    position: fixed;
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 13px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    pointer-events: none;
    z-index: 999;
    min-width: 160px;
    box-shadow: var(--shadow);
    line-height: 1.9;
  }}
  #tooltip .tip-key  {{ color: var(--muted); font-size: 10px; letter-spacing:.06em; }}
  #tooltip .tip-val  {{ color: var(--text); }}
  #tooltip .tip-head {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }}
  #tooltip .swatch   {{ width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }}

  /* ── Controls ── */
  .controls {{
    position: absolute;
    bottom: 16px;
    right: 16px;
    display: flex;
    gap: 6px;
  }}
  .ctrl-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    width: 32px;
    height: 32px;
    border-radius: var(--radius);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color .15s, color .15s;
    box-shadow: var(--shadow);
  }}
  .ctrl-btn:hover {{ border-color: var(--accent); color: var(--accent); }}

  /* ── Scrollbar ── */
  .sidebar::-webkit-scrollbar {{ width: 4px; }}
  .sidebar::-webkit-scrollbar-track {{ background: transparent; }}
  .sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
</style>
</head>
<body>

<header>
  <h1>t-SNE &mdash; {patch_model}</h1>
  <span class="meta-tag">level: {level}</span>
  <span class="meta-tag">perp: {perplexity}</span>
  <span class="meta-tag">iter: {n_iter}</span>
  <button id="btnTheme">
    <span id="themeIcon">☀️</span>
    <span class="theme-label" id="themeLabel">Light mode</span>
  </button>
</header>

<div class="workspace">
  <aside class="sidebar">

    <div class="sidebar-section">
      <h2>Dataset</h2>
      <div class="meta-item"><span>Points</span><span id="nPoints">—</span></div>
      <div class="meta-item"><span>Groups</span><span id="nGroups">—</span></div>
    </div>

    <div class="sidebar-section">
      <h2>Color by</h2>
      <select id="colSelect" class="col-select"></select>
    </div>

    <div class="sidebar-section">
      <h2>Silhouette Score</h2>
      <div class="silhouette-row">
        <div class="silhouette-val" id="silVal">—</div>
        <span id="silhouette-spinner">computing…</span>
      </div>
      <div class="silhouette-bar-wrap">
        <div class="silhouette-bar" id="silBar"></div>
      </div>
      <div class="sil-range"><span>−1</span><span>0</span><span>+1</span></div>
      <div class="silhouette-note" style="margin-top:6px">
        Measures cluster separation on t-SNE coords.<br>
        Higher = better separation for this feature.
      </div>
    </div>

    <div class="sidebar-section">
      <h2>Legend</h2>
      <div id="legend"></div>
    </div>

    <div class="sidebar-section">
      <h2>Navigation</h2>
      <div style="font-size:11px;color:var(--muted);line-height:1.8">
        Scroll → zoom<br>
        Drag → pan<br>
        Click legend → toggle group
      </div>
    </div>

  </aside>

  <div class="canvas-wrap">
    <canvas id="cvs"></canvas>
    <div class="controls">
      <button class="ctrl-btn" id="btnZoomIn"  title="Zoom in">+</button>
      <button class="ctrl-btn" id="btnZoomOut" title="Zoom out">−</button>
      <button class="ctrl-btn" id="btnReset"   title="Reset view">⌂</button>
    </div>
  </div>
</div>

<div id="tooltip"></div>

<script>
const RAW         = {data_json};
const COLS        = {cols_json};
const PALETTE     = {palette_json};
const INITIAL_COL = {initial_col_json};

/* ══════════════════════════════════════════
   THEME TOGGLE
══════════════════════════════════════════ */
const html      = document.documentElement;
const btnTheme  = document.getElementById('btnTheme');
const themeIcon = document.getElementById('themeIcon');
const themeLbl  = document.getElementById('themeLabel');

function setTheme(t) {{
  html.setAttribute('data-theme', t);
  if (t === 'dark') {{
    themeIcon.textContent = '☀️';
    themeLbl.textContent  = 'Light mode';
  }} else {{
    themeIcon.textContent = '🌙';
    themeLbl.textContent  = 'Dark mode';
  }}
  draw();
}}

btnTheme.addEventListener('click', () => {{
  setTheme(html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}});

/* ══════════════════════════════════════════
   COLOR MAPPING
══════════════════════════════════════════ */
let colorMap   = {{}};
let active     = new Set();
let currentCol = INITIAL_COL;

function buildColorMap(col) {{
  const vals = [...new Set(RAW.map(p => p.tip[col] ?? 'N/A'))].sort(
    (a, b) => (a === 'N/A') - (b === 'N/A') || a.localeCompare(b)
  );
  const map = {{}};
  vals.forEach((v, i) => map[v] = PALETTE[i % PALETTE.length]);
  return map;
}}

function applyCol(col) {{
  currentCol = col;
  colorMap   = buildColorMap(col);
  active     = new Set(Object.keys(colorMap));
  buildLegend();
  draw();
  scheduleSilhouette(col);
}}

/* ══════════════════════════════════════════
   SILHOUETTE SCORE
   Computed on t-SNE 2D coords, sampled ≤2000 pts
══════════════════════════════════════════ */
let silTimer = null;

function scheduleSilhouette(col) {{
  document.getElementById('silhouette-spinner').style.display = 'inline';
  document.getElementById('silVal').textContent = '—';
  const bar = document.getElementById('silBar');
  bar.style.width = '0%';
  if (silTimer) clearTimeout(silTimer);
  silTimer = setTimeout(() => computeSilhouette(col), 60);
}}

function computeSilhouette(col) {{
  const MAX_N = 2000;
  let pts = RAW.map(p => ({{ x: p.x, y: p.y, label: p.tip[col] ?? 'N/A' }}));

  // Shuffle + sample if too large
  if (pts.length > MAX_N) {{
    for (let i = pts.length - 1; i > 0; i--) {{
      const j = Math.floor(Math.random() * (i + 1));
      [pts[i], pts[j]] = [pts[j], pts[i]];
    }}
    pts = pts.slice(0, MAX_N);
  }}

  const n      = pts.length;
  const labels = [...new Set(pts.map(p => p.label))];

  if (labels.length < 2) {{
    document.getElementById('silVal').textContent = 'N/A';
    document.getElementById('silhouette-spinner').style.display = 'none';
    return;
  }}

  // Pre-index by label
  const labelIdx = {{}};
  labels.forEach(l => labelIdx[l] = []);
  pts.forEach((p, i) => labelIdx[p.label].push(i));

  let total = 0;
  for (let i = 0; i < n; i++) {{
    const li = pts[i].label;
    if (labelIdx[li].length < 2) continue;

    // a(i): mean intra-cluster distance
    let sumA = 0;
    for (const j of labelIdx[li]) {{
      if (j === i) continue;
      sumA += Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
    }}
    const a = sumA / (labelIdx[li].length - 1);

    // b(i): min mean inter-cluster distance
    let b = Infinity;
    for (const l2 of labels) {{
      if (l2 === li) continue;
      let sumB = 0;
      for (const j of labelIdx[l2])
        sumB += Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
      const mb = sumB / labelIdx[l2].length;
      if (mb < b) b = mb;
    }}

    total += (b - a) / Math.max(a, b);
  }}

  const score = total / n;
  // Map [-1, 1] → [0%, 100%] for the bar
  const pct   = ((score + 1) / 2 * 100).toFixed(1);
  const color = score > 0.5 ? '#06d6a0' : score > 0.2 ? '#f4a261' : '#e63946';

  document.getElementById('silVal').textContent = score.toFixed(4);
  const bar = document.getElementById('silBar');
  bar.style.width           = pct + '%';
  bar.style.backgroundColor = color;
  document.getElementById('silhouette-spinner').style.display = 'none';
}}

/* ══════════════════════════════════════════
   LEGEND
══════════════════════════════════════════ */
function buildLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = '';
  document.getElementById('nGroups').textContent = Object.keys(colorMap).length;
  for (const [label, color] of Object.entries(colorMap)) {{
    const div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = `<span class="legend-dot" style="background:${{color}}"></span>${{label}}`;
    div.addEventListener('click', () => {{
      if (active.has(label)) {{ active.delete(label); div.classList.add('dimmed'); }}
      else                   {{ active.add(label);    div.classList.remove('dimmed'); }}
      draw();
    }});
    el.appendChild(div);
  }}
}}

/* ══════════════════════════════════════════
   COLUMN SELECTOR
══════════════════════════════════════════ */
const colSelect = document.getElementById('colSelect');
COLS.forEach(c => {{
  const opt = document.createElement('option');
  opt.value = c; opt.textContent = c;
  if (c === INITIAL_COL) opt.selected = true;
  colSelect.appendChild(opt);
}});
colSelect.addEventListener('change', () => applyCol(colSelect.value));

/* ══════════════════════════════════════════
   CANVAS + RENDER
══════════════════════════════════════════ */
const cvs = document.getElementById('cvs');
const ctx  = cvs.getContext('2d');
let W, H;
let transform  = {{ x: 0, y: 0, scale: 1 }};
let hoveredPt  = null;

let xMin, xMax, yMin, yMax, baseScale, basePanX, basePanY;

function initView() {{
  const r  = cvs.getBoundingClientRect();
  const cw = r.width, ch = r.height;
  xMin = Math.min(...RAW.map(p => p.x)); xMax = Math.max(...RAW.map(p => p.x));
  yMin = Math.min(...RAW.map(p => p.y)); yMax = Math.max(...RAW.map(p => p.y));
  const dw  = xMax - xMin || 1, dh = yMax - yMin || 1;
  const pad = 60;
  baseScale = Math.min((cw - pad*2) / dw, (ch - pad*2) / dh);
  basePanX  = (cw - dw * baseScale) / 2 - xMin * baseScale;
  basePanY  = (ch - dh * baseScale) / 2 - yMin * baseScale;
  transform = {{ x: 0, y: 0, scale: 1 }};
}}

function toScreen(px, py) {{
  const sx = px * baseScale + basePanX + transform.x;
  const sy = py * baseScale + basePanY + transform.y;
  const cx = W / devicePixelRatio / 2;
  const cy = H / devicePixelRatio / 2;
  return [
    sx * transform.scale + (1 - transform.scale) * cx,
    sy * transform.scale + (1 - transform.scale) * cy,
  ];
}}

const R = 4.5;

function draw() {{
  const r = cvs.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  for (const p of RAW) {{
    const label = p.tip[currentCol] ?? 'N/A';
    if (!active.has(label)) continue;
    const [sx, sy] = toScreen(p.x, p.y);
    const alpha    = (hoveredPt && hoveredPt !== p) ? 0.12 : 0.82;
    ctx.beginPath();
    ctx.arc(sx, sy, R, 0, Math.PI * 2);
    ctx.fillStyle = colorMap[label] + Math.round(alpha * 255).toString(16).padStart(2, '0');
    ctx.fill();
    if (hoveredPt === p) {{
      ctx.strokeStyle = html.getAttribute('data-theme') === 'dark'
        ? '#ffffff55' : '#00000055';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }}
  }}
}}

function resize() {{
  const r    = cvs.getBoundingClientRect();
  cvs.width  = r.width  * devicePixelRatio;
  cvs.height = r.height * devicePixelRatio;
  W = cvs.width; H = cvs.height;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  initView(); draw();
}}

/* ══════════════════════════════════════════
   TOOLTIP
══════════════════════════════════════════ */
const tip = document.getElementById('tooltip');

function nearest(mx, my) {{
  let best = null, bestD = 14 * 14;
  for (const p of RAW) {{
    const label = p.tip[currentCol] ?? 'N/A';
    if (!active.has(label)) continue;
    const [sx, sy] = toScreen(p.x, p.y);
    const d = (sx - mx) ** 2 + (sy - my) ** 2;
    if (d < bestD) {{ bestD = d; best = p; }}
  }}
  return best;
}}

cvs.addEventListener('mousemove', e => {{
  if (isDragging) {{ doPan(e); return; }}
  const rect = cvs.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const pt = nearest(mx, my);
  if (pt !== hoveredPt) {{ hoveredPt = pt; draw(); }}
  if (pt) {{
    const label = pt.tip[currentCol] ?? 'N/A';
    let html_str = `<div class="tip-head"><span class="swatch" style="background:${{colorMap[label]}}"></span><strong>${{label}}</strong></div>`;
    for (const [k, v] of Object.entries(pt.tip))
      html_str += `<span class="tip-key">${{k}}</span>: <span class="tip-val">${{v}}</span><br>`;
    tip.innerHTML = html_str;
    tip.style.display = 'block';
    tip.style.left    = (e.clientX + 14) + 'px';
    tip.style.top     = (e.clientY - 10) + 'px';
  }} else {{
    tip.style.display = 'none';
  }}
}});
cvs.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; hoveredPt = null; draw(); }});

/* ══════════════════════════════════════════
   PAN
══════════════════════════════════════════ */
let isDragging = false, dragX0, dragY0, panX0, panY0;
cvs.addEventListener('mousedown', e => {{
  isDragging = true;
  dragX0 = e.clientX; dragY0 = e.clientY;
  panX0  = transform.x; panY0 = transform.y;
  cvs.style.cursor = 'grabbing';
}});
function doPan(e) {{
  transform.x = panX0 + (e.clientX - dragX0) / transform.scale;
  transform.y = panY0 + (e.clientY - dragY0) / transform.scale;
  draw();
}}
window.addEventListener('mouseup', () => {{ isDragging = false; cvs.style.cursor = 'crosshair'; }});

/* ══════════════════════════════════════════
   ZOOM
══════════════════════════════════════════ */
cvs.addEventListener('wheel', e => {{
  e.preventDefault();
  transform.scale = Math.max(0.1, Math.min(40, transform.scale * (e.deltaY < 0 ? 1.1 : 0.91)));
  draw();
}}, {{ passive: false }});

document.getElementById('btnZoomIn').onclick  = () => {{ transform.scale = Math.min(40, transform.scale * 1.25); draw(); }};
document.getElementById('btnZoomOut').onclick = () => {{ transform.scale = Math.max(0.1, transform.scale / 1.25); draw(); }};
document.getElementById('btnReset').onclick   = () => {{ initView(); draw(); }};

/* ══════════════════════════════════════════
   INIT
══════════════════════════════════════════ */
document.getElementById('nPoints').textContent = RAW.length.toLocaleString();
window.addEventListener('resize', resize);
resize();
applyCol(INITIAL_COL);
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_dir = os.path.join(args.embeddings_folder, args.patch_model)
    if not os.path.isdir(model_dir):
        print(f"[ERROR] No existe la carpeta de embeddings: {model_dir}", flush=True)
        sys.exit(1)

    if args.output_html is None:
        out_dir = os.path.join("/workspace/tools/tsne_results", args.patch_model)
        os.makedirs(out_dir, exist_ok=True)
        suffix  = args.color_col if args.color_col else "all"
        args.output_html = os.path.join(
            out_dir, f"tsne_{suffix}_{args.level}.html"
        )

    df = create_file_label_df(
        images_path=args.images_folder,
        data_path=args.df_path,
    )
    print(f"WSIs en el DataFrame: {len(df)}", flush=True)

    if args.color_col and args.color_col not in df.columns:
        print(f"[ERROR] '{args.color_col}' no existe en el DataFrame. "
              f"Columnas disponibles: {list(df.columns)}", flush=True)
        sys.exit(1)

    embeddings, meta_df, tooltip_cols = load_embeddings(
        df=df,
        model_dir=model_dir,
        level=args.level,
        sample_per_wsi=args.sample_per_wsi,
        color_col=args.color_col,
        tooltip_cols=args.tooltip_cols,
    )
    print(f"Embeddings cargados: {embeddings.shape}", flush=True)

    if len(embeddings) == 0:
        print("[ERROR] No se cargó ningún embedding.", flush=True)
        sys.exit(1)

    tsne_coords = reduce(
        embeddings,
        pca_components=args.pca_components,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    html = build_html(
        tsne_coords=tsne_coords,
        meta_df=meta_df,
        color_col=args.color_col,
        tooltip_cols=tooltip_cols,
        patch_model=args.patch_model,
        level=args.level,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )

    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML guardado en: {args.output_html}", flush=True)


if __name__ == "__main__":
    main()