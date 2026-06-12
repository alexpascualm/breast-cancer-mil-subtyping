import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import h5py
from tqdm import tqdm
import gc
from PIL import Image
import sys

from utils.data_utils import map_molecular_subtype, map_ki67, MOLECULAR_SUBTYPE_MAP, BIOMARKER_BINARY_MAP, KI67_THRESHOLD

# ──────────────────────────────────────────────────────────────────────────────
# DATAFRAME BUILDER
# ──────────────────────────────────────────────────────────────────────────────
def create_file_label_df(
    images_path: str,
    data_path: str,
    slide_col: str = 'Patient ID',
    small_dataset_for_test: bool = False,
    ki67_threshold: float = KI67_THRESHOLD,
    ) -> pd.DataFrame:
    """
    Builds a DataFrame linking each H5 patch file to its labels.

    Parameters
    ----------
    images_path : str
        Directory containing the .h5 patch files.
    data_path : str
        Path to the Excel (.xlsx) file with clinical/pathological metadata.
    slide_col : str
        Column in the Excel whose values match the .h5 file stems (without extension).
    small_dataset_for_test : bool
        If True, only the first rows of the Excel are used.
    ki67_threshold : float
        Cut-off to binarise Ki67 (default KI67_THRESHOLD = 0.14).

    Returns
    -------
    pd.DataFrame with columns:
        features, patient_id,
        molecular_subtype, molecular_subtype_label,
        er, pr, her2, ki67      ← binary labels (0/1, -1 if missing/unparseable)
        ki67_raw                ← original value from Excel
        Extra info
    """
    df = pd.read_excel(data_path)
    df.columns = [c.strip() for c in df.columns]

    records = []
    wsis = [f for f in os.listdir(images_path) if f.endswith('.h5')]
    if small_dataset_for_test:
        wsis = wsis[:50]

    for f in tqdm(wsis, desc="Processing breast WSIs", file=sys.stdout):
        slide_id = f.replace('.h5', '')
        row = df[df[slide_col] == int(slide_id)]
        if row.empty:
            continue

        row = row.iloc[0]

        try:
            molecular_subtype_label = map_molecular_subtype(row.get('Molecular subtype', -1))
        except ValueError as e:
            print(f"  [WARN] {slide_id}: {e} - skipping.")
            continue

        ki67_raw = row.get('Ki67', np.nan)

        record = {
            'features':                os.path.join(images_path, f),
            'patient_id':              row.get('Patient ID', slide_id),
            # ── Molecular subtype ──────────────────────────────────────────
            'molecular_subtype':     molecular_subtype_label,        
            'molecular_subtype_str': str(row.get('Molecular subtype', '')).strip(),  
            # ── Biomarkers: one binary label per marker ───────────────────
            'er':   BIOMARKER_BINARY_MAP.get(str(row.get('ER',   '')).strip(), -1),
            'pr':   BIOMARKER_BINARY_MAP.get(str(row.get('PR',   '')).strip(), -1),
            'her2': BIOMARKER_BINARY_MAP.get(str(row.get('HER2', '')).strip(), -1),
            'ki67': map_ki67(ki67_raw, threshold=ki67_threshold),
            # ── Extra info (reference / stratification) ───────────────────
            'ki67_raw':        ki67_raw,
            'her2_expression': row.get('HER2 Expression', np.nan),
            'tumour_type':     str(row.get('Tumour Type', '')).strip(),
            'tumour_size':     row.get('Tumour Size(cm)', np.nan),
            'age':             row.get('Age(years)', np.nan),
            'histological_grade':   row.get('Histological grading', np.nan),
            'n_lymph_nodes':   row.get('Number of lymph node metastases', np.nan),
        }
        records.append(record)

    return pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────
class Patches_WSI_Dataset(Dataset):
    """
    Dataset for WSI patches stored in HDF5 files.

    Supported tasks
    ---------------
    'molecular_subtype'
        label → LongTensor scalar (0-3, see MOLECULAR_SUBTYPE_MAP)

    'er' | 'pr' | 'her2' | 'ki67'
        label → LongTensor scalar (0 or 1)
        One dataset / model per biomarker.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Output of create_file_label_df().
    indices : list[int]
        Row indices into dataframe to use.
    task : str
        'molecular_subtype', 'er', 'pr', 'her2', or 'ki67'.
    patch_resize_size: int
        WSI-patchs resize cuadratic size for model inputs
    load_on_demand : bool
        False → pre-load all patch tensors into CPU RAM.
        True  → read H5 in __getitem__ (lower RAM, slower I/O).
    """
    SUPPORTED_TASKS = ('molecular_subtype', 'er', 'pr', 'her2', 'ki67')

    def __init__(
        self,
        dataframe: pd.DataFrame,
        indices,
        task: str = 'molecular_subtype',
        patch_resize_size: int = 224,
        load_on_demand: bool = False,
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {self.SUPPORTED_TASKS}, got '{task}'")

        gc.collect()
        self.dataframe         = dataframe
        self.indices           = list(indices)
        self.task              = task
        self.patch_resize_size = patch_resize_size
        self.load_on_demand    = bool(load_on_demand)

        self.transform = T.Compose([
            T.Resize((patch_resize_size, patch_resize_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.images, self.labels, self.wsi_ids, self.coords = [], [], [], []

        # ── Pre-load labels ───────────────────────────────────────────────────
        for i in self.indices:
            row = self.dataframe.iloc[i]
            self.labels.append(self._build_label(row))

        # ── Pre-load images (optional) ────────────────────────────────────────
        if not self.load_on_demand:
            for i in tqdm(self.indices, desc="Loading breast WSI patches", file=sys.stdout):
                row = self.dataframe.iloc[i]
                fpath = row['features']
                with h5py.File(fpath, "r") as h5:
                    imgs   = h5["imgs"][:]
                    coords = h5["coords"][:] if "coords" in h5 else None

                imgs_tensor = torch.stack([self.transform(Image.fromarray(img)) for img in imgs])
                self.images.append(imgs_tensor)
                self.wsi_ids.append(fpath)
                self.coords.append(coords)
        else:
            self.feature_paths = [self.dataframe.iloc[i]['features'] for i in self.indices]
            self.coords        = [None] * len(self.indices)
            self.wsi_ids     = list(self.feature_paths)

    # ── Label builder ─────────────────────────────────────────────────────────
    def _build_label(self, row) -> torch.Tensor:
        return torch.tensor(int(row[self.task]), dtype=torch.long)

    # ── PyTorch interface ─────────────────────────────────────────────────────
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if not self.load_on_demand:
            imgs_tensor = self.images[idx]
            wsi_id    = self.wsi_ids[idx]
            coords      = self.coords[idx]
        else:
            fpath    = self.feature_paths[idx]
            wsi_id = fpath
            coords   = None
            with h5py.File(fpath, "r") as h5:
                imgs = h5["imgs"][:]
                coords = h5["coords"][:] if "coords" in h5 else None
            imgs_tensor = torch.stack([self.transform(Image.fromarray(img)) for img in imgs])

        return imgs_tensor, label, wsi_id, coords
        

    # ── Convenience ───────────────────────────────────────────────────────────
    @property
    def num_classes(self) -> int:
        """Output units needed: 4 for molecular subtype, 2 for any binary biomarker."""
        if self.task == 'molecular_subtype':
            return len(MOLECULAR_SUBTYPE_MAP)   # 4
        return 2                                # binary: 0 / 1


class Embeddings_WSI_Dataset(Dataset):
    """
    Dataset que lee embeddings precomputados desde ficheros .h5,
    evitando usar el encoder en cada época.

    Estructura esperada del .h5:
        embeddings : (N, embed_dim)  float32
        coords     : (N, 2)          int      [opcional]

    Parameters
    ----------
    dataframe : pd.DataFrame
        Output de create_file_label_df(), con columna 'features'
        apuntando a los .h5 ORIGINALES de parches.
    indices : list[int]
    embeddings_folder : str
        Carpeta donde están los .h5 de embeddings precomputados.
        Se asume que el stem del fichero coincide con el de 'features'.
    task : str
    load_on_demand : bool
        False → carga todos los embeddings en RAM al iniciar.
        True  → lee el .h5 en cada __getitem__.
        Con embeddings (floats) el tamaño en RAM es mucho menor
        que con imágenes, así que load_on_demand=False es viable
        incluso para datasets grandes.
    """

    SUPPORTED_TASKS = ('molecular_subtype', 'er', 'pr', 'her2', 'ki67')

    def __init__(
        self,
        dataframe: pd.DataFrame,
        indices,
        embeddings_folder: str,
        task: str = 'molecular_subtype',
        load_on_demand: bool = False,
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"task must be one of {self.SUPPORTED_TASKS}")

        self.dataframe         = dataframe
        self.indices           = list(indices)
        self.embeddings_folder = embeddings_folder
        self.task              = task
        self.load_on_demand    = load_on_demand

        # Construir rutas a los .h5 de embeddings
        self.emb_paths = []
        for i in self.indices:
            src = dataframe.iloc[i]["features"]
            stem = os.path.splitext(os.path.basename(src))[0]
            self.emb_paths.append(
                os.path.join(embeddings_folder, f"{stem}.h5")
            )

        # Validar que existen antes de empezar
        missing = [p for p in self.emb_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} ficheros de embeddings no encontrados. "
                f"Primeros: {missing[:3]}"
            )

        # Labels
        self.labels = [
            torch.tensor(int(dataframe.iloc[i][task]), dtype=torch.long)
            for i in self.indices
        ]

        # Pre-carga opcional
        self.embeddings_cache = []
        self.coords_cache     = []

        if not self.load_on_demand:
            print(f"Pre-cargando embeddings en RAM...", flush=True)
            for path in tqdm(self.emb_paths, file=sys.stdout):
                emb, coords = self._read_h5(path)
                self.embeddings_cache.append(emb)
                self.coords_cache.append(coords)

    def _read_h5(self, path):
        with h5py.File(path, "r") as h5:
            emb    = torch.from_numpy(h5["embeddings"][:])   # (N, D)
            coords = h5["coords"][:] if "coords" in h5 else None
        return emb, coords

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label = self.labels[idx]
        wsi_id = self.emb_paths[idx]

        if not self.load_on_demand:
            emb    = self.embeddings_cache[idx]
            coords = self.coords_cache[idx]
        else:
            emb, coords = self._read_h5(self.emb_paths[idx])

        
        return emb, label, wsi_id, coords
       

    @property
    def num_classes(self) -> int:
        if self.task == 'molecular_subtype':
            return len(MOLECULAR_SUBTYPE_MAP)
        return 2
        
# ──────────────────────────────────────────────────────────────────────────────
# COLLATE
# ──────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    """
    Handles batches of imgs/embeddings, label, wsi_id, coords).
    Returns:
        list_features  - list of per-WSI tensors
        labels         - stacked LongTensor  [B] or [B, ...]
        wsi_ids      - list of WSI identifiers 
        coords         - list coords arrays
    """
    if len(batch[0]) == 4:
        features, labels, wsi_ids, coords = zip(*batch)
        return list(features), torch.stack(labels, dim=0), list(wsi_ids), list(coords)

    else:
        raise ValueError(f"Unexpected batch structure with {len(batch[0])} elements.")