# WSI Breast Cancer Molecular Subtyping

Pipeline de Multiple Instance Learning (MIL) para la predicción de subtipos moleculares y
biomarcadores de cáncer de mama a partir de Whole Slide Images (WSI), usando el dataset BCNB.

Incluye extracción de embeddings con distintos patch encoders (UNI, GigaPath, DINO, KEEP...),
varios agregadores MIL (Attention MIL, TransMIL...), entrenamiento con validación cruzada,
evaluación final sobre test y herramientas de explicabilidad (heatmaps, t-SNE).

## Estructura del repositorio

```
config.py          # Rutas por defecto a datos/resultados (BCNB)
data_load.py        # Datasets y dataloaders (parches / embeddings precalculados)
train.py            # Entrenamiento con k-fold CV y búsqueda de configuraciones
test.py             # Selección de la mejor configuración + entrenamiento final + test
models/              # Patch encoders y agregadores MIL
utils/               # Utilidades de entrenamiento, métricas y explicabilidad
tools/               # Extracción de embeddings, heatmaps, t-SNE, inferencia batch
notebooks/           # Exploración de resultados
```

## Requisitos previos

- Docker con soporte para GPU ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
- Acceso a los datos de BCNB (parches/imágenes, `patient-clinical-data.xlsx` y, opcionalmente,
  embeddings precalculados).

## Uso con Docker

El contenedor solo trae instaladas las dependencias de Python (`requirements.txt`) sobre la
imagen base de PyTorch/CUDA. El código del repositorio y los datos **no** se copian a la imagen:
se montan como volúmenes al crear el contenedor, así que cualquier cambio en el código se refleja
sin necesidad de reconstruir la imagen.

### 1. Construir la imagen

```bash
docker build -t wsi-breast-subtyping:latest .
```

### 2. Crear el contenedor (rutas configurables)

Define las rutas de tu máquina como variables de entorno y créalo con `docker run`. Así puedes
reutilizar el mismo comando cambiando solo estas variables:

```bash
export WORKSPACE_DIR=/home/apasmel/nas_alejandro_pascual/WSI-Breast-Cancer-Molecular-Subtyping
export DATA_DIR=/mnt/ART-HDD-001
export CONTAINER_NAME=apasmel_breast_0
export GPU_DEVICE=0

docker run --name "$CONTAINER_NAME" \
  -it \
  --ipc=host \
  --gpus "device=$GPU_DEVICE" \
  -v "$WORKSPACE_DIR":/workspace \
  -v "$DATA_DIR":/data \
  -w /workspace \
  wsi-breast-subtyping:latest
```

- `WORKSPACE_DIR` → carpeta local de este repositorio, montada en `/workspace`.
- `DATA_DIR` → carpeta/disco con los datos de BCNB, montada en `/data` (debe coincidir con las
  rutas usadas en `config.py`, p. ej. `/data/BCNB/...`).
- `GPU_DEVICE` → índice de la GPU a usar (`--gpus 'device=0'`, `'device=0,1'`, `all`, ...).

Este comando crea el contenedor y abre una sesión interactiva (`bash`) dentro de él.

### 3. Parar / iniciar el contenedor

El contenedor creado con `docker run` persiste hasta que lo elimines. Para no tener que
recrearlo cada vez:

```bash
docker stop "$CONTAINER_NAME"    # Detener
docker start "$CONTAINER_NAME"   # Volver a arrancar (en segundo plano)
```

### 4. Conectarte a un contenedor ya iniciado (attach / exec)

Con el contenedor iniciado (`docker start`), abre una nueva sesión de terminal dentro de él:

```bash
docker exec -it "$CONTAINER_NAME" /bin/bash
```

Recomendado: usa `tmux` dentro del contenedor para no perder la sesión al desconectarte:

```bash
tmux new -s mi_sesion          # Crear sesión
# ... trabajo dentro de tmux ...
# Ctrl+B, D para salir sin cerrar la sesión

tmux attach -t mi_sesion       # Reconectar
tmux ls                        # Listar sesiones activas
tmux kill-session -t mi_sesion # Eliminar sesión
```

### 5. Eliminar el contenedor

```bash
docker rm -f "$CONTAINER_NAME"
```

## Configuración de rutas

Las rutas por defecto a datos y resultados están en [`config.py`](config.py) (pensadas para el
layout dentro del contenedor: `/data/...`, `/workspace/...`). Todos los scripts principales
permiten sobreescribirlas por línea de comandos (`--images_folder`, `--df_path`,
`--precomputed_embeddings_folder`, `--results_path`, ...), por lo que normalmente no hace falta
editar `config.py`.

## Entrenamiento

```bash
python train.py --run_all_configs --results_path /workspace/results/experiment
```

## Evaluación final (selección de mejor configuración + test)

```bash
python test.py --source_results_path /workspace/results/experiment
```

Selección manual de encoder x MIL por tarea:

```bash
python test.py \
  --source_results_path /workspace/results/experiment \
  --encoder_mil_override er:gigapath_abMIL-bag-level \
  --encoder_mil_override her2:gigapath_transMIL
```

## Preprocesamiento de WSIs (opcional)

El parcheo de los `.tif` y la extracción de embeddings se hace con
[CPathPipeline](https://github.com/PabloMeseguerEsbri/CPathPipeline), en un contenedor aparte:

```bash
git clone https://github.com/PabloMeseguerEsbri/CPathPipeline.git

docker run --name apasmel_CPathPipeline_0 \
  -it \
  --ipc=host \
  --gpus 'device=0' \
  -v /home/apasmel/nas_alejandro_pascual/CPathPipeline:/workspace \
  -v /mnt/ART-HDD-001:/data \
  -w /workspace \
  python:3.10-bullseye \
  bash
```

Dentro del contenedor:

```bash
export PYTHONPATH=/workspace:$PYTHONPATH
pip install -r /workspace/patching/requirements.txt

apt-get update
apt-get install -y libgl1 openslide-tools libopenslide0 libopenslide-dev
pip install torch torchvision

python patching/extract_patches.py \
  --source /data/BCNB/TIF/ \
  --save_dir /data/BCNB/patches/ \
  --seg --patch \
  --patch_size 1024 --step_size 1024 --custom_downsample 1
```