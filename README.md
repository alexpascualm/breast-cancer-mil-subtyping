# Procesamiento y parcheo de las Tifs

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

export PYTHONPATH=/workspace:$PYTHONPATH
pip install -r /workspace/patching/requirements.txt

apt-get update
apt-get install -y libgl1
apt-get install -y openslide-tools libopenslide0
apt-get install -y libopenslide-dev
pip install torch
pip install torchvision

python patching/extract_patches.py --source /data/BCNB/TIF/ --save_dir /data/BCNB/patches/ --seg --patch --patch_size 1024 --step_size 1024 --custom_downsample 1

# Extracción de embeddings con Encoders

python extract_embeddings.py --patch_model uni  --output_folder /data/BCNB/processed_1024/embeddings

# Crear el contenedor para los experimentos de mama

docker run --name apasmel_breast_0 -it --ipc=host --gpus 'device=0' -v /home/apasmel/nas_alejandro_pascual/WSI-Breast-Cancer-Molecular-Subtyping:/workspace -v /mnt/ART-HDD-001:/data  nvcr.io/nvidia/pytorch:25.03-py3

# Montar y desmontar disco duro

sudo mount -t ntfs-3g /dev/sdb1 /mnt/ART-HDD-001 -o uid=1001,gid=1001,umask=027,exec

sudo umount /mnt/ART-HDD-001