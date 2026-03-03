
# Crear el contenedor desde

docker run --name apasmel_breast_0 -it --ipc=host --gpus 'device=0' -v /home/apasmel/nas_alejandro_pascual/WSI-Breast-Cancer-Molecular-Subtyping:/workspace -v /mnt/ART-HDD-001:/data  nvcr.io/nvidia/pytorch:24.02-py3

# Montar y desmontar disco duro

sudo mount -t ntfs-3g /dev/sda1 /mnt/ART-HDD-001 -o uid=1001,gid=1001,umask=027,exec

sudo umount /mnt/ART-HDD-001