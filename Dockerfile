# Imagen base con PyTorch + CUDA ya instalados
FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /workspace

# Solo copiamos requirements.txt para instalar dependencias: el código del
# proyecto y los datos se montan como volúmenes en tiempo de ejecución
# (ver README.md), así que no hace falta reconstruir la imagen al editar código.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]