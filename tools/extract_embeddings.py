# extract_embeddings.py
import argparse
import os
import sys
import torch
import h5py
import numpy as np
from tqdm import tqdm
from huggingface_hub import login

from torchvision import transforms as T
from PIL import Image

from utils.train_utils import ENCODER_CONFIGS
from models.patch_encoders import ENCODER_REGISTRY
from data_load import create_file_label_df
from config import bcnb_images_folder_path, bcnb_labels_df_path, bcnb_embeddings_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_model",     type=str, default="dino",
                        choices=["dino", "uni", "uni2", "conch", "keep"])
    parser.add_argument("--images_folder",   type=str, default=bcnb_images_folder_path)
    parser.add_argument("--df_path",         type=str, default=bcnb_labels_df_path)
    parser.add_argument("--output_folder",   type=str, default=bcnb_embeddings_path,
                        help="Carpeta donde guardar los .h5 de embeddings")
    parser.add_argument("--chunk_size",      type=int, default=512)
    parser.add_argument("--patch_resize_size", type=int, default=224)
    parser.add_argument("--hf_token",        type=str, default=None)
    parser.add_argument("--overwrite",       action="store_true",
                        help="Reextrae aunque el .h5 ya exista")
    return parser.parse_args()


def extract_embeddings(args):
    if args.hf_token:
        login(token=args.hf_token)

    # Añadir el modelo al output_folder
    model_output_dir = os.path.join(args.output_folder, args.patch_model)
    os.makedirs(model_output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}", flush=True)

    # Construir solo el encoder (no el clasificador MIL)
    key = args.patch_model.lower()
    EncoderClass   = ENCODER_REGISTRY[key]
    encoder_kwargs = {k: v for k, v in ENCODER_CONFIGS[key].items() if k != "embed_dim"}
    encoder = EncoderClass(
        chunk_size=args.chunk_size,
        unfreeze_backbone=False,   # siempre congelado en extracción
        **encoder_kwargs,
    ).to(device)
    encoder.eval()
    print(f"Encoder '{args.patch_model}' cargado. embed_dim={encoder.embed_dim}", flush=True)

    # DataFrame con todas las WSIs
    df = create_file_label_df(
        images_path=args.images_folder,
        data_path=args.df_path,
    )
    print(f"WSIs encontradas: {len(df)}", flush=True)

    transform = T.Compose([
        T.Resize((args.patch_resize_size, args.patch_resize_size),
                 interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    skipped, processed, failed = 0, 0, 0
    patch_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Extrayendo embeddings", file=sys.stdout):
        h5_src = row["features"]
        wsi_id = os.path.splitext(os.path.basename(h5_src))[0]
        out_path = os.path.join(model_output_dir, f"{wsi_id}.h5")

        if os.path.exists(out_path) and not args.overwrite:
            try:
                with h5py.File(out_path, "r") as h5_out:
                    if "n_patches" in h5_out.attrs:
                        patch_counts.append(h5_out.attrs["n_patches"])
                    else:
                        patch_counts.append(h5_out["embeddings"].shape[0])
            except Exception as e:
                print(f"\n[ERROR leyendo skip] {wsi_id}: {e}", flush=True)
            skipped += 1
            continue

        try:
            with h5py.File(h5_src, "r", swmr=True) as h5:
                imgs   = h5["imgs"][:]
                coords = h5["coords"][:] if "coords" in h5 else None

            # Transformar y extraer
            patches = torch.stack([transform(Image.fromarray(img)) for img in imgs])
            with torch.no_grad():
                embeddings = encoder.forward(patches) # (N, embed_dim)

            n_patches = len(embeddings)
            patch_counts.append(n_patches)

            # Guardar en nuevo h5
            with h5py.File(out_path, "w") as h5_out:
                h5_out.create_dataset("embeddings", data=embeddings.cpu().numpy(),
                                      compression="gzip", compression_opts=4)
                if coords is not None:
                    h5_out.create_dataset("coords", data=coords,
                                          compression="gzip", compression_opts=4)
                # Guardar metadatos útiles
                h5_out.attrs["encoder"]    = args.patch_model
                h5_out.attrs["embed_dim"]  = encoder.embed_dim
                h5_out.attrs["n_patches"]  = n_patches

            processed += 1

        except Exception as e:
            print(f"\n[ERROR] {wsi_id}: {e}", flush=True)
            failed += 1

    # --- Estadísticas finales ---
    if len(patch_counts) > 0:
        patch_counts = np.array(patch_counts)
        mean_val = np.mean(patch_counts)
        q1, q2, q3 = np.percentile(patch_counts, [25, 50, 75])
        min_val = np.min(patch_counts)
        max_val = np.max(patch_counts)

        print("\nEstadísticas de número de parches por WSI:", flush=True)
        print(f"Media: {mean_val:.2f}", flush=True)
        print(f"Q1 (25%): {q1:.2f}", flush=True)
        print(f"Mediana (50%): {q2:.2f}", flush=True)
        print(f"Q3 (75%): {q3:.2f}", flush=True)
        print(f"Mínimo: {min_val}", flush=True)
        print(f"Máximo: {max_val}", flush=True)

    print(f"\nExtracción completa: {processed} procesadas, "
          f"{skipped} saltadas, {failed} fallidas.", flush=True)


if __name__ == "__main__":
    args = parse_args()
    extract_embeddings(args)