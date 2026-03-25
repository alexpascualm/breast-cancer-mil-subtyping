import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from models.MIL_model import MILArchitecture


class PatchClassifierVIT(nn.Module):
    """
    Encoder ViT + cabeza de clasificación + MIL aggregation.

    Para tareas binarias (er, pr, her2, ki67)  → num_classes=2
    Para subtipo molecular                      → num_classes=4

    Args:
        num_classes (int): número de clases de salida.
        unfreeze_backbone (bool): descongelar los últimos 3 bloques del ViT.
        model_name (str): nombre del modelo en timm.
        mil_aggregation_mode (str): 'max' | 'mean' | 'attentionMIL'.
        chunk_size (int): tamaño de los mini-batches internos de parches (control de VRAM).
    """

    def __init__(
        self,
        num_classes: int = 2,
        unfreeze_backbone: bool = False,
        model_name: str = "vit_large_patch14_dinov2",
        mil_aggregation_mode: str = "attentionMIL",
        chunk_size: int = 128,
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
                            model_name,
                            pretrained=True,
                            img_size=448
                        )
        self.backbone.reset_classifier(0)           # quitar cabeza de clasificación
        embed_dim = self.backbone.num_features

        # Congelar todo por defecto
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_backbone:
            for param in self.backbone.blocks[9:].parameters():   # últimos 3 bloques
                param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            self.backbone.set_grad_checkpointing(enable=True)

        # ── Clasificador de parche ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # ── MIL aggregation ───────────────────────────────────────────────────
        self.mil = MILArchitecture(
            classes=list(range(num_classes)),
            aggregation=mil_aggregation_mode,
            L_value=embed_dim,
        )

    def forward(self, inputs_list):
        """
        Args:
            inputs_list: list of Tensors, cada uno [N_patches, C, H, W] para una WSI.

        Returns:
            wsi_logits         : Tensor (B, num_classes)
            patch_probs_list   : list of B Tensors [N_patches, num_classes]
            weights_list       : list of B Tensors [N_patches] o None (según agregación)
        """
        device = next(self.parameters()).device

        lengths = [x.size(0) for x in inputs_list]
        if sum(lengths) == 0:
            raise RuntimeError("El batch no contiene parches.")

        # Concatenar en CPU y dividir en chunks para controlar VRAM
        all_patches_cpu = torch.cat(inputs_list, dim=0)
        chunks = torch.split(all_patches_cpu, self.chunk_size, dim=0)

        all_cls_tokens, all_logits = [], []
        for chunk in chunks:
            chunk_gpu = chunk.to(device, non_blocking=True)
            feats     = self.backbone.forward_features(chunk_gpu)  # (chunk, N+1, D)
            
            cls       = feats[:, 0, :]             # CLS (chunk, D)
            # cls = feats[:, 1:, :].mean(dim=1)    # mean pooling sin CLS (chunk, D)
            # cls = feats.mean(dim=1)              # mean pooling incluyendo CLS (chunk, D)
            
            logits    = self.classifier(cls)                       # (chunk, num_classes)
            
            all_cls_tokens.append(cls)
            all_logits.append(logits)

        total_feats = torch.cat(all_cls_tokens, dim=0)  # (N_total, D)
        total_logtis = torch.cat(all_logits,      dim=0)  # (N_total, num_classes)

        # MIL aggregation
        wsi_logits, weights_list = self.mil(total_logtis, total_feats, lengths)

        patch_logits_list = list(torch.split(total_logtis, lengths, dim=0))

        return wsi_logits, patch_logits_list, weights_list
