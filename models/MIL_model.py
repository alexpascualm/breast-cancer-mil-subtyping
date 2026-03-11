import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, L=512):
        super().__init__()
        D = L // 2
        self.branch_1 = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.branch_2 = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.branch_3 = nn.Linear(D, 1)

    def forward(self, x):
        # x: (N, L)
        xa = self.branch_1(x)
        xb = self.branch_2(x)
        A = xa * xb
        A = self.branch_3(A)    # (N, 1)
        return A.squeeze(-1)    # (N,)
    
class attentionMIL_instance(nn.Module):
    def __init__(self, L_value=512):
        super().__init__()
        self.L = L_value
        self.weight = Attention(L=self.L)

    def forward(self, logits_ws, features_ws):
        """
        logits_ws: (N, num_classes) probabilities/logits per parche de esta WSI
        features_ws: (N, L) embeddings (cls tokens) per parche de esta WSI
        """
        if features_ws.numel() == 0:
            # si no hay parches (caso borde) devolvemos ceros
            return torch.zeros(logits_ws.size(1), device=logits_ws.device), torch.zeros(0, device=logits_ws.device)
        
        weights = torch.softmax(self.weight(features_ws), dim=0)      # (N,)
        weighted = logits_ws * weights.unsqueeze(1)  # (N, num_classes)
        w_logits = weighted.sum(dim=0)            # (num_classes,)
        return w_logits, weights

class MILArchitecture(nn.Module):
    def __init__(self, classes, aggregation='max', L_value=None):
        super().__init__()
        self.aggregation = aggregation
        self.L = L_value
        self.nClasses = len(classes)

        if self.aggregation == 'attentionMIL':
            assert self.L is not None, "L_value required for attentionMIL"
            self.attentionModule = attentionMIL_instance(self.L)

    def forward(self, patch_probs, patch_feats, lengths):
        """
        patch_probs: Tensor (N_total, C)  (probabilidades por parche)
        patch_feats: Tensor (N_total, L)  (embeddings cls per parche)
        lengths: list[int] (n_parches por WSI), sum(lengths) == N_total
        Devuelve: Tensor (batch_size, C) con logits/score por WSI
        """
        assert patch_probs.dim() == 2
        assert patch_feats.dim() == 2

        # Dividir por WSI
        probs_splits = torch.split(patch_probs, lengths, dim=0)
        feats_splits = torch.split(patch_feats, lengths, dim=0)

        wsis, weights = [],[]
        if self.aggregation == 'max':
            for p in probs_splits:
                if p.size(0) == 0:
                    wsis.append(torch.zeros(self.nClasses, device=patch_probs.device))
                else:
                    wsis.append(torch.max(p, dim=0)[0])
                weights.append(None)
        elif self.aggregation == 'mean':
            for p in probs_splits:
                if p.size(0) == 0:
                    wsis.append(torch.zeros(self.nClasses, device=patch_probs.device))
                else:
                    wsis.append(torch.mean(p, dim=0))
                weights.append(None)
        elif self.aggregation == 'attentionMIL':
            for p, f in zip(probs_splits, feats_splits):
                wsi, weight = self.attentionModule(p, f)
                wsis.append(wsi)
                weights.append(weight)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        wsi_logits = torch.stack(wsis, dim=0)  # (batch, nClasses)
        
        return wsi_logits, weights