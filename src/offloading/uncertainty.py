import torch

def uncertainty_score(y_samples):
    return torch.var(y_samples, dim=0).max().item()
