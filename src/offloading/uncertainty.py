import torch

def uncertainty_score(model, x, B=10):
    model.train() 

    samples = []
    with torch.no_grad():
        for _ in range(B):
            _, yL = model.edge(x)
            samples.append(yL.unsqueeze(0))

    YL = torch.cat(samples, dim=0)   # [B, batch, 49]
    var = torch.var(YL, dim=0)       # [batch, 49]

    return var.max(dim=1).values     # per-sample U
