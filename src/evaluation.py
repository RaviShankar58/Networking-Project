import torch

def evaluate_ddnn(model, offloader, loader, cu=10.0, co=1.0):
    model.eval()
    offloader.eval()

    total_cost = 0
    local_count = 0
    total = 0

    with torch.no_grad():
        for x, d in loader:
            z, yL = model.edge(x)
            yR = model.cloud(z)

            p = offloader(z).squeeze()
            use_local = (p < 0.5)

            y = torch.where(use_local.unsqueeze(1), yL, yR)

            Iu = (y < d).float()
            Io = 1 - Iu
            cost = cu * Iu + co * Io * (y - d).clamp(min=0)

            total_cost += cost.sum().item()
            local_count += use_local.sum().item()
            total += d.size(0)

    C = total_cost / total
    L = local_count / total

    return C, L
