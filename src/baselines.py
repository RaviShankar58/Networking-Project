import torch

def edge_only(model, loader):
    cost, total = 0, 0
    with torch.no_grad():
        for x, d in loader:
            _, yL = model.edge(x)
            cost += ((yL - d) ** 2).sum().item()
            total += d.size(0)
    return cost / total


def cloud_only(model, loader):
    cost, total = 0, 0
    with torch.no_grad():
        for x, d in loader:
            z, _ = model.edge(x)
            yR = model.cloud(z)
            cost += ((yR - d) ** 2).sum().item()
            total += d.size(0)
    return cost / total


def random_offload(model, loader, L0=0.5):
    cost, total = 0, 0
    with torch.no_grad():
        for x, d in loader:
            z, yL = model.edge(x)
            yR = model.cloud(z)

            mask = torch.rand(d.size(0)) < L0
            y = torch.where(mask.unsqueeze(1), yL, yR)

            cost += ((y - d) ** 2).sum().item()
            total += d.size(0)
    return cost / total
