import torch

def sla_cost(y, d, cu=10.0, co=1.0):
    under = torch.clamp(d - y, min=0)
    over = torch.clamp(y - d, min=0)
    cost = cu * (under > 0).float() + co * over
    return cost.mean()
