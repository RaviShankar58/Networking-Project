import torch

def sla_cost(y, d, cu=10.0, co=1.0):
    Iu = (y < d).float()          
    Io = 1.0 - Iu                

    cost = cu * Iu + co * Io * (y - d).clamp(min=0)

    return cost.sum(dim=1).mean()
