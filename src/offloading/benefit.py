import torch

def compute_b_star(CL, CR, L0):
    b = CL - CR
    b_sorted, _ = torch.sort(b)
    idx = int(L0 * len(b_sorted))
    return b_sorted[idx]

def oracle_offload(CL, CR, L0):
    b = CL - CR
    b_sorted, _ = torch.sort(b)
    b_star = b_sorted[int(L0 * len(b))]

    return (b < b_star).float()  # 1 = local
