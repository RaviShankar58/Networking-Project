import torch
from torch.utils.data import DataLoader, TensorDataset
from offloading.benefit import compute_b_star
from offloading.optimized import OptimizedOffloader

def train_offloader(z, CL, CR, L0):
    b_star = compute_b_star(CL, CR, L0)
    labels = ((CL - CR) >= b_star).float()

    loader = DataLoader(TensorDataset(z, labels), batch_size=32, shuffle=True)

    model = OptimizedOffloader().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(50):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            p = model(x).squeeze()
            loss = loss_fn(p, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
