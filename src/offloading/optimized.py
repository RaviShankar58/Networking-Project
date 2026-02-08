import torch.nn as nn
import torch.nn.functional as F

class OptimizedOffloader(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv3d(32, 32, kernel_size=3)
        self.fc = nn.Linear(32 * 2 * 3 * 3, 1)

    def forward(self, z):
        x = F.relu(self.conv(z))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))
