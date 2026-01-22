import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(32, 49)

    def forward(self, x):
        z = F.relu(self.conv(x))
        pooled = self.pool(z).view(z.size(0), -1)
        yL = self.fc(pooled)
        return z, yL
