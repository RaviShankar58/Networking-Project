import torch.nn as nn
import torch.nn.functional as F

class CloudModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(32, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 49)

    def forward(self, z):
        x = F.relu(self.conv(z))
        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        yR = self.fc2(x)
        return yR
