import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # F0
        self.conv = nn.Conv3d(1, 32, kernel_size=3)
        self.dropout = nn.Dropout(p=0.2)

        # F1
        self.fc = nn.Linear(32 * 4 * 5 * 5, 49)  

    def forward(self, x):
        z = F.relu(self.conv(x))
        z = self.dropout(z)

        flat = z.view(z.size(0), -1)
        yL = self.fc(flat)

        return z, yL
