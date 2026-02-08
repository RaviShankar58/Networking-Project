import torch.nn as nn
import torch.nn.functional as F

class CloudModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv3d(32, 16, kernel_size=5)
        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(16 * 0 * 1 * 1 + 16, 128)  # flatten safe
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 49)

    def forward(self, z):
        x = F.relu(self.conv(z))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)
