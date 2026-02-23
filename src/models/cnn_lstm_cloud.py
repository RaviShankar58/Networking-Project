import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMCloud(nn.Module):
    """
    Cloud:
    - Deeper LSTM refinement
    - Remote exit
    """

    def __init__(self, hidden_dim=64, cloud_hidden=128):
        super().__init__()

        self.lstm_cloud = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=cloud_hidden,
            batch_first=True
        )

        self.fc1 = nn.Linear(cloud_hidden, 64)
        self.fc2 = nn.Linear(64, 49)

    def forward(self, h_edge):
        """
        h_edge: (B, hidden_dim)
        """

        # treat as sequence length 1
        h_edge = h_edge.unsqueeze(1)

        out, _ = self.lstm_cloud(h_edge)
        h_cloud = out[:, -1, :]

        x = F.relu(self.fc1(h_cloud))
        y_R = self.fc2(x)

        return y_R