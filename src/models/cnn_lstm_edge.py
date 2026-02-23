import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMEdge(nn.Module):
    """
    Edge:
    - CNN (spatial per time step)
    - Small LSTM (short-term temporal)
    - Local exit
    """

    def __init__(self, feature_dim=64, hidden_dim=64):
        super().__init__()

        # Spatial CNN (applied per timestep)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.feature_dim = 32

        # Small LSTM (edge temporal model)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Local exit
        self.fc_local = nn.Linear(hidden_dim, 49)

    def forward(self, x):
        """
        x: (B, 1, N, 7, 7)
        """

        B, C, N, H, W = x.shape

        # reshape for CNN per timestep
        x = x.view(B * N, 1, H, W)
        features = self.cnn(x)  # (B*N, 32,1,1)
        features = features.view(B, N, self.feature_dim)

        # LSTM
        out, _ = self.lstm(features)
        h_edge = out[:, -1, :]  # last timestep

        # local prediction
        y_L = self.fc_local(h_edge)

        return h_edge, y_L