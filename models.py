import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Spatial CNN
        self.conv2d = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # Spatial attention
        self.spatial_attn = nn.Conv2d(16, 1, kernel_size=1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=16*7*7,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # Temporal attention
        self.temporal_attn = nn.Linear(128, 1)

    def forward(self, x):
        # x: (N, 1, T, 7, 7)
        N, C, T, H, W = x.shape

        spatial_seq = []

        for t in range(T):
            xt = x[:, :, t, :, :]  # (N,1,7,7)

            # CNN
            ft = F.relu(self.conv2d(xt))  # (N,16,7,7)

            # Spatial attention map
            attn_map = torch.sigmoid(self.spatial_attn(ft))  # (N,1,7,7)

            # Apply attention
            ft = ft * attn_map  # (N,16,7,7)

            # Flatten
            ft = ft.view(N, -1)

            spatial_seq.append(ft)

        spatial_seq = torch.stack(spatial_seq, dim=1)
        # (N, T, 16*7*7)

        # LSTM
        lstm_out, _ = self.lstm(spatial_seq)
        # (N, T, 128)

        # Temporal attention
        attn_weights = torch.softmax(self.temporal_attn(lstm_out), dim=1)

        z = torch.sum(attn_weights * lstm_out, dim=1)

        return z, attn_map, attn_weights


#lstm one
class EdgePredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 49)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        yL = self.fc2(x)
        return yL

class EdgeModel(nn.Module):# lstm one
    def __init__(self):
        super().__init__()
        self.feature_extractor = EdgeFeatureExtractor()
        self.predictor = EdgePredictor()

    def forward(self, x):
      z, spatial_attn, temporal_attn = self.feature_extractor(x)
      yL = self.predictor(z)
      return yL, z, spatial_attn, temporal_attn


# after lstm

class CloudModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 49)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        yR = self.fc4(x)
        return yR

# class CloudModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.fc1 = nn.Linear(128, 256)
#         self.bn1 = nn.BatchNorm1d(256)

#         self.fc2 = nn.Linear(256, 256)
#         self.bn2 = nn.BatchNorm1d(256)

#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 49)

#     def forward(self, z):
#         x = F.relu(self.bn1(self.fc1(z)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = F.relu(self.fc3(x))
#         yR = self.fc4(x)
#         return yR
