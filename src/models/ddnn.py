import torch.nn as nn
from .edge_model import EdgeModel
from .cloud_model import CloudModel

class DDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = EdgeModel()
        self.cloud = CloudModel()

    def forward(self, x):
        z, yL = self.edge(x)
        yR = self.cloud(z)
        return yL, yR
