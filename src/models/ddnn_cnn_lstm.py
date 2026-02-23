import torch.nn as nn
from .cnn_lstm_edge import CNNLSTMEdge
from .cnn_lstm_cloud import CNNLSTMCloud

class DDNN_CNN_LSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.edge = CNNLSTMEdge()
        self.cloud = CNNLSTMCloud()

    def forward(self, x):
        h_edge, y_L = self.edge(x)
        y_R = self.cloud(h_edge)

        return y_L, y_R, h_edge