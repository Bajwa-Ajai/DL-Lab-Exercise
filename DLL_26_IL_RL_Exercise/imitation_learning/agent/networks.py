import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=5):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(history_length, 16, kernel_size=8, stride=4),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # (32, 10, 10)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # (64, 8, 8)
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
