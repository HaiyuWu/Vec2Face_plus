import torch
import torch.nn as nn


class PoseCondModel(nn.Module):
    def __init__(self):
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()

        # Second conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()

        # Third conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()

        # Fourth conv layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()

        # Flatten
        self.flatten = nn.Flatten()

        # Linear layer to get to the target size
        self.linear = nn.Linear(7 * 7 * 256, 768)

        # Initialize the output layer weights and bias to zero
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Apply conv layers
        x = self.relu1(self.conv1(x))  # [bs, 32, 56, 56]
        x = self.relu2(self.conv2(x))  # [bs, 64, 28, 28]
        x = self.relu3(self.conv3(x))  # [bs, 128, 14, 14]
        x = self.relu4(self.conv4(x))  # [bs, 256, 7, 7]

        # Flatten
        x = self.flatten(x)  # [bs, 12544]

        # Apply zeroed linear layer
        x = self.linear(x)  # [bs, 512] (initially all zeros)

        return x[:, None, :].contiguous()
