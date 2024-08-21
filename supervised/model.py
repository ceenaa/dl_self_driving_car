import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_actions):
        super(CNN, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=2)

        # Calculate the size after convolutional layers

        # Fully connected layers
        self.fc1 = nn.Linear(140, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
