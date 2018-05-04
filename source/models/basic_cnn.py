import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_channels, 8, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool1d(2))
        self.fc = nn.Linear(58 * 4, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out