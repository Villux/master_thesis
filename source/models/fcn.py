import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, n_channels):
        super(FCN, self).__init__()
        # In: 252 out 239
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=10, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # In: 248 out 237
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        # In: 245 out 244
        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 81, kernel_size=3, stride=1),
            nn.BatchNorm1d(81),
            nn.ReLU())

        self.GPL = nn.AvgPool1d(122)
        self.sofmax = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.GPL(out)
        out = out.view(-1, 81)
        return self.sofmax(out)