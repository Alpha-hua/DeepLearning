import torch.nn as nn

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(429, 858),
            nn.BatchNorm1d(858),
            nn.LeakyReLU(0.01),
            nn.Linear(858, 429),
            nn.BatchNorm1d(429),
            nn.LeakyReLU(0.01),
            nn.Linear(429, 39),
        )

    def forward(self, x):
        return self.net(x)