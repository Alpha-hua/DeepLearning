import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 2*in_features),
            nn.LeakyReLU(0.01),
            nn.Linear(2*in_features, in_features),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.net(x)
