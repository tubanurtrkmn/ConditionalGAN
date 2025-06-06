import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, class_dim, img_channels):
        super(Generator, self).__init__()
        input_dim = noise_dim + class_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, img_channels * 128 * 128),
            nn.Tanh()
        )
        self.img_channels = img_channels

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        x = self.model(x)
        return x.view(x.size(0), self.img_channels, 128, 128)
