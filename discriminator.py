import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, class_dim):
        super(Discriminator, self).__init__()
        input_dim = img_channels * 128 * 128 + class_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = img.view(img.size(0), -1)
        x = torch.cat([x, labels], dim=1)
        return self.model(x)
