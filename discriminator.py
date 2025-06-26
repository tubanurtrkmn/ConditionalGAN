import torch
import torch.nn as nn
from config import IMG_CHANNELS, NUM_CLASSES, IMG_SIZE

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, IMG_SIZE * IMG_SIZE)

        self.model = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels).view(labels.size(0), 1, IMG_SIZE, IMG_SIZE)
        d_input = torch.cat((img, label_input), dim=1)
        return self.model(d_input)
