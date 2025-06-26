import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from config import LATENT_DIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class CGAN:
    def __init__(self):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.adversarial_loss = nn.BCELoss()

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def train_step(self, imgs, labels):
        imgs = imgs.to(device)
        labels = labels.to(device)

        batch_size = imgs.size(0)
        valid = torch.empty(batch_size, 1).uniform_(0.8, 1.0).to(device)
        fake = torch.empty(batch_size, 1).uniform_(0.0, 0.1).to(device)

        self.optimizer_G.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_imgs = self.generator(z, labels)
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, labels), valid)
        g_loss.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        return g_loss.item(), d_loss.item()
