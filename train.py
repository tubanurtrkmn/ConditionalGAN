import torch
import torch.nn as nn
import torchvision.utils as vutils
from generator import Generator
from discriminator import Discriminator
from data_loader import get_loader
import os

# Basit görüntü kaydetme fonksiyonu
def generate_and_save_image(image_tensor, filename):
    image_tensor = (image_tensor + 1) / 2  # [-1, 1] → [0, 1]
    vutils.save_image(image_tensor, filename)

# Donanım kontrolü
device = torch.device("cpu")  # GPU olmadığı için CPU

# Hiperparametreler
epochs = 200
batch_size = 4
z_dim = 100
num_classes = 4
image_size = 64  # Daha küçük çözünürlük
learning_rate = 0.0002

# Model tanımı
G = Generator(z_dim, num_classes, image_size).to(device)
D = Discriminator(num_classes, image_size).to(device)
criterion = nn.BCELoss()
G_opt = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_opt = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Veriyi al
dataloader = get_loader(batch_size=batch_size, image_size=image_size, num_workers=0)

# Early stopping
early_stopping_counter = 0
early_stopping_patience = 15
min_G_loss = float("inf")

# Çıktı klasörü oluştur
os.makedirs("outputs", exist_ok=True)

# Eğitim döngüsü
for epoch in range(epochs):
    for real_images, labels in dataloader:
        real_images, labels = real_images.to(device), labels.to(device)
        bs = real_images.size(0)

        # Gerçek / sahte hedefler
        real_targets = torch.ones(bs, 1).to(device)
        fake_targets = torch.zeros(bs, 1).to(device)

        # === Discriminator Eğitimi ===
        z = torch.randn(bs, z_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (bs,)).to(device)
        fake_images = G(z, gen_labels)

        D_real = D(real_images, labels)
        D_fake = D(fake_images.detach(), gen_labels)
        D_loss = criterion(D_real, real_targets) + criterion(D_fake, fake_targets)

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        # === Generator Eğitimi ===
        z = torch.randn(bs, z_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (bs,)).to(device)
        fake_images = G(z, gen_labels)
        D_fake = D(fake_images, gen_labels)
        G_loss = criterion(D_fake, real_targets)

        G.zero_grad()
        G_loss.backward()
        G_opt.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")

    # === Early Stopping ===
    if G_loss.item() < min_G_loss:
        min_G_loss = G_loss.item()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # === Görsel Üretimi: Her 20 epoch'ta ve ilk epoch'ta ===
    if (epoch + 1) % 20 == 0 or epoch == 0:
        with torch.no_grad():
            z = torch.randn(1, z_dim).to(device)
            class_label = torch.tensor([3]).to(device)  # lower front
            fake_image = G(z, class_label)
            generate_and_save_image(fake_image, f"outputs/epoch_{epoch+1}_lower_front.png")

# Modeli kaydet
torch.save(G.state_dict(), "generator.pth")
