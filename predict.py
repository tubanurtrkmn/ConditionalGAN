import torch
from generator import Generator
from torchvision.utils import save_image
from config import LATENT_DIM, IMG_SIZE, IMG_CHANNELS, NUM_CLASSES
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
generator.eval()

os.makedirs("predicted_images", exist_ok=True)

num_samples = 10
z = torch.randn(num_samples, LATENT_DIM, device=device)
labels = torch.randint(0, NUM_CLASSES, (num_samples,), device=device)

with torch.no_grad():
    gen_imgs = generator(z, labels)

save_image(gen_imgs.data, "predicted_images/generated.png", nrow=5, normalize=True)
print("Görseller 'predicted_images/generated.png' dosyasına kaydedildi.")
