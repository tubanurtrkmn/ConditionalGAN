import os
import torch
from torchvision.utils import save_image
from data_loader import get_loader
from cgan import CGAN
from config import EPOCHS, SAVE_INTERVAL, LATENT_DIM


os.makedirs("images", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    dataloader, label_map = get_loader()
    model = CGAN()
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dataloader):
            g_loss, d_loss = model.train_step(imgs, labels)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")

        if (epoch + 1) % SAVE_INTERVAL == 0:
            model.generator.eval()
            with torch.no_grad():
                sample_count = min(300, len(dataloader.dataset))
                z = torch.randn(sample_count, LATENT_DIM, device=device)
                labels = torch.randint(0, len(label_map), (sample_count,), device=device)
                gen_imgs = model.generator(z, labels)

                real_imgs = imgs[:sample_count].to(device)
                real_imgs_fid = ((real_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                gen_imgs_fid = ((gen_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)

                fid.update(real_imgs_fid, real=True)
                fid.update(gen_imgs_fid, real=False)
                fid_score = fid.compute().item()
                fid.reset()

            print(f"FID Score: {fid_score:.4f}")
            save_image(gen_imgs.data[:25], f"images/{epoch+1}.png", nrow=5, normalize=True)
            model.generator.train()

    torch.save(model.generator.state_dict(), "generator_final.pth")
    print("Eğitilen generator modeli 'generator_final.pth' olarak kaydedildi.")

if __name__ == '__main__':
    train()
