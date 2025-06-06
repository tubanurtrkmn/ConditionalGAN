from generator import Generator
import torch
import torchvision.utils as vutils
import torch.nn.functional as F

LABELS = {
    "upper_back": 0,
    "upper_front": 1,
    "lower_back": 2,
    "lower_front": 3
}
NUM_CLASSES = len(LABELS)

def generate_image(target_class="lower_front", model_path="generator.pth"):
    device = torch.device("cpu")
    G = Generator(noise_dim=100, class_dim=NUM_CLASSES, img_channels=3).to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()

    noise = torch.randn(1, 100).to(device)
    label_idx = torch.tensor([LABELS[target_class]]).to(device)
    label_onehot = F.one_hot(label_idx, NUM_CLASSES).float().to(device)

    with torch.no_grad():
        fake_img = G(noise, label_onehot)
        vutils.save_image(fake_img, f"{target_class}_generated.png", normalize=True)

generate_image("upper_back")
