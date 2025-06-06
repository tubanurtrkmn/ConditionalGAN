from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch

LABELS = {
    "upper_back": 0,
    "upper_front": 1,
    "lower_back": 2,
    "lower_front": 3
}

class ThermalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for label_name, label in LABELS.items():
            folder_path = os.path.join(root_dir, label_name)
            for img_file in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_loader(batch_size=4, image_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1] aralığı
    ])

    dataset = ImageFolder(root="thermal", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader
