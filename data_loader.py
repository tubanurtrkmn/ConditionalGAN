from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import DATA_PATH, IMG_SIZE, BATCH_SIZE

label_map = {
    "lower_back": 0,
    "lower_front": 1,
    "upper_back": 2,
    "upper_front": 3
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_loader():
    dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), label_map
