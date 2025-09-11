# src/utils/data_utils.py
import torch
import os
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_device():
    """تحديد الجهاز المتاح (GPU أو CPU)"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_dataset_info(dataset_sizes, class_names):
    """طباعة معلومات عن حجم البيانات والفئات"""
    print("\n" + "="*40)
    print("📊 DATASET INFORMATION")
    print("="*40)
    print(f"🏷️ Classes: {class_names}")
    print(f"📈 Training samples: {dataset_sizes['train']:,}")
    print(f"📉 Validation samples: {dataset_sizes['val']:,}")
    print(f"📊 Total samples: {sum(dataset_sizes.values()):,}")
    
    # حساب النسب
    total = sum(dataset_sizes.values())
    train_ratio = dataset_sizes['train'] / total * 100
    val_ratio = dataset_sizes['val'] / total * 100
    print(f"📈 Train ratio: {train_ratio:.1f}%")
    print(f"📉 Val ratio: {val_ratio:.1f}%")
    print("="*40)


def seed_everything(seed: int = 42) -> None:
    """Make runs reproducible across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 224, augment: bool = True):
    """Return torchvision transforms for train/val."""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    device: torch.device | None = None,
):
    """Create train/val dataloaders with consistent transforms and options."""
    if device is None:
        device = get_device()

    train_tf = build_transforms(image_size=image_size, augment=True)
    val_tf = build_transforms(image_size=image_size, augment=False)

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_tf),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_tf),
    }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        ),
        'val': DataLoader(
            image_datasets['val'], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        ),
    }

    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names