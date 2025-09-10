# src/data/prepare_data.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils.data_utils import get_device, print_dataset_info

def prepare_data(data_dir="data/raw", batch_size=32, num_workers=4, image_size=224):
    """إعداد البيانات للتدريب"""
    
    # التحقق من وجود البيانات
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training folder not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation folder not found: {val_path}")
    
    # تحديد الجهاز
    device = get_device()
    print(f"Using device: {device}")
    
    # إعداد التحويلات المحسنة
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # تحميل البيانات
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
    }
    
    # إنشاء DataLoaders محسنة
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        ),
        'val': DataLoader(
            image_datasets['val'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    }
    
    # استخراج المعلومات
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    # طباعة المعلومات
    print_dataset_info(dataset_sizes, class_names)
    
    return dataloaders, dataset_sizes, class_names

if __name__ == "__main__":
    dataloaders, dataset_sizes, class_names = prepare_data()
    print("Data preparation completed successfully!")