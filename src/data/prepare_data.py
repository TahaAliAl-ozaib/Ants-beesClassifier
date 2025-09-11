# src/data/prepare_data.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils.data_utils import get_device, print_dataset_info

# ============================
# 1. تحديد الجهاز
# ============================
device = get_device()
print(f"Using device: {device}")

# ============================
# 2. إعداد التحويلات
# ============================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# 3. تحديد مسار البيانات
# ============================
data_dir = "data/raw"  # تأكد أن train و val داخل هذا المجلد

# ============================
# 4. تحميل البيانات
# ============================
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
}

# ============================
# 5. إنشاء DataLoader
# ============================
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}

# ============================
# 6. استخراج معلومات البيانات
# ============================
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# ============================
# 7. اختبار سريع
# ============================
if __name__ == "__main__":
    print_dataset_info(dataset_sizes, class_names)