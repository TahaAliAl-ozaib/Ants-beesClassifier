# src/data/prepare_data.py
import os
import platform
import multiprocessing
import torch
from torchvision import datasets, transforms

# ============================
# 1. تحديد الـ device (GPU / CPU)
# ============================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================
# 2. إعداد التحويلات (transforms)
# ============================
# للتحسين والتدريب
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),    # إعادة التحجيم
    transforms.RandomHorizontalFlip(), # انعكاس أفقي عشوائي
    transforms.ToTensor(),             # تحويل الصورة إلى Tensor
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]) # التطبيع
])

# للتحقق (validation) – فقط إعادة تحجيم وتطبيع
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# 3. تحميل البيانات (ImageFolder)
# ============================
data_dir = "E:/FiveLevel/AI/OurProject/AIPROJECT/data/raw"  # غيّر المسار إذا بياناتك في مكان ثاني

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)
}

# ============================
# 4. إنشاء DataLoader
# ============================
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False)
}

# ============================
# 5. استخراج معلومات البيانات
# ============================
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# ============================
# 6. اختبار سريع
# ============================
if __name__ == "__main__":
    print("Dataset sizes:", dataset_sizes)
    print("Classes:", class_names)
# دالة مساعدة لطباعه ملخص سريع
def print_dataset_info():
    print("Device:", device)
    print("Data dir:", data_dir)
    print("Num workers:", num_workers)
    print("Batch size:", batch_size)
    print("Dataset sizes:", dataset_sizes)
    print("Classes:", class_names)