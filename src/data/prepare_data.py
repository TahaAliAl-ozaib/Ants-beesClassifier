# src/data/prepare_data.py
import os
import platform
import multiprocessing
import torch
from torchvision import datasets, transforms

# ---------------------------
# 1) تحديد الجهاز (GPU أو CPU)
# ---------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2) إعداد التحويلات (transforms)
# ---------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),   # قص وتغيير الحجم عشوائياً
        transforms.RandomHorizontalFlip(),   # قلب أفقي عشوائي
        transforms.ToTensor(),               # تحويل لصيغة Tensor
        transforms.Normalize([0.485, 0.456, 0.406],  # تطبيع كقيم ImageNet
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# ---------------------------
# 3) مسار البيانات (قابل للتعديل)
#    - افتراضي: project_root/data/hymenoptera_data/{train,val}
#    - يمكن تجاوزها عبر متغير بيئة HYMENOPTERA_DATA_DIR
# ---------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
default_data_dir = os.path.join(_project_root, 'data', 'hymenoptera_data')
data_dir = os.environ.get('HYMENOPTERA_DATA_DIR', default_data_dir)

# ---------------------------
# 4) إعداد عدد العمال (num_workers) بطريقة آمنة لويندوز
# ---------------------------
if platform.system() == "Windows":
    num_workers = 0  # على ويندوز غالبًا اجعل 0 لتفادي مشاكل multiprocessing
else:
    num_workers = min(4, multiprocessing.cpu_count())

# متغيرات يمكن تعديلها بسهولة
batch_size = 4
pin_memory = True if device.type == 'cuda' else False

# ---------------------------
# 5) تحميل المجلدات باستخدام ImageFolder
# ---------------------------
# تأكد أن المجلدات موجودة: data_dir/train  و data_dir/val
for part in ['train', 'val']:
    expected = os.path.join(data_dir, part)
    if not os.path.isdir(expected):
        raise FileNotFoundError(f"Folder not found: {expected}\nضع بياناتك في هذا المسار أو عدّل `data_dir` في prepare_data.py")

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

# ---------------------------
# 6) إنشاء DataLoader
# ---------------------------
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    for x in ['train', 'val']
}

# ---------------------------
# 7) معلومات مفيدة
# ---------------------------
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# دالة مساعدة لطباعه ملخص سريع
def print_dataset_info():
    print("Device:", device)
    print("Data dir:", data_dir)
    print("Num workers:", num_workers)
    print("Batch size:", batch_size)
    print("Dataset sizes:", dataset_sizes)
    print("Classes:", class_names)