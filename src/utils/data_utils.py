# src/utils/data_utils.py
import torch

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