# src/utils/data_utils.py
import torch

def get_device():
    """تحديد الجهاز المتاح (GPU أو CPU)"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_dataset_info(dataset_sizes, class_names):
    """طباعة معلومات عن حجم البيانات والفئات"""
    print("Dataset sizes:", dataset_sizes)
    print("Classes:", class_names)