# src/data/prepare_data.py
import os
from src.utils.data_utils import (
    get_device,
    print_dataset_info,
    create_dataloaders,
)

def prepare_data(data_dir="data/raw", batch_size=32, num_workers=4, image_size=224):
    """إعداد البيانات للتدريب (واجهة بسيطة تُفوّض التنفيذ إلى utils)."""
    # التحقق من وجود البيانات سريعاً
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training folder not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation folder not found: {val_path}")

    device = get_device()
    dataloaders, dataset_sizes, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        device=device,
    )
    print_dataset_info(dataset_sizes, class_names)
    return dataloaders, dataset_sizes, class_names

if __name__ == "__main__":
    dataloaders, dataset_sizes, class_names = prepare_data()
    print("Data preparation completed successfully!")