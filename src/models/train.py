# src/models/train.py
import os
import torch
from src.models.model import ImageClassifierModel
from config import CONFIG


def detect_data_root(provided_path: str | None = None) -> str:
    if provided_path:
        return provided_path
    candidates = [
        os.path.join("Data", "raw"),
        os.path.join("data", "raw"),
        "Data",
        "data",
    ]
    for candidate in candidates:
        train_dir = os.path.join(candidate, "train")
        val_dir = os.path.join(candidate, "val")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            return candidate
    return "Data"


def main() -> None:
    # اختيار مصدر البيانات من CONFIG
    data_root = detect_data_root(CONFIG['data']['data_dir'])

    # بناء نموذج باستخدام الإعدادات
    model_wrapper = ImageClassifierModel(
        data_dir=data_root,
        num_classes=CONFIG['model']['num_classes'],
        batch_size=CONFIG['data']['batch_size'],
        num_epochs=CONFIG['training']['num_epochs'],
    )

    # تدريب النموذج
    model = model_wrapper.train_model()

    # حفظ النتائج
    save_path = CONFIG['paths']['model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"✅ Model saved to: {save_path}")


    

