import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

from src.utils.data_utils import get_device


def build_val_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', ['ants', 'bees'])
    num_classes = checkpoint.get('num_classes', len(class_names))

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, class_names


@torch.inference_mode()
def predict(image_path: str, model_path: str, image_size: int = 224, show: bool = True, save_path: str | None = None):
    device = get_device()
    model, class_names = load_model(model_path, device)
    tfm = build_val_transforms(image_size)

    img = Image.open(image_path).convert('RGB')
    tensor = tfm(img).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0)
    conf, pred_idx = torch.max(probs, dim=0)
    label = class_names[pred_idx.item()]

    print(f"Predicted: {label} (confidence: {conf.item():.4f})")
    # Also print per-class probabilities
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {probs[i].item():.4f}")

    # Visualization
    if show or save_path:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        title = f"Pred: {label}  |  Conf: {conf.item():.2%}"
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        if show:
            plt.show()


def main():
    # Configure your test here (no terminal arguments needed)
    image_path = "data/raw/val/ants/10308379_1b6c72e180.jpg"  # ضع مسار الصورة هنا
    model_path = "ants_bees_model.pth"
    image_size = 224
    show_image = True
    save_path = None  # مثال: "prediction.png" لحفظ النتيجة

    predict(
        image_path=image_path,
        model_path=model_path,
        image_size=image_size,
        show=show_image,
        save_path=save_path,
    )
if __name__ == '__main__':
    main()


