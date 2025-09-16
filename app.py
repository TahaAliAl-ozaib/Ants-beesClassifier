import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from src.utils.data_utils import get_device
from config import CONFIG


def build_val_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
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


device = get_device()
model, class_names = load_model(CONFIG['paths']['model_save_path'], device)
tfm = build_val_transforms(CONFIG['data']['image_size'])


def predict(img: Image.Image):
    tensor = tfm(img).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0)

    results = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    label = max(results, key=results.get)
    return label, results


# -------- واجهة Gradio -------- #
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.JSON(label="Probabilities"),
    ],
    title="🐜🐝 Ants vs Bees Classifier",
    description="ارفع صورة نملة أو نحلة وسيقوم النموذج بالتنبؤ بها."
)

if __name__ == "__main__":
    demo.launch()
