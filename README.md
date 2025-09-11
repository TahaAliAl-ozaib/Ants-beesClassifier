# 🐜🐝 Ants vs Bees Classification Project

## Overview
This project classifies images of ants and bees using deep learning with PyTorch. It follows course requirements: Git, UV for dependencies, clear structure, and comprehensive docs.




## Project Structure
```
AIPROJECT/
├── main.py                 # Main training script
├── config.py              # Configuration settings
├── src/
│   ├── data/
│   │   └── prepare_data.py # Data preparation
│   └── utils/
│       └── data_utils.py   # Utility functions
└── data/
    └── raw/
        ├── train/
        │   ├── ants/
        │   └── bees/
        └── val/
            ├── ants/
            └── bees/
```

## Installation and Setup (UV)

### Prerequisites
- Python 3.12+
- UV package manager

### Steps
```bash
# Install deps
uv sync

# Train
uv run python main.py
```

## Configuration
Edit `config.py` to modify:
- Batch size
- Number of epochs
- Learning rate
- Model architecture

## Expected Output
```
🐜🐝 Ants vs Bees Classification Project
==================================================
📋 Configuration:
  data_dir: data/raw
  batch_size: 32
  num_epochs: 25

📊 Step 1: Preparing data...
Using device: cuda:0
✅ Data preparation completed successfully!

🤖 Step 2: Creating model...
✅ Model created with 2 classes: ['ants', 'bees']

🎯 Step 3: Training model...
🚀 Starting training for 25 epochs...
[Training progress...]

💾 Step 4: Saving model...
✅ Model saved successfully!

🎉 Project completed successfully!
```

## Files Created
- `ants_bees_model.pth` - Trained model
- Training logs and metrics

## Usage (Inference)
Classify a single image and visualize the prediction (example code in notebooks suggested):
```python
from torchvision.models import resnet18, ResNet18_Weights
import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms

ckpt = torch.load("ants_bees_model.pth", map_location="cpu")
classes = ckpt['class_names']
m = resnet18(weights=ResNet18_Weights.DEFAULT)
m.fc = nn.Linear(m.fc.in_features, len(classes))
m.load_state_dict(ckpt['model_state_dict'])
m.eval()

val_tfm = transforms.Compose([
    transforms.Resize((256,256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img = Image.open("data/raw/val/ants/10308379_1b6c72e180.jpg").convert('RGB')
x = val_tfm(img).unsqueeze(0)
with torch.inference_mode():
    p = torch.softmax(m(x), dim=1)[0]
conf, idx = torch.max(p, 0)
print(f"Pred: {classes[idx]} ({conf.item():.2%})")
```

## Team Members
| AC.NO | Name | Role | Contributions |
|---|---|---|---|
| 1 | Your Name | Lead Developer | Data prep, model training |
| 2 | Teammate | Data Analyst | EDA, visualization |
| 3 | Teammate | ML Engineer | Optimization, deployment |

## Next Steps
1. Test the model on new images
2. Create inference script
3. Deploy the model