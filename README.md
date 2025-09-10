# 🐜🐝 Ants vs Bees Classification Project

## Overview
This project classifies images of ants and bees using deep learning with PyTorch.

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

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib numpy
```

### 2. Run Training
```bash
python main.py
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

## Next Steps
1. Test the model on new images
2. Create inference script
3. Deploy the model