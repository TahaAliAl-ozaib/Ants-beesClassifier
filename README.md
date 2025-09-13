# 🐜🐝 Ants vs Bees Classification Project

## Overview
This project classifies images of ants and bees using deep learning with PyTorch. It follows course requirements: Git, UV for dependencies, clear structure, and comprehensive docs.

--------------------------------------------------------------------------------------------------------------------------------------------------
## Team Members
|  AC.NO   |      Name     |    Role          |                 Contributions                  |
|----------|---------------|------------------|------------------------------------------------|
| 202274263| Taha Al-Ozaib | Lead Developer   |         Data preprocessing, Model development  |
| 202274324| Abdulslam Aldaei |  DL Engineer  |      Optimization, deployment model training   |
| 202174009| Sakhr Altyeb  |    Data Analyst  |                 EDA, visualization             |
--------------------------------------------------------------------------------------------------------------------------------------------------

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
- Python 3.12.4 (specified in `.python-version`)
- UV package manager

### Installation Steps
1. Clone the repository:
```bash
   git clone https://github.com/TahaAliAl-ozaib/Ants-beesClassifier
   cd Ants-beesClassifier
```

2. Install dependencies using UV:
```bash
    uv sync
```
3. Run the project:
```bash
   uv run python main.py
   ```
4. Run the training script
    ```bash
      uv run python src/models/train.py
    ```
 5. Run the project tast:
```bash
uv run streamlit run test.py
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

**Usage:**
```bash
uv run streamlit run test.py
```

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