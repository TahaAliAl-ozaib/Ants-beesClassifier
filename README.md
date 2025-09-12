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
