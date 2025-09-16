    # Ants vs Bees Classification Project

## Overview
This project classifies images of ants and bees using deep learning with PyTorch. 

--------------------------------------------------------------------------------------------------------------------------------------------------
    ## Team Members
    |  AC.NO   |      Name     |    Role          |                     Contributions                  |
    |----------|---------------|------------------|------------------------------------------------|
    | 202274263| Taha Al-Ozaib | Lead Developer   |         Data preprocessing, Model development  |
    | 202274324| Abdulslam Aldaei |  DL Engineer  |      Optimization, deployment     model training   |
    | 202174009| Sakhr Altyeb  |    Data Analyst  |                 EDA,    visualization             |
--------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure
```
AIPROJECT/
├── test.py                 # Model Testing
├── REDAME.md
├── main.py                 # Main training script
├── config.py               # Configuration settings
├── src/
│   ├── data/
│   │   └── prepare_data.py # Data preparation
│   ├── models/
|   |   ├── model.py        # Image classification system 
|   |   └── train.py        # Model Training
|   └── utils/
│       └── data_utils.py   # Utility functions
├── data/
|    └── raw/
|       ├── train/
|       │   ├── ants/
|       │   └── bees/
|       └── val/
|           ├── ants/
|           └── bees/
├──norebooks/
|   ├── EDA.pynb            # Visualization
|   └── notes.ipynb         # 
└── docs/                   # Additional documentation
    └── Exp.ipyb

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



## Files Created
- `ants_bees_model.pth` - Trained model
- Training logs and metrics

**Usage:**
```bash
uv run streamlit run test.py
```