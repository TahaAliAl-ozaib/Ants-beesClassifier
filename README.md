# Ant&Bees Classifire AI Project

## Project Description
A project for image processing and differentiating between ants and bees . The project is also connected to ESP32 via the ESP NOW protocol and MQTT protocol. the project is for learning purposes

_______________________________________________________________________________________________________________

## Team Members and Responsibilities

| AC.NO     | Name          |              Role              | Branch |               Contributions                  | Files/Folder |
|-----------|---------------|--------------------------------|--------|--------------------------------------------------|----------|
| 202274263 | Taha AL-Ozaib | Lead Developer & Data Engineer | `data` | Data collection, preprocessing, train/test split | `src/data/prepare_data.py`, `data/` |
| 202274 | Abdulsalam Aldaai | ML Engineer | `models` | Model creation, training, evaluation, hyperparameter tuning | `src/models/` |
| 202170009 | Sakhr Altyeb | Utils & Deployment | `utils` | Helper functions, visualization, optional Streamlit app | `src/utils/`, `notebooks/`, `docs/` |

____________________________________________________________________________________________________________________________________
## Installation and Setup

### Prerequisites
- Python 3.12+
- UV package manager
- PyTorch
- torchvision
- matplotlib
- PIL
____________________________________________________________________________________________________________________________________
### Installation Steps
1. Clone repository
```bash
git clone https://github.com/TahaAliAl-ozaib/Ants-beesClassifier
cd AIPROJECT

2. Sync dependencies with UV

uv sync

3. المشروع تشغيل ملفات 

uv run python src/data/prepare_data.py ( uv run python prepare_data.py اكتب (data)واذا_كنت_داخل_ملف)

---

Project Structure

AIPROJECT/
├── README.md
├── pyproject.toml
├── src/
│   ├── data/
│   │   └── prepare_data.py
│   └── models/
│   └── utils/
├── data/
│   ├── train/
│   │   ├── ants/
│   │   └── bees/
│   └── val/
│       ├── ants/
│       └── bees/
└── docs/

---

Current Progress

DataLoader جاهز 

Dataset sizes:

train: 244

val: 153

Classes: ['ants', 'bees']

التدريب جهاز : CPU 

---

Usage

Prepare Data

uv run python src/data/prepare_data.py