# config.py
"""
Configuration file for Ants vs Bees Classification Project
"""

# Data configuration
DATA_CONFIG = {
    'data_dir': 'data/raw',
    'batch_size': 32,
    'num_workers': 4,
    'image_size': 224,
    'train_ratio': 0.8,
    'val_ratio': 0.2
}

# Model configuration
MODEL_CONFIG = {
    'model_name': 'resnet18',
    'pretrained': True,
    'num_classes': 2,
    'dropout_rate': 0.5
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 25,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'step_size': 7,
    'gamma': 0.1,
    'patience': 10  # For early stopping
}

# Paths configuration
PATHS_CONFIG = {
    'model_save_path': 'ants_bees_model.pth',
    'logs_dir': 'logs',
    'results_dir': 'results',
    'checkpoints_dir': 'checkpoints'
}

# Device configuration
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0
}

# All configurations combined
CONFIG = {
    'data': DATA_CONFIG,
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'paths': PATHS_CONFIG,
    'device': DEVICE_CONFIG
}
