# main.py
"""
Main script for Ants vs Bees Classification Project
This script coordinates data preparation and model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os

# Import our custom modules
from src.data.prepare_data import prepare_data
from src.utils.data_utils import get_device

def create_model(num_classes=2, pretrained=True):
    """Create and configure the model"""
    from torchvision import models
    
    # Use ResNet18 as base model
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final layer for our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, dataloaders, dataset_sizes, device, num_epochs=25):
    """Train the model"""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("="*50)
    
    since = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 20)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if it's the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    time_elapsed = time.time() - since
    print(f'\n⏱️ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best validation accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def save_model(model, class_names, save_path='model.pth'):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names)
    }, save_path)
    print(f"💾 Model saved to: {save_path}")

def main():
    """Main function to run the complete pipeline"""
    
    print("🐜🐝 Ants vs Bees Classification Project")
    print("="*50)
    
    # Configuration
    config = {
        'data_dir': 'data/raw',
        'batch_size': 32,
        'num_workers': 4,
        'image_size': 224,
        'num_epochs': 25,
        'model_save_path': 'ants_bees_model.pth'
    }
    
    print("📋 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Prepare data
    print("\n📊 Step 1: Preparing data...")
    try:
        dataloaders, dataset_sizes, class_names = prepare_data(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=config['image_size']
        )
        print("✅ Data preparation completed successfully!")
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        return
    
    # Step 2: Create model
    print("\n🤖 Step 2: Creating model...")
    try:
        model = create_model(num_classes=len(class_names))
        print(f"✅ Model created with {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Step 3: Get device
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    # Step 4: Train model
    print("\n🎯 Step 3: Training model...")
    try:
        trained_model = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            num_epochs=config['num_epochs']
        )
        print("✅ Model training completed successfully!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Step 5: Save model
    print("\n💾 Step 4: Saving model...")
    try:
        save_model(trained_model, class_names, config['model_save_path'])
        print("✅ Model saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print("\n🎉 Project completed successfully!")
    print("="*50)
    print("📁 Files created:")
    print(f"  - Trained model: {config['model_save_path']}")
    print("\n🚀 Next steps:")
    print("  1. Test the model on new images")
    print("  2. Create an inference script")
    print("  3. Deploy the model")

if __name__ == "__main__":
    main()
