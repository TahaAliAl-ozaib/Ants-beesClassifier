# src/models/train.py
import argparse
import os
import torch
import time

from src.models.model import ImageClassifierModel
from src.utils.data_utils import seed_everything
from config import CONFIG


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with config.py defaults"""
    parser = argparse.ArgumentParser(description="Train Ants vs Bees classifier")
    
    # Use config.py as defaults
    data_config = CONFIG['data']
    training_config = CONFIG['training']
    paths_config = CONFIG['paths']
    
    parser.add_argument("--data-dir", type=str, default=data_config['data_dir'], 
                       help="Root folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=training_config['num_epochs'],
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=data_config['batch_size'],
                       help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=data_config['num_workers'],
                       help="Number of workers for data loading")
    parser.add_argument("--num-classes", type=int, default=data_config.get('num_classes', 2),
                       help="Number of classes")
    parser.add_argument("--save-path", type=str, default=paths_config['model_save_path'],
                       help="Path to save the trained model")
    parser.add_argument("--seed", type=int, default=training_config.get('seed', 42),
                       help="Random seed for reproducibility")
    return parser.parse_args()


def detect_data_root(provided_path: str | None = None) -> str:
    """Detect data root directory if not provided"""
    if provided_path:
        return provided_path
    candidates = [
        os.path.join("Data", "raw"),
        os.path.join("data", "raw"),
        "Data",
        "data",
    ]
    for candidate in candidates:
        train_dir = os.path.join(candidate, "train")
        val_dir = os.path.join(candidate, "val")
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            return candidate
    return "Data"


def main() -> None:
    """Main training function with detailed progress tracking"""
    args = parse_args()
    
    print("🐜🐝 Ants vs Bees Classification Project - Advanced Training")
    print("="*60)
    
    # Print configuration
    print("📋 Training Configuration:")
    print(f"  data_dir: {args.data_dir}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  num_classes: {args.num_classes}")
    print(f"  save_path: {args.save_path}")
    print(f"  seed: {args.seed}")
    
    # Set reproducibility
    print("\n🔧 Setting up reproducibility...")
    seed_everything(args.seed)
    print("✅ Reproducibility setup completed!")
    
    # Detect data root
    data_root = detect_data_root(args.data_dir)
    print(f"\n📊 Using data directory: {data_root}")

    # Create and train model
    print("\n🤖 Creating and training model...")
    try:
        model_wrapper = ImageClassifierModel(
            data_dir=data_root,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_epochs=args.epochs,
        )
        
        print("✅ Model created successfully!")
        print(f"🏷️ Classes: {model_wrapper.class_names}")
        print(f"📊 Dataset sizes: {model_wrapper.dataset_sizes}")
        print(f"🖥️ Using device: {model_wrapper.device}")
        
        # Train the model
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        print("="*50)
        
        start_time = time.time()
        model = model_wrapper.train_model()
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"✅ Model training completed successfully!")
        print(f"⏱️ Total training time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    # Save model
    print(f"\n💾 Saving model to: {args.save_path}")
    try:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
        # Save with metadata like main.py
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': model_wrapper.class_names,
            'num_classes': len(model_wrapper.class_names)
        }, args.save_path)
        
        print(f"✅ Model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print("\n🎉 Advanced training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()


