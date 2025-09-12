import argparse
import os
import torch

from src.models.model import ImageClassifierModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ants vs Bees classifier")
    parser.add_argument("--data-dir", type=str, default=None, help="Root folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--save-path", type=str, default=os.path.join("notebooks", "ant_bee_model.pt"))
    return parser.parse_args()


def detect_data_root(provided_path: str | None = None) -> str:
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
    args = parse_args()
    data_root = detect_data_root(args.data_dir)

    model_wrapper = ImageClassifierModel(
        data_dir=data_root,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    model = model_wrapper.train_model()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model weights to: {args.save_path}")


if __name__ == "__main__":
    main()


