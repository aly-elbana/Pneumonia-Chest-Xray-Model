import os
import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import model

def build_dataloader(data_root: str, batch_size: int, num_workers: int) -> DataLoader:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_root, transform=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def flexible_load_state_dict(model_obj: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model_obj.load_state_dict(ckpt["model_state_dict"])
            return
        # raw state_dict
        try:
            model_obj.load_state_dict(ckpt)
            return
        except Exception:
            pass
    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")


def evaluate(model_obj: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model_obj.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_obj(xb)
            _, predicted = torch.max(pred, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on a dataset split")
    parser.add_argument("--models_dir", type=str, default="./saved_models_final", help="Directory containing .pth files")
    parser.add_argument("--data_root", type=str, default="./chest_xray/test", help="ImageFolder root to evaluate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    loader = build_dataloader(args.data_root, args.batch_size, args.num_workers)

    best_acc: float = 0.0
    best_file: str = None

    if not os.path.isdir(args.models_dir):
        raise FileNotFoundError(f"Models directory not found: {args.models_dir}")

    files = [f for f in os.listdir(args.models_dir) if f.endswith(".pth")]
    if not files:
        raise FileNotFoundError(f"No .pth files found in: {args.models_dir}")

    for filename in files:
        path = os.path.join(args.models_dir, filename)
        model_to_test = model.to(device)
        try:
            flexible_load_state_dict(model_to_test, path, device)
        except Exception as e:
            print(f"[WARN] Skip {filename}: {e}")
            continue

        acc = evaluate(model_to_test, loader, device)
        print(f"[TEST RESULT] Model: {filename} | Test Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_file = filename

    if best_file:
        print(f"\n# Best Model {best_file} With Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

