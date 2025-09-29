import os
import argparse
from typing import List

import torch
from torchvision import transforms
from PIL import Image

from model import model


CLASSES: List[str] = ["Bacteria_PNEUMONIA", "NORMAL", "Virus_PNEUMONIA"]


def build_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def flexible_load_state_dict(model_obj: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_obj.load_state_dict(ckpt["model_state_dict"])
        return
    if isinstance(ckpt, dict):
        try:
            model_obj.load_state_dict(ckpt)
            return
        except Exception:
            pass
    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")


def predict_image(model_obj: torch.nn.Module, image_path: str, device: torch.device, preprocess: transforms.Compose) -> str:
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_obj(input_tensor)
        _, predicted = torch.max(output, 1)
        return CLASSES[predicted.item()]


def main():
    parser = argparse.ArgumentParser(description="Single-image inference over a folder")
    parser.add_argument("--images_dir", type=str, default="./test_images", help="Folder containing images")
    parser.add_argument("--model_path", type=str, default="./saved_models_final/best_model_epoch68_acc0.8824_loss0.1624_20250924_053949.pth", help="Path to .pth checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    preprocess = build_preprocess()
    model_to_test = model.to(device)
    flexible_load_state_dict(model_to_test, args.model_path, device)
    model_to_test.eval()

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(exts)]
    if not files:
        print("[WARN] No images found.")
        return

    for img_file in files:
        img_path = os.path.join(args.images_dir, img_file)
        pred = predict_image(model_to_test, img_path, device, preprocess)
        print(f"[PREDICTION] {img_file} => {pred}")


if __name__ == "__main__":
    main()
