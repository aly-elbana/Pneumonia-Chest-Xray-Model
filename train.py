import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm

from model import model
from model_cnn import model_cnn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"[INFO] Using device: {device}")
    print("=" * 60)

    # ===========================
    # Data Preprocessing
    # ===========================
    preprocess_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root="./chest_xray/train", transform=preprocess_train)
    val_dataset   = datasets.ImageFolder(root="./chest_xray/val", transform=preprocess)
    test_dataset  = datasets.ImageFolder(root="./chest_xray/test", transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # ===========================
    # Model / Optimizer / Loss
    # ===========================
    model_to_train = model.to(device)
    optimizer = torch.optim.AdamW(model_to_train.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    use_model = True
    model_to_train = model_to_train if use_model else model_cnn
    model_to_train = model_to_train.to(device)
    epochs = 100
    best_score = -float("inf")
    best_model_path = None
    save_dir = "./saved_models_final" if use_model else "./saved_models_cnn"
    os.makedirs(save_dir, exist_ok=True)

    lambda_loss = 0.7

    last_checkpoint = None

    # ===========================
    # Training Loop
    # ===========================
    for epoch in range(epochs):
        model_to_train.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for xb, yb in train_bar:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_to_train(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)

        model_to_train.eval()
        correct, total = 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for xb, yb in val_bar:
                xb, yb = xb.to(device), yb.to(device)
                pred = model_to_train(xb)
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        val_acc = correct / total
        score = (1 - lambda_loss) * val_acc - lambda_loss * avg_loss

        print(f"[RESULT] Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Score: {score:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # best model saving
        if score > best_score:
            best_score = score
            best_model_path = os.path.join(
                save_dir, f"best_model_epoch{epoch+1}_acc{val_acc:.4f}_loss{avg_loss:.4f}_{timestamp}.pth"
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model_to_train.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "val_acc": val_acc,
                "score": score,
                "timestamp": timestamp
            }, best_model_path)
            print(f"[INFO] New best model saved at: {best_model_path}")

        last_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_to_train.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "val_acc": val_acc,
            "score": score,
            "timestamp": timestamp
        }
        last_model_path = os.path.join(save_dir, "last_model_checkpoint.pth")
        torch.save(last_checkpoint, last_model_path)

    # ===========================
    # Save last accuracy separately (JSON)
    # ===========================
    if last_checkpoint is not None:
        acc_file = os.path.join(save_dir, "last_accuracy.json")
        with open(acc_file, "w") as f:
            json.dump({"last_val_accuracy": last_checkpoint["val_acc"]}, f, indent=4)
        print(f"[INFO] Last validation accuracy saved at {acc_file}")
        print(f"[INFO] Last checkpoint saved at {last_model_path}")

    # ===========================
    # Final Testing
    # ===========================
    print("\n" + "=" * 60)
    print("[INFO] Starting final testing on best saved model")
    print("=" * 60)

    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        model_to_train.load_state_dict(checkpoint["model_state_dict"])
        model_to_train.eval()
        correct, total = 0, 0

        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for xb, yb in test_bar:
                xb, yb = xb.to(device), yb.to(device)
                pred = model_to_train(xb)
                _, predicted = torch.max(pred, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

        test_acc = correct / total
        print(f"[FINAL RESULT] Test Accuracy: {test_acc:.4f}")
    else:
        print("[WARNING] No model was saved. Skipping testing.")


if __name__ == "__main__":
    main()
