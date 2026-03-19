import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from app.services.training.datasets import ECGImageClassifierDataset, CLASS_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = len(CLASS_MAP)
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

TRAIN_DIR = "data/image_training/classifier/train"
VAL_DIR = "data/image_training/classifier/val"
MODEL_OUT = "app/models/ecg_image_classifier.pt"


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    train_ds = ECGImageClassifierDataset(TRAIN_DIR, train=True)
    val_ds = ECGImageClassifierDataset(VAL_DIR, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        running_correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += images.size(0)

            loop.set_postfix(
                loss=running_loss / max(total, 1),
                acc=running_correct / max(total, 1),
            )

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"[VAL] loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"Modelo guardado en {MODEL_OUT}")

    print(f"Mejor val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()