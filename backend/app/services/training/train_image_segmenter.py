import os
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Estructura esperada:
# data/image_training/segmentation/images/xxx.png
# data/image_training/segmentation/masks/xxx.png
IMAGES_DIR = Path("data/image_training/segmentation/images")
MASKS_DIR = Path("data/image_training/segmentation/masks")
MODEL_PATH = Path("app/models/ecg_image_segmenter.pt")

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42


class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.p1 = torch.nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = torch.nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256)

        self.u1 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c1 = DoubleConv(256, 128)
        self.u2 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c2 = DoubleConv(128, 64)
        self.out = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))

        x = self.u1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.c1(x)

        x = self.u2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.c2(x)

        return self.out(x)


class ECGSegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_files = sorted(
            [
                p for p in images_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ]
        )

        if not self.image_files:
            raise FileNotFoundError(f"No se encontraron imágenes en: {images_dir}")

        missing_masks = []
        for img_path in self.image_files:
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                missing_masks.append(img_path.name)

        if missing_masks:
            raise FileNotFoundError(
                "Faltan máscaras para estas imágenes:\n" + "\n".join(missing_masks[:20])
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # binarizar máscara
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return 1.0 - dice


def combined_loss(logits, targets):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss(logits, targets)
    return bce + dloss


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = combined_loss(logits, masks)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"No existe carpeta de imágenes: {IMAGES_DIR}")
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"No existe carpeta de máscaras: {MASKS_DIR}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    full_dataset = ECGSegmentationDataset(
        IMAGES_DIR,
        MASKS_DIR,
        transform=None,
    )

    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = ECGSegmentationDataset(IMAGES_DIR, MASKS_DIR, transform=get_train_transform())
    val_dataset = ECGSegmentationDataset(IMAGES_DIR, MASKS_DIR, transform=get_val_transform())

    # usar mismos índices del split
    train_dataset.image_files = [full_dataset.image_files[i] for i in train_subset.indices]
    val_dataset.image_files = [full_dataset.image_files[i] for i in val_subset.indices]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UNetSmall().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    print(f"Entrenando en: {DEVICE}")
    print(f"Train: {len(train_dataset)} imágenes")
    print(f"Val: {len(val_dataset)} imágenes")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate(model, val_loader)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Mejor modelo guardado en: {MODEL_PATH}")

    print("Entrenamiento finalizado.")
    print(f"Mejor val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()