import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


CLASS_MAP = {
    "normal": 0,
    "abnormal": 1,
    "poor_quality": 2,
}


def build_classifier_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.Rotate(limit=3, p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ])


class ECGImageClassifierDataset(Dataset):
    def __init__(self, root_dir: str, train=True):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.transforms = build_classifier_transforms(train=train)

        for class_name, class_id in CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    self.samples.append((str(img_path), class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image=np.array(image))["image"]
        return image, label