import os
import io
import cv2
import torch
import numpy as np
from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "app/models/ecg_image_segmenter.pt"


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


_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No existe modelo de segmentación: {MODEL_PATH}")
        model = UNetSmall()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def get_transforms():
    return Compose([
        Resize(512, 512),
        Normalize(),
        ToTensorV2(),
    ])


def segment_trace_mask(raw_bytes: bytes):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image_np = np.array(image)

    transforms = get_transforms()
    tensor = transforms(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (probs > 0.5).astype(np.uint8) * 255
    return mask