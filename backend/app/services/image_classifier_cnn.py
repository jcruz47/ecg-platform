import io
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
MODEL_PATH = BASE_DIR / "app" / "models" / "ecg_image_classifier.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IDX_TO_CLASS = {
    0: "normal",
    1: "abnormal",
    2: "poor_quality",
}


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 3)
    return model


def get_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ])


_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No existe modelo: {MODEL_PATH}")

        model = build_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def classify_ecg_image(raw_bytes: bytes):
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image_np = np.array(image)

    transforms = get_transforms()
    tensor = transforms(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities = {
        IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(probs))
    }

    return {
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_name": "efficientnet_b0_ecg_image_v1",
    }