import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
import torch.nn as nn

from app.services.training.datasets import ECGImageClassifierDataset, CLASS_MAP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "data/image_training/classifier/test"
MODEL_PATH = "app/models/ecg_image_classifier.pt"

IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_MAP))
    return model


def main():
    ds = ECGImageClassifierDataset(TEST_DIR, train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    print("Test samples:", len(ds))

    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(
        y_true,
        y_pred,
        target_names=[IDX_TO_CLASS[i] for i in range(len(CLASS_MAP))]
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
