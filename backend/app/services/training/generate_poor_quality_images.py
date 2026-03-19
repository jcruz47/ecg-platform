from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path("data/image_training/classifier")
RNG = random.Random(42)
SPLITS = ["train", "val", "test"]


def degrade_image(img):
    h, w = img.shape[:2]

    if RNG.random() < 0.7:
        k = RNG.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if RNG.random() < 0.7:
        alpha = RNG.uniform(0.5, 1.5)
        beta = RNG.randint(-50, 50)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if RNG.random() < 0.7:
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        shift = min(h, w) * 0.08
        dst = np.float32([
            [RNG.uniform(0, shift), RNG.uniform(0, shift)],
            [w - 1 - RNG.uniform(0, shift), RNG.uniform(0, shift)],
            [w - 1 - RNG.uniform(0, shift), h - 1 - RNG.uniform(0, shift)],
            [RNG.uniform(0, shift), h - 1 - RNG.uniform(0, shift)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h))

    if RNG.random() < 0.6:
        noise = np.random.normal(0, 12, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def main():
    for split in SPLITS:
        pq_dir = ROOT / split / "poor_quality"
        pq_dir.mkdir(parents=True, exist_ok=True)

        normal_imgs = list((ROOT / split / "normal").glob("*.png"))
        abnormal_imgs = list((ROOT / split / "abnormal").glob("*.png"))
        candidates = normal_imgs + abnormal_imgs

        RNG.shuffle(candidates)

        existing = len(list(pq_dir.glob("*.png")))
        target_count = min(len(candidates) // 3, 800 if split == "train" else 150)

        if existing >= target_count:
            print(f"[INFO] poor_quality {split} ya tiene suficientes imágenes ({existing})")
            continue

        needed = target_count - existing

        for img_path in tqdm(candidates[:needed], desc=f"Generando poor_quality {split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            degraded = degrade_image(img)
            out_path = pq_dir / f"{img_path.stem}_pq.png"
            cv2.imwrite(str(out_path), degraded)


if __name__ == "__main__":
    main()
