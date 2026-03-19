from pathlib import Path
import cv2
import numpy as np

IMAGES_DIR = Path("data/image_training/segmentation/images")
MASKS_DIR = Path("data/image_training/segmentation/masks")
MIN_COMPONENT_AREA = 40  # sube o baja según cuánto ruido salga


def build_mask(image_bgr: np.ndarray) -> np.ndarray:
    # Gris
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Mejora de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Resaltar trazos oscuros del ECG sobre fondo claro
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # Suavizado ligero
    blur = cv2.GaussianBlur(blackhat, (3, 3), 0)

    # Umbral automático
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Refinar con morfología
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Quitar componentes pequeñas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == label] = 255

    return cleaned


def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"No existe carpeta: {IMAGES_DIR}")

    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts])

    if not image_files:
        raise FileNotFoundError(f"No se encontraron imágenes en: {IMAGES_DIR}")

    total = len(image_files)
    print(f"Generando máscaras para {total} imágenes...")

    ok = 0
    failed = 0

    for i, img_path in enumerate(image_files, start=1):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError("cv2 no pudo leer la imagen")

            mask = build_mask(image)
            out_path = MASKS_DIR / img_path.name
            cv2.imwrite(str(out_path), mask)

            ok += 1
            if i % 100 == 0 or i == total:
                print(f"[{i}/{total}] máscaras generadas: {ok}, errores: {failed}")

        except Exception as e:
            failed += 1
            print(f"Error con {img_path.name}: {e}")

    print("Proceso terminado.")
    print(f"Máscaras generadas correctamente: {ok}")
    print(f"Errores: {failed}")


if __name__ == "__main__":
    main()
