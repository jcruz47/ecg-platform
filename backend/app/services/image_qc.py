import cv2
import numpy as np


def evaluate_image_qc(raw_bytes: bytes) -> dict:
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Foreground = lo que no es casi blanco
    foreground_mask = gray < 245
    foreground_fraction = float(np.mean(foreground_mask) * 100.0)

    pct_very_bright = float(np.mean(gray > 250) * 100.0)
    pct_very_dark = float(np.mean(gray < 15) * 100.0)

    if np.any(foreground_mask):
        fg_pixels = gray[foreground_mask]
        foreground_mean = float(np.mean(fg_pixels))
        foreground_std = float(np.std(fg_pixels))
    else:
        foreground_mean = None
        foreground_std = None

    issues = []

    # Imagen casi vacía
    if foreground_fraction < 0.15:
        issues.append("almost_blank")

    # Desenfoque real
    if sharpness < 20:
        issues.append("blurry")

    # Oscuridad real
    if brightness < 25 and pct_very_dark > 40:
        issues.append("too_dark")

    # Bajo contraste real
    if contrast < 8 and foreground_fraction < 0.5:
        issues.append("low_contrast")

    # Sobreexposición real
    if pct_very_bright > 98 and foreground_fraction < 0.2:
        issues.append("overexposed")

    usable = len(issues) == 0

    if usable:
        quality_label = "good"
    else:
        priority = [
            "almost_blank",
            "overexposed",
            "too_dark",
            "blurry",
            "low_contrast",
        ]
        quality_label = next((label for label in priority if label in issues), issues[0])

    return {
        "usable": usable,
        "quality_label": quality_label,
        "issues": issues,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "foreground_fraction": foreground_fraction,
        "pct_very_bright": pct_very_bright,
        "pct_very_dark": pct_very_dark,
        "foreground_mean": foreground_mean,
        "foreground_std": foreground_std,
        "shape": list(img.shape),
    }