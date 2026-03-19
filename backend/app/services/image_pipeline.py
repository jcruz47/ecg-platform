import cv2
import numpy as np


def analyze_image(raw_bytes: bytes) -> dict:
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("No se pudo decodificar la imagen")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_intensity = float(np.mean(gray))

    findings = []

    if sharpness < 50:
        findings.append({
            "label": "Imagen con posible desenfoque",
            "severity": "low",
            "confidence": 0.70,
            "details_json": {"sharpness": sharpness}
        })

    summary = {
        "width_px": int(img.shape[1]),
        "height_px": int(img.shape[0]),
        "mean_intensity": mean_intensity,
        "note": "MVP: control de calidad de imagen. La reconstrucción del trazo se implementa después."
    }

    return {
        "summary": summary,
        "metrics": {
            "signal_quality_score": sharpness
        },
        "findings": findings,
        "qc_score": sharpness
    }