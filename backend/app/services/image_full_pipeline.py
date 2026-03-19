import uuid
import cv2
import numpy as np

from app.core.config import settings
from app.services.image_qc import evaluate_image_qc
from app.services.image_preprocess import rectify_ecg_image, remove_grid
from app.services.image_segmentation_cnn import segment_trace_mask
from app.services.image_reconstruct import reconstruct_signal_from_mask
from app.services.signal_pipeline import analyze_signal
from app.services.image_classifier_cnn import classify_ecg_image
from app.services.fusion_engine import fuse_image_and_signal_results


def _encode_png_bytes(gray_or_bgr):
    ok, buf = cv2.imencode(".png", gray_or_bgr)
    if not ok:
        raise ValueError("No se pudo codificar imagen PNG")
    return buf.tobytes()


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _safe_reconstruct_and_analyze(mask: np.ndarray, estimated_fs: float):
    try:
        reconstruction = reconstruct_signal_from_mask(mask, estimated_fs=estimated_fs)
    except Exception:
        return None, None

    try:
        signal_payload = analyze_signal(
            reconstruction["signal"],
            float(reconstruction["estimated_sampling_rate_hz"])
        )
    except Exception:
        signal_payload = {
            "error": "Falló el análisis de la señal reconstruida.",
            "metrics": None,
            "summary": {
                "samples": int(len(reconstruction["signal"])),
                "sampling_rate": float(reconstruction["estimated_sampling_rate_hz"]),
                "rpeaks_detected": 0,
                "duration_seconds": float(
                    len(reconstruction["signal"]) / reconstruction["estimated_sampling_rate_hz"]
                ) if reconstruction["estimated_sampling_rate_hz"] > 0 else 0.0,
            },
            "findings": [],
            "qc_score": None,
        }

    return reconstruction, signal_payload


def _min_quality_threshold(duration_seconds: float) -> float:
    """
    Umbral mínimo de calidad de reconstrucción.
    Para tiras largas exigimos más calidad.
    """
    if duration_seconds >= 8.0:
        return 0.60
    return 0.40


def _variant_bonus(name: str) -> float:
    """
    Pequeña preferencia por variantes que ya demostraron funcionar mejor
    en ECG limpio/fotografiado.
    """
    if name == "cleaned":
        return 0.05
    if name == "warped":
        return 0.01
    return 0.0


def _score_candidate(name: str, reconstruction: dict | None, signal_payload: dict | None) -> float:
    if reconstruction is None:
        return -1.0

    rq = float(reconstruction.get("reconstruction_quality_score") or 0.0)

    summary = (signal_payload or {}).get("summary") or {}
    metrics = (signal_payload or {}).get("metrics") or {}

    duration = float(summary.get("duration_seconds") or 0.0)
    rpeaks = int(summary.get("rpeaks_detected") or 0)
    hr = metrics.get("heart_rate_bpm")
    signal_error = (signal_payload or {}).get("error")

    min_rq = _min_quality_threshold(duration)

    # Si la reconstrucción es demasiado mala, descartarla.
    if rq < min_rq:
        return -1.0

    score = rq

    # Bonus si el análisis de señal no tronó
    if signal_payload is not None and not signal_error:
        score += 0.25
    elif signal_payload is not None and signal_error:
        score -= 0.10

    # Un poco de continuidad fisiológica mínima
    if duration >= 1.5 and rpeaks >= 2:
        score += 0.10

    # Rango plausible de FC
    if hr is not None:
        hr = float(hr)
        if 45 <= hr <= 160:
            score += 0.20
        elif 35 <= hr <= 180:
            score += 0.10
        else:
            score -= 0.10

    # Preferencia ligera por cleaned
    score += _variant_bonus(name)

    return score


def run_full_image_pipeline(sb, study_id: str, raw_bytes: bytes):
    qc = evaluate_image_qc(raw_bytes)

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    original_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if original_img is None:
        raise ValueError("No se pudo leer la imagen original")

    original, warped = rectify_ecg_image(raw_bytes)
    cleaned = remove_grid(warped)

    original = _ensure_bgr(original)
    warped = _ensure_bgr(warped)
    cleaned = _ensure_bgr(cleaned)

    assumed_duration_seconds = float(
        getattr(settings, "ECG_IMAGE_ASSUMED_DURATION_SECONDS", 2.5)
    )

    candidates = [
        ("original", original),
        ("warped", warped),
        ("cleaned", cleaned),
    ]

    candidate_results = []
    best_result = None
    best_score = -1.0

    for name, img in candidates:
        try:
            img_png = _encode_png_bytes(img)
            mask = segment_trace_mask(img_png)

            estimated_fs = float(mask.shape[1]) / assumed_duration_seconds

            reconstruction = None
            signal_payload = None

            if qc["usable"]:
                reconstruction, signal_payload = _safe_reconstruct_and_analyze(
                    mask,
                    estimated_fs=estimated_fs
                )

            score = _score_candidate(name, reconstruction, signal_payload)

            result = {
                "name": name,
                "image": img,
                "mask": mask,
                "reconstruction": reconstruction,
                "signal_payload": signal_payload,
                "score": score,
            }
            candidate_results.append(result)

            if score > best_score:
                best_score = score
                best_result = result

        except Exception:
            continue

    if best_result is None or best_score < 0:
        raise ValueError("No se pudo generar ninguna reconstrucción confiable desde la imagen")

    chosen_name = best_result["name"]
    chosen_mask = best_result["mask"]
    reconstruction = best_result["reconstruction"]
    signal_payload = best_result["signal_payload"]

    # Clasificación SIEMPRE sobre la imagen original
    image_cls = classify_ecg_image(raw_bytes)

    fusion = fuse_image_and_signal_results(qc, reconstruction, signal_payload, image_cls)
    fusion["preprocess_variant"] = chosen_name

    # Guardar máscara elegida
    mask_path = f"{study_id}/{uuid.uuid4()}_{chosen_name}_mask.png"
    sb.storage.from_("ecg-image-masks").upload(
        path=mask_path,
        file=_encode_png_bytes(chosen_mask),
        file_options={"content-type": "image/png", "upsert": "false"}
    )

    # Guardar señal reconstruida elegida
    derived_signal_path = None
    if reconstruction is not None:
        import tempfile
        import os

        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(tmp.name, reconstruction["signal"])
        tmp.close()

        derived_signal_path = f"{study_id}/{uuid.uuid4()}_{chosen_name}_reconstructed.npy"
        with open(tmp.name, "rb") as f:
            sb.storage.from_("ecg-image-derived").upload(
                path=derived_signal_path,
                file=f,
                file_options={"content-type": "application/octet-stream", "upsert": "false"}
            )
        os.unlink(tmp.name)

    return {
        "qc": qc,
        "mask_path": mask_path,
        "reconstruction": reconstruction,
        "derived_signal_path": derived_signal_path,
        "signal_payload": signal_payload,
        "image_classification": image_cls,
        "fusion": fusion,
        "selected_preprocess_variant": chosen_name,
        "candidate_scores": [
            {
                "name": r["name"],
                "score": r["score"],
                "reconstruction_quality_score": (
                    r["reconstruction"]["reconstruction_quality_score"]
                    if r["reconstruction"] is not None else None
                ),
                "signal_error": (
                    r["signal_payload"].get("error")
                    if r["signal_payload"] is not None else None
                ),
            }
            for r in candidate_results
        ],
    }