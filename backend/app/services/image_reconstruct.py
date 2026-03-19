import cv2
import numpy as np


def _split_contiguous(indices: np.ndarray):
    if len(indices) == 0:
        return []

    groups = []
    start = indices[0]
    prev = indices[0]

    for v in indices[1:]:
        if v == prev + 1:
            prev = v
        else:
            groups.append(np.arange(start, prev + 1))
            start = v
            prev = v

    groups.append(np.arange(start, prev + 1))
    return groups


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()

    window = int(window)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def reconstruct_signal_from_mask(mask: np.ndarray, estimated_fs: float = 500.0):
    if mask.ndim != 2:
        raise ValueError("La máscara debe ser 2D")

    # binarizar
    mask_bin = (mask > 0).astype(np.uint8) * 255
    h, w = mask_bin.shape

    # limpiar ruido pequeño y unir pequeños cortes del trazo
    kernel_small = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_small)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_small)

    ys = []
    prev_y = None

    # tolerancia máxima de salto entre columnas
    max_jump_px = max(6, int(h * 0.10))

    for x in range(w):
        col = mask_clean[:, x]
        indices = np.where(col > 0)[0]

        if len(indices) == 0:
            ys.append(np.nan)
            continue

        groups = _split_contiguous(indices)

        # centro de cada segmento vertical detectado
        centers = np.array([float(np.mean(g)) for g in groups], dtype=np.float32)
        lengths = np.array([len(g) for g in groups], dtype=np.int32)

        if prev_y is None:
            # al inicio: elegir el segmento más largo
            chosen_y = float(centers[np.argmax(lengths)])
        else:
            # luego: elegir el centro más cercano al valor previo
            dists = np.abs(centers - prev_y)
            best_idx = int(np.argmin(dists))

            # si el salto es demasiado grande, mejor marcar como hueco
            if dists[best_idx] > max_jump_px:
                ys.append(np.nan)
                continue

            chosen_y = float(centers[best_idx])

        ys.append(chosen_y)
        prev_y = chosen_y

    ys = np.asarray(ys, dtype=np.float32)

    if np.isnan(ys).all():
        raise ValueError("No se pudo reconstruir señal: máscara vacía o sin trazo útil")

    valid = ~np.isnan(ys)
    valid_ratio = float(np.mean(valid))

    if valid.sum() < max(10, int(0.05 * w)):
        raise ValueError("No se pudo reconstruir señal: muy pocos puntos válidos")

    xs = np.arange(w, dtype=np.float32)
    ys_interp = np.interp(xs, xs[valid], ys[valid]).astype(np.float32)

    # invertir eje Y: arriba = amplitud positiva
    signal = (h / 2.0) - ys_interp

    # eliminar tendencia lenta
    trend_window = max(31, (w // 12) | 1)  # impar
    trend = _moving_average(signal, trend_window)
    signal_detrended = signal - trend

    # suavizado final para evitar picos falsos
    smooth_window = max(5, (w // 80) | 1)  # impar
    signal_smooth = _moving_average(signal_detrended, smooth_window)

    # normalización robusta
    mean = float(np.mean(signal_smooth))
    std = float(np.std(signal_smooth))
    if std < 1e-8:
        raise ValueError("No se pudo reconstruir señal: varianza casi nula")

    signal_norm = (signal_smooth - mean) / std

    # quality más realista: puntos válidos + continuidad
    diffs = np.diff(ys_interp)
    continuity_penalty = float(np.mean(np.abs(diffs) > max_jump_px)) if len(diffs) > 0 else 0.0
    quality = float(max(0.0, min(1.0, valid_ratio * (1.0 - continuity_penalty))))

    return {
        "signal": signal_norm.astype(np.float32),
        "estimated_sampling_rate_hz": float(estimated_fs),
        "reconstruction_quality_score": quality,
    }