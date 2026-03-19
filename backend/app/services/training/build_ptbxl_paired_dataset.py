import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt


PTBXL_ROOT = Path("data/ptbxl")
OUT_ROOT = Path("data/ptbxl_paired")

DB_CSV = PTBXL_ROOT / "ptbxl_database.csv"

# Usa records100 por ahora porque es lo que ya tienes descargado
RECORDS_SUBDIR = "records100"

# Configuración
LEAD_INDEX = 1          # 0=I, 1=II, 2=III, etc. Derivación II suele ser buena opción
LEAD_NAME = "II"
FS = 100                # porque estás usando records100
STRIP_SECONDS = 2.5     # tiras cortas
FULL_SECONDS = 10.0     # señal completa si existe
MAX_SAMPLES = 500       # para probar rápido; luego súbelo


def ensure_dirs():
    (OUT_ROOT / "signals").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images" / "clean").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "masks" / "clean").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)


def load_ptbxl_metadata():
    df = pd.read_csv(DB_CSV)
    return df


def record_path_from_row(row: pd.Series) -> Path:
    # Para records100, usa filename_lr
    rel = row["filename_lr"]
    return PTBXL_ROOT / rel


def classify_image_label(row: pd.Series) -> str:
    # Base simple para arrancar.
    # Luego puedes reemplazarla por una lógica clínica más fina.
    scp_codes = str(row.get("scp_codes", ""))

    # Si viene vacío, lo dejamos como abnormal por seguridad
    if not scp_codes:
        return "abnormal"

    # Heurística simple:
    if "NORM" in scp_codes:
        return "normal"
    return "abnormal"


def save_signal_npy(signal: np.ndarray, sample_id: str):
    out = OUT_ROOT / "signals" / f"{sample_id}.npy"
    np.save(out, signal.astype(np.float32))
    return out


def render_clean_image_and_mask(signal: np.ndarray, sample_id: str, duration_sec: float):
    """
    Genera:
    - imagen limpia del ECG
    - máscara binaria aproximada del trazo
    """
    width_px = 1200
    height_px = 300
    dpi = 100

    t = np.linspace(0, duration_sec, len(signal), endpoint=False)

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(t, signal, linewidth=1.2)
    ax.axis("off")
    fig.tight_layout(pad=0)

    img_path = OUT_ROOT / "images" / "clean" / f"{sample_id}.png"
    fig.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Crear máscara sencilla a partir de la imagen renderizada
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar el trazo oscuro
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Limpiar ruido
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    mask_path = OUT_ROOT / "masks" / "clean" / f"{sample_id}.png"
    cv2.imwrite(str(mask_path), mask)

    return img_path, mask_path


def estimate_rpeaks_simple(signal: np.ndarray, fs: int):
    """
    Ground truth inicial simple para arrancar.
    Luego puedes reemplazar esto por un detector mejor.
    """
    from scipy.signal import find_peaks

    sig = signal.astype(np.float32)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    peaks, _ = find_peaks(
        sig,
        distance=max(1, int(0.35 * fs)),
        prominence=0.5,
    )
    return peaks.astype(int)


def compute_metrics_from_rpeaks(rpeaks: np.ndarray, fs: int, duration_sec: float):
    if len(rpeaks) < 2:
        return {
            "heart_rate_bpm": None,
            "rr_mean_ms": None,
            "sdnn_ms": None,
            "rmssd_ms": None,
            "pnn50": None,
            "hrv_reliable": False,
        }

    rr_ms = np.diff(rpeaks) / fs * 1000.0
    heart_rate_bpm = float(60000.0 / np.mean(rr_ms))
    rr_mean_ms = float(np.mean(rr_ms))

    hrv_reliable = duration_sec >= 10.0 and len(rpeaks) >= 8

    if hrv_reliable:
        sdnn_ms = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
        rr_diff = np.diff(rr_ms)
        rmssd_ms = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else 0.0
        pnn50 = float(np.mean(np.abs(rr_diff) > 50.0) * 100.0) if len(rr_diff) > 0 else 0.0
    else:
        sdnn_ms = None
        rmssd_ms = None
        pnn50 = None

    return {
        "heart_rate_bpm": heart_rate_bpm,
        "rr_mean_ms": rr_mean_ms,
        "sdnn_ms": sdnn_ms,
        "rmssd_ms": rmssd_ms,
        "pnn50": pnn50,
        "hrv_reliable": hrv_reliable,
    }


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build():
    ensure_dirs()
    df = load_ptbxl_metadata()

    manifest_rows = []

    subset = df.head(MAX_SAMPLES)

    for _, row in subset.iterrows():
        record_id = str(row["ecg_id"]).zfill(5)
        rec_path = record_path_from_row(row)

        try:
            signal, meta = wfdb.rdsamp(str(rec_path))
        except Exception as e:
            print(f"[WARN] No se pudo leer {rec_path}: {e}")
            continue

        if signal.ndim != 2 or signal.shape[1] <= LEAD_INDEX:
            print(f"[WARN] Señal inválida para {record_id}")
            continue

        lead_signal = signal[:, LEAD_INDEX].astype(np.float32)

        total_seconds = len(lead_signal) / FS
        if total_seconds < STRIP_SECONDS:
            print(f"[WARN] Señal demasiado corta en {record_id}")
            continue

        # 1) guardar full 10s si alcanza
        if total_seconds >= FULL_SECONDS:
            sample_id = f"{record_id}_full"
            sig_full = lead_signal[: int(FULL_SECONDS * FS)]

            signal_path = save_signal_npy(sig_full, sample_id)
            img_path, mask_path = render_clean_image_and_mask(sig_full, sample_id, FULL_SECONDS)

            rpeaks = estimate_rpeaks_simple(sig_full, FS)
            metrics = compute_metrics_from_rpeaks(rpeaks, FS, FULL_SECONDS)

            rpeaks_path = OUT_ROOT / "metadata" / f"{sample_id}_rpeaks.json"
            metrics_path = OUT_ROOT / "metadata" / f"{sample_id}_metrics.json"

            save_json(
                {
                    "sample_id": sample_id,
                    "fs_gt": FS,
                    "rpeaks_samples": rpeaks.tolist(),
                    "rpeaks_seconds": (rpeaks / FS).tolist(),
                },
                rpeaks_path,
            )

            save_json(
                {
                    "sample_id": sample_id,
                    **metrics,
                    "duration_sec": FULL_SECONDS,
                },
                metrics_path,
            )

            manifest_rows.append({
                "sample_id": sample_id,
                "record_id": record_id,
                "split": row.get("strat_fold", 0),
                "lead_name": LEAD_NAME,
                "duration_sec": FULL_SECONDS,
                "fs_gt": FS,
                "signal_path": str(signal_path),
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "rpeaks_path": str(rpeaks_path),
                "metrics_path": str(metrics_path),
                "image_label": classify_image_label(row),
                "usable": 1,
                "variant": "clean",
            })

        # 2) guardar strips de 2.5 s
        strip_len = int(STRIP_SECONDS * FS)
        n_strips = len(lead_signal) // strip_len

        for s in range(n_strips):
            start = s * strip_len
            end = start + strip_len
            sig_strip = lead_signal[start:end]
            if len(sig_strip) != strip_len:
                continue

            sample_id = f"{record_id}_s{s}"

            signal_path = save_signal_npy(sig_strip, sample_id)
            img_path, mask_path = render_clean_image_and_mask(sig_strip, sample_id, STRIP_SECONDS)

            rpeaks = estimate_rpeaks_simple(sig_strip, FS)
            metrics = compute_metrics_from_rpeaks(rpeaks, FS, STRIP_SECONDS)

            rpeaks_path = OUT_ROOT / "metadata" / f"{sample_id}_rpeaks.json"
            metrics_path = OUT_ROOT / "metadata" / f"{sample_id}_metrics.json"

            save_json(
                {
                    "sample_id": sample_id,
                    "fs_gt": FS,
                    "rpeaks_samples": rpeaks.tolist(),
                    "rpeaks_seconds": (rpeaks / FS).tolist(),
                },
                rpeaks_path,
            )

            save_json(
                {
                    "sample_id": sample_id,
                    **metrics,
                    "duration_sec": STRIP_SECONDS,
                },
                metrics_path,
            )

            manifest_rows.append({
                "sample_id": sample_id,
                "record_id": record_id,
                "split": row.get("strat_fold", 0),
                "lead_name": LEAD_NAME,
                "duration_sec": STRIP_SECONDS,
                "fs_gt": FS,
                "signal_path": str(signal_path),
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "rpeaks_path": str(rpeaks_path),
                "metrics_path": str(metrics_path),
                "image_label": classify_image_label(row),
                "usable": 1,
                "variant": "clean",
            })

        print(f"[OK] {record_id}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_ROOT / "manifests" / "studies_master.csv", index=False)

    # split simple por strat_fold
    train = manifest[manifest["split"].astype(int) <= 8]
    val = manifest[manifest["split"].astype(int) == 9]
    test = manifest[manifest["split"].astype(int) == 10]

    train.to_csv(OUT_ROOT / "manifests" / "train.csv", index=False)
    val.to_csv(OUT_ROOT / "manifests" / "val.csv", index=False)
    test.to_csv(OUT_ROOT / "manifests" / "test.csv", index=False)

    print("\nDataset pareado generado en:", OUT_ROOT.resolve())
    print("Total muestras:", len(manifest))
    print("Train:", len(train), "Val:", len(val), "Test:", len(test))


if __name__ == "__main__":
    build()