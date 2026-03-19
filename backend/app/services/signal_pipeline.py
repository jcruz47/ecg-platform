import csv
import io
import json
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _extract_extension(filename):
    if not filename:
        return ""

    filename = str(filename).strip().lower()
    if "." in filename:
        return filename.rsplit(".", 1)[-1]

    # Si te pasan directamente "csv", "txt", "json", etc.
    return filename


def _choose_best_signal_column(arr2d):
    arr2d = np.asarray(arr2d, dtype=np.float32)

    if arr2d.ndim != 2:
        raise ValueError("Se esperaba una matriz 2D para elegir columna de señal")

    if arr2d.shape[1] == 1:
        col = arr2d[:, 0]
        return col[np.isfinite(col)]

    best_idx = 0
    best_score = -1e18

    for j in range(arr2d.shape[1]):
        col = arr2d[:, j]
        valid = col[np.isfinite(col)]

        if valid.size == 0:
            continue

        std = float(np.std(valid)) if valid.size > 1 else 0.0
        score = float(valid.size) + min(std, 10.0)

        # Penaliza columnas que parecen "time" o índice: muy monótonas
        if valid.size > 3:
            diffs = np.diff(valid)
            if diffs.size > 0:
                monotonic_ratio = float(np.mean(diffs >= 0))
                if monotonic_ratio > 0.95:
                    score -= 2.0

        if score > best_score:
            best_score = score
            best_idx = j

    col = arr2d[:, best_idx]
    return col[np.isfinite(col)]


def _extract_from_structured_array(arr):
    if not (isinstance(arr, np.ndarray) and arr.dtype.names):
        return arr

    preferred = (
        "ecg",
        "signal",
        "value",
        "values",
        "samples",
        "data",
        "lead_i",
        "lead_ii",
    )

    names_map = {name.lower(): name for name in arr.dtype.names}

    for key in preferred:
        if key in names_map:
            col = np.asarray(arr[names_map[key]], dtype=np.float32)
            col = col[np.isfinite(col)]
            if col.size > 0:
                return col

    cols = []
    for name in arr.dtype.names:
        col = np.asarray(arr[name], dtype=np.float32)
        cols.append(col)

    mat = np.column_stack(cols)
    return _choose_best_signal_column(mat)


def _manual_parse_csv(text):
    rows = []
    reader = csv.reader(io.StringIO(text))

    for row in reader:
        if not row:
            continue
        cleaned = [cell.strip() for cell in row]
        if any(cell != "" for cell in cleaned):
            rows.append(cleaned)

    if not rows:
        raise ValueError("CSV vacío")

    max_cols = max(len(r) for r in rows)
    numeric_rows = []

    for row in rows:
        vals = []
        numeric_count = 0

        for cell in row:
            if cell == "":
                vals.append(np.nan)
                continue

            try:
                vals.append(float(cell))
                numeric_count += 1
            except ValueError:
                vals.append(np.nan)

        if len(vals) < max_cols:
            vals.extend([np.nan] * (max_cols - len(vals)))

        # Ignorar filas que no tengan ningún valor numérico
        if numeric_count > 0:
            numeric_rows.append(vals)

    if not numeric_rows:
        raise ValueError("CSV sin datos numéricos válidos")

    arr = np.asarray(numeric_rows, dtype=np.float32)

    # Quitar columnas totalmente vacías
    valid_cols = np.where(np.sum(np.isfinite(arr), axis=0) > 0)[0]
    if valid_cols.size == 0:
        raise ValueError("CSV sin columnas numéricas válidas")

    arr = arr[:, valid_cols]

    if arr.ndim == 2 and arr.shape[1] >= 1:
        return _choose_best_signal_column(arr)

    return arr


def _load_csv_text(text):
    # 1) CSV numérico puro
    try:
        return np.loadtxt(io.StringIO(text), delimiter=",")
    except Exception:
        pass

    # 2) CSV con encabezado
    try:
        arr = np.genfromtxt(
            io.StringIO(text),
            delimiter=",",
            names=True,
            dtype=np.float32,
            encoding=None,
        )
        arr = _extract_from_structured_array(arr)
        return arr
    except Exception:
        pass

    # 3) Parse manual robusto
    return _manual_parse_csv(text)


def _load_txt_text(text):
    # 1) TXT numérico separado por espacios/tabulaciones
    try:
        return np.loadtxt(io.StringIO(text))
    except Exception:
        pass

    # 2) TXT separado por comas
    try:
        return np.loadtxt(io.StringIO(text), delimiter=",")
    except Exception:
        pass

    # 3) Intentar con encabezados
    try:
        arr = np.genfromtxt(
            io.StringIO(text),
            delimiter=None,
            names=True,
            dtype=np.float32,
            encoding=None,
        )
        arr = _extract_from_structured_array(arr)
        return arr
    except Exception:
        pass

    # 4) Extraer valores numéricos línea por línea
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            pass

    if values:
        return np.asarray(values, dtype=np.float32)

    # 5) Último intento: todos los tokens numéricos
    data = np.fromstring(text.replace(",", " "), sep=" ")
    if data.size > 0:
        return data.astype(np.float32)

    raise ValueError("TXT sin datos numéricos válidos")


def _extract_signal_from_json(obj):
    if isinstance(obj, dict):
        for key in ("ecg", "signal", "samples", "data", "values"):
            if key in obj:
                return np.asarray(obj[key], dtype=np.float32)
        raise ValueError("JSON sin clave reconocible de señal")

    return np.asarray(obj, dtype=np.float32)


def _finalize_signal_array(arr):
    if arr is None:
        raise ValueError("Formato de señal no soportado o no reconocido")

    arr = _extract_from_structured_array(arr)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = _choose_best_signal_column(arr)

    if arr.ndim != 1:
        raise ValueError("La señal debe ser un vector 1D")

    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        raise ValueError("La señal cargada está vacía")

    return arr.astype(np.float32)


def load_signal_from_bytes(raw_bytes, filename=None):
    """
    Carga una señal 1D desde bytes.
    Soporta principalmente:
    - .npy
    - .npz
    - .csv
    - .txt
    - .json

    También tolera:
    - filename="csv" en lugar de "archivo.csv"
    - CSV con encabezados como: ecg
    - CSV con varias columnas como: time, ecg
    """
    if raw_bytes is None or len(raw_bytes) == 0:
        raise ValueError("Archivo de señal vacío")

    ext = _extract_extension(filename)
    arr = None

    try:
        if ext == "npy":
            arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)

        elif ext == "npz":
            npz = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            if len(npz.files) == 0:
                raise ValueError("Archivo NPZ sin arreglos")
            arr = npz[npz.files[0]]

        elif ext == "csv":
            text = raw_bytes.decode("utf-8", errors="ignore")
            arr = _load_csv_text(text)

        elif ext == "txt":
            text = raw_bytes.decode("utf-8", errors="ignore")
            arr = _load_txt_text(text)

        elif ext == "json":
            text = raw_bytes.decode("utf-8", errors="ignore")
            obj = json.loads(text)
            arr = _extract_signal_from_json(obj)

        else:
            # Intento 1: NPY/NPZ sin extensión clara
            try:
                loaded = np.load(io.BytesIO(raw_bytes), allow_pickle=False)

                if isinstance(loaded, np.lib.npyio.NpzFile):
                    if len(loaded.files) == 0:
                        raise ValueError("Archivo NPZ sin arreglos")
                    arr = loaded[loaded.files[0]]
                else:
                    arr = loaded
            except Exception:
                pass

            # Intento 2: texto CSV/TXT
            if arr is None:
                try:
                    text = raw_bytes.decode("utf-8", errors="ignore")
                    try:
                        arr = _load_csv_text(text)
                    except Exception:
                        arr = _load_txt_text(text)
                except Exception:
                    pass

            # Intento 3: JSON
            if arr is None:
                try:
                    text = raw_bytes.decode("utf-8", errors="ignore")
                    obj = json.loads(text)
                    arr = _extract_signal_from_json(obj)
                except Exception:
                    pass

    except Exception as e:
        raise ValueError(f"No se pudo cargar la señal: {e}")

    return _finalize_signal_array(arr)


def normalize_signal(signal, method="zscore"):
    signal = np.asarray(signal, dtype=np.float32).flatten()

    if len(signal) == 0:
        raise ValueError("La señal está vacía")

    if method == "zscore":
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        if std < 1e-8:
            return (signal - mean).astype(np.float32)
        return ((signal - mean) / std).astype(np.float32)

    if method == "minmax":
        s_min = float(np.min(signal))
        s_max = float(np.max(signal))
        if abs(s_max - s_min) < 1e-8:
            return np.zeros_like(signal, dtype=np.float32)
        return ((signal - s_min) / (s_max - s_min)).astype(np.float32)

    raise ValueError(f"Método de normalización no soportado: {method}")


def bandpass_filter(signal, sampling_rate, low=0.5, high=20.0, order=3):
    signal = np.asarray(signal, dtype=np.float32).flatten()

    if len(signal) == 0:
        raise ValueError("La señal está vacía")

    if sampling_rate <= 0:
        raise ValueError("Frecuencia de muestreo inválida")

    nyq = 0.5 * float(sampling_rate)
    if nyq <= 0:
        raise ValueError("Frecuencia de Nyquist inválida")

    low = max(0.01, float(low))
    high = min(float(high), nyq * 0.95)

    if low >= high:
        low = max(0.01, nyq * 0.02)
        high = min(nyq * 0.80, nyq * 0.95)

    if low >= high:
        return signal

    if len(signal) < max(15, order * 6):
        return signal

    wn = [low / nyq, high / nyq]

    try:
        b, a = butter(order, wn, btype="band")
        return filtfilt(b, a, signal)
    except ValueError:
        return signal


def _score_peak_series(peaks, duration_seconds):
    if len(peaks) == 0:
        return -1.0

    bpm = (len(peaks) / max(duration_seconds, 1e-6)) * 60.0

    if bpm < 35 or bpm > 180:
        return -1.0

    return float(len(peaks))


def detect_rpeaks(signal, sampling_rate):
    signal = np.asarray(signal, dtype=np.float32).flatten()

    if len(signal) < 3:
        return np.array([], dtype=int)

    std = float(np.std(signal))
    if std < 1e-8:
        return np.array([], dtype=int)

    norm = (signal - np.mean(signal)) / (std + 1e-8)
    duration_seconds = len(norm) / float(sampling_rate)

    min_distance = max(1, int(0.45 * float(sampling_rate)))
    prominence = 0.45
    height = 0.45

    pos_peaks, _ = find_peaks(
        norm,
        distance=min_distance,
        prominence=prominence,
        height=height,
    )

    neg_peaks, _ = find_peaks(
        -norm,
        distance=min_distance,
        prominence=prominence,
        height=height,
    )

    pos_score = _score_peak_series(pos_peaks, duration_seconds)
    neg_score = _score_peak_series(neg_peaks, duration_seconds)

    if pos_score < 0 and neg_score < 0:
        return np.array([], dtype=int)

    peaks = pos_peaks if pos_score >= neg_score else neg_peaks
    return peaks.astype(int)


def estimate_qc_score(filtered_signal, rpeaks, duration_seconds):
    filtered_signal = np.asarray(filtered_signal, dtype=np.float32).flatten()

    if len(filtered_signal) == 0 or duration_seconds <= 0:
        return None

    amp_std = float(np.std(filtered_signal))
    amp_range = float(np.ptp(filtered_signal))

    variability_score = min(1.0, amp_std / 0.5)
    range_score = min(1.0, amp_range / 2.0)

    expected_min_peaks = max(1.0, duration_seconds * 0.6)
    peak_score = min(1.0, len(rpeaks) / expected_min_peaks)

    qc = 0.4 * peak_score + 0.3 * variability_score + 0.3 * range_score
    return float(max(0.0, min(1.0, qc)))


def analyze_signal(signal, sampling_rate):
    signal = np.asarray(signal, dtype=np.float32).flatten()
    sampling_rate = float(sampling_rate)

    if len(signal) == 0:
        return {
            "error": "La señal está vacía.",
            "metrics": None,
            "summary": {
                "samples": 0,
                "sampling_rate": sampling_rate,
                "rpeaks_detected": 0,
                "duration_seconds": 0.0,
            },
            "findings": [],
            "qc_score": None,
        }

    duration_seconds = float(len(signal) / sampling_rate) if sampling_rate > 0 else 0.0

    summary = {
        "samples": int(len(signal)),
        "sampling_rate": float(sampling_rate),
        "rpeaks_detected": 0,
        "duration_seconds": duration_seconds,
    }

    if sampling_rate < 10:
        return {
            "error": f"Frecuencia de muestreo demasiado baja ({sampling_rate:.2f} Hz).",
            "metrics": None,
            "summary": summary,
            "findings": [],
            "qc_score": None,
        }

    if duration_seconds < 1.5:
        return {
            "error": f"Señal demasiado corta ({duration_seconds:.2f}s). Se requieren al menos 1.5 segundos.",
            "metrics": None,
            "summary": summary,
            "findings": [],
            "qc_score": None,
        }

    filtered = bandpass_filter(signal, sampling_rate)
    rpeaks = detect_rpeaks(filtered, sampling_rate)

    summary["rpeaks_detected"] = int(len(rpeaks))

    if len(rpeaks) < 2:
        return {
            "error": "No se detectaron suficientes picos R para calcular métricas confiables.",
            "metrics": None,
            "summary": summary,
            "findings": [],
            "qc_score": estimate_qc_score(filtered, rpeaks, duration_seconds),
        }

    rr_ms = np.diff(rpeaks) / sampling_rate * 1000.0

    if len(rr_ms) == 0:
        return {
            "error": "No se pudieron calcular intervalos RR.",
            "metrics": None,
            "summary": summary,
            "findings": [],
            "qc_score": estimate_qc_score(filtered, rpeaks, duration_seconds),
        }

    heart_rate_bpm = float(60000.0 / np.mean(rr_ms))
    rr_mean_ms = float(np.mean(rr_ms))

    hrv_reliable = duration_seconds >= 10.0 and len(rpeaks) >= 8

    if hrv_reliable:
        sdnn_ms = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
        rr_diff = np.diff(rr_ms)
        rmssd_ms = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else 0.0
        pnn50 = float(np.mean(np.abs(rr_diff) > 50.0) * 100.0) if len(rr_diff) > 0 else 0.0
    else:
        sdnn_ms = None
        rmssd_ms = None
        pnn50 = None

    findings = []

    if duration_seconds >= 5.0 and len(rpeaks) >= 3:
        if heart_rate_bpm < 60:
            findings.append({
                "label": "Posible bradicardia",
                "severity": "low",
                "confidence": 0.60,
            })
        elif heart_rate_bpm > 100:
            findings.append({
                "label": "Posible taquicardia",
                "severity": "medium",
                "confidence": 0.60,
            })
    else:
        findings.append({
            "label": "Frecuencia cardíaca estimada con baja confiabilidad por duración corta",
            "severity": "low",
            "confidence": 0.80,
        })

    if hrv_reliable and sdnn_ms is not None and sdnn_ms > 120:
        findings.append({
            "label": "Variabilidad RR elevada",
            "severity": "low",
            "confidence": 0.55,
        })
    elif not hrv_reliable:
        findings.append({
            "label": "HRV no confiable por duración corta del registro",
            "severity": "low",
            "confidence": 0.80,
        })

    metrics = {
        "heart_rate_bpm": heart_rate_bpm,
        "rr_mean_ms": rr_mean_ms,
        "sdnn_ms": sdnn_ms,
        "rmssd_ms": rmssd_ms,
        "pnn50": pnn50,
    }

    qc_score = estimate_qc_score(filtered, rpeaks, duration_seconds)

    return {
        "error": None,
        "metrics": metrics,
        "summary": summary,
        "findings": findings,
        "qc_score": qc_score,
    }