import numpy as np

from app.services.signal_pipeline import load_signal_from_bytes, normalize_signal


def get_latest_analysis_bundle(sb, study_id: str) -> dict:
    analysis_res = (
        sb.table("ecg_analysis")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not analysis_res.data:
        return {
            "analysis": None,
            "metrics": None,
            "findings": [],
            "report": None,
            "ai": None,
        }

    analysis = analysis_res.data[0]
    analysis_id = analysis["id"]

    metrics_res = (
        sb.table("ecg_metrics")
        .select("*")
        .eq("analysis_id", analysis_id)
        .execute()
    )
    findings_res = (
        sb.table("ecg_findings")
        .select("*")
        .eq("analysis_id", analysis_id)
        .execute()
    )
    report_res = (
        sb.table("reports")
        .select("*")
        .eq("analysis_id", analysis_id)
        .execute()
    )
    ai_res = (
        sb.table("ai_interpretations")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    return {
        "analysis": analysis,
        "metrics": metrics_res.data[0] if metrics_res.data else None,
        "findings": findings_res.data,
        "report": report_res.data[0] if report_res.data else None,
        "ai": ai_res.data[0] if ai_res.data else None,
    }


def get_signal_preview_for_study(sb, study_id: str, max_points: int = 1200):
    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        return None

    study = study_res.data[0]
    if study["source_type"] != "signal":
        return None

    file_res = (
        sb.table("ecg_signal_files")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not file_res.data:
        return None

    file_row = file_res.data[0]
    raw_bytes = sb.storage.from_(file_row["bucket_name"]).download(file_row["object_path"])

    signal = load_signal_from_bytes(
        raw_bytes=raw_bytes,
        filename=file_row.get("object_path") or file_row.get("file_format"),
    )

    signal = np.asarray(signal, dtype=float).squeeze()

    if signal.ndim > 1:
        signal = signal[:, 0]

    if len(signal) == 0:
        return None

    fs = float(study.get("sampling_rate_hz") or 360)
    signal = normalize_signal(signal)

    point_count = min(len(signal), max_points)
    indices = np.linspace(0, len(signal) - 1, point_count).astype(int)

    preview_values = np.round(signal[indices], 5).tolist()
    preview_times = np.round(indices / fs, 5).tolist()

    return {
        "sampling_rate_hz": fs,
        "count_original": int(len(signal)),
        "count_displayed": int(point_count),
        "duration_seconds": round(len(signal) / fs, 3),
        "times_sec": preview_times,
        "values": preview_values,
    }