def metric_diff(a, b):
    if a is None or b is None:
        return None
    try:
        return round(float(b) - float(a), 4)
    except Exception:
        return None


def compare_studies_payload(study_a_bundle: dict, study_b_bundle: dict):
    metrics_a = study_a_bundle.get("metrics") or {}
    metrics_b = study_b_bundle.get("metrics") or {}

    comparison = {
        "heart_rate_bpm": {
            "a": metrics_a.get("heart_rate_bpm"),
            "b": metrics_b.get("heart_rate_bpm"),
            "delta": metric_diff(metrics_a.get("heart_rate_bpm"), metrics_b.get("heart_rate_bpm")),
        },
        "rr_mean_ms": {
            "a": metrics_a.get("rr_mean_ms"),
            "b": metrics_b.get("rr_mean_ms"),
            "delta": metric_diff(metrics_a.get("rr_mean_ms"), metrics_b.get("rr_mean_ms")),
        },
        "sdnn_ms": {
            "a": metrics_a.get("sdnn_ms"),
            "b": metrics_b.get("sdnn_ms"),
            "delta": metric_diff(metrics_a.get("sdnn_ms"), metrics_b.get("sdnn_ms")),
        },
        "rmssd_ms": {
            "a": metrics_a.get("rmssd_ms"),
            "b": metrics_b.get("rmssd_ms"),
            "delta": metric_diff(metrics_a.get("rmssd_ms"), metrics_b.get("rmssd_ms")),
        },
        "pnn50": {
            "a": metrics_a.get("pnn50"),
            "b": metrics_b.get("pnn50"),
            "delta": metric_diff(metrics_a.get("pnn50"), metrics_b.get("pnn50")),
        },
        "signal_quality_score": {
            "a": metrics_a.get("signal_quality_score"),
            "b": metrics_b.get("signal_quality_score"),
            "delta": metric_diff(metrics_a.get("signal_quality_score"), metrics_b.get("signal_quality_score")),
        },
    }

    return comparison