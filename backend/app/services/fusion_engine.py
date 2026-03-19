def fuse_image_and_signal_results(
    qc: dict,
    reconstruction: dict | None,
    signal_payload: dict | None,
    image_cls: dict | None,
):
    fusion_label = "review_recommended"
    fusion_confidence = 0.5

    if not qc.get("usable", False):
        fusion_label = "poor_quality"
        fusion_confidence = 0.9
        return {
            "fusion_label": fusion_label,
            "fusion_confidence": fusion_confidence,
            "reason": "Imagen de baja calidad",
        }

    img_label = image_cls["predicted_label"] if image_cls else None
    img_conf = image_cls["confidence"] if image_cls else None
    recon_q = reconstruction["reconstruction_quality_score"] if reconstruction else None

    hr = None
    if signal_payload and signal_payload.get("metrics"):
        hr = signal_payload["metrics"].get("heart_rate_bpm")

    if img_label == "normal" and recon_q is not None and recon_q > 0.8:
        fusion_label = "likely_normal_or_low_risk_visual_pattern"
        fusion_confidence = 0.75

    if img_label == "abnormal":
        fusion_label = "visual_abnormality_detected"
        fusion_confidence = max(0.6, float(img_conf or 0.6))

    if img_label == "poor_quality":
        fusion_label = "poor_quality"
        fusion_confidence = max(0.7, float(img_conf or 0.7))

    if hr is not None and (hr < 50 or hr > 100):
        fusion_label = "rhythm_attention_needed"
        fusion_confidence = max(fusion_confidence, 0.8)

    return {
        "fusion_label": fusion_label,
        "fusion_confidence": fusion_confidence,
        "reason": {
            "image_label": img_label,
            "reconstruction_quality": recon_q,
            "heart_rate_bpm": hr,
        },
    }