from app.services.study_helpers import get_latest_analysis_bundle
 
 
def get_patient_timeline(sb, patient_id: str):
    patient_res = sb.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_res.data:
        return None
 
    patient = patient_res.data[0]
 
    studies_res = (
        sb.table("ecg_studies")
        .select("*")
        .eq("patient_id", patient_id)
        .order("study_datetime")
        .execute()
    )
 
    studies = studies_res.data
 
    timeline = []
    for study in studies:
        bundle = get_latest_analysis_bundle(sb, study["id"])
        metrics = bundle.get("metrics") or {}
 
        timeline.append({
            "study_id": study["id"],
            "study_datetime": study.get("study_datetime"),
            "source_type": study.get("source_type"),
            "status": study.get("status"),
            "metrics": {
                "heart_rate_bpm": metrics.get("heart_rate_bpm"),
                "rr_mean_ms": metrics.get("rr_mean_ms"),
                "sdnn_ms": metrics.get("sdnn_ms"),
                "rmssd_ms": metrics.get("rmssd_ms"),
                "pnn50": metrics.get("pnn50"),
                "signal_quality_score": metrics.get("signal_quality_score"),
            },
            "findings_count": len(bundle.get("findings") or []),
            "has_ai_interpretation": bundle.get("ai") is not None,
        })
 
    return {
        "patient": patient,
        "timeline": timeline,
        "summary": {
            "study_count": len(timeline),
            "first_study_datetime": timeline[0]["study_datetime"] if timeline else None,
            "last_study_datetime": timeline[-1]["study_datetime"] if timeline else None,
        }
    }