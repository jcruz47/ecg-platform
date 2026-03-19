from app.services.study_helpers import get_latest_analysis_bundle


def get_patient_with_studies(sb, patient_id: str):
    patient_res = sb.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_res.data:
        return None

    patient = patient_res.data[0]

    studies_res = (
        sb.table("ecg_studies")
        .select("*")
        .eq("patient_id", patient_id)
        .order("study_datetime", desc=True)
        .execute()
    )

    studies = studies_res.data

    enriched_studies = []
    for study in studies:
        bundle = get_latest_analysis_bundle(sb, study["id"])
        enriched_studies.append({
            "study": study,
            "analysis": bundle["analysis"],
            "metrics": bundle["metrics"],
            "findings": bundle["findings"],
            "report": bundle["report"],
            "ai": bundle["ai"],
        })

    return {
        "patient": patient,
        "studies": enriched_studies,
    }