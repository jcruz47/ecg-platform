from fastapi import APIRouter, HTTPException, Query
from app.db.supabase_client import get_supabase
from app.schemas.study import StudyCreate
from app.services.study_helpers import get_latest_analysis_bundle, get_signal_preview_for_study

router = APIRouter()


@router.post("")
def create_study(payload: StudyCreate):
    if payload.source_type not in {"signal", "image"}:
        raise HTTPException(status_code=400, detail="source_type inválido")

    sb = get_supabase()
    result = sb.table("ecg_studies").insert(payload.model_dump()).execute()
    return result.data[0]


@router.get("")
def list_studies(patient_id: str | None = None):
    sb = get_supabase()
    query = sb.table("ecg_studies").select("*").order("created_at", desc=True)
    if patient_id:
        query = query.eq("patient_id", patient_id)
    result = query.execute()
    return result.data


@router.get("/{study_id}")
def get_study(study_id: str):
    sb = get_supabase()
    result = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    return result.data[0]


@router.get("/{study_id}/full-result")
def get_study_full_result(study_id: str):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    study = study_res.data[0]

    patient_res = sb.table("patients").select("*").eq("id", study["patient_id"]).execute()
    patient = patient_res.data[0] if patient_res.data else None

    bundle = get_latest_analysis_bundle(sb, study_id)

    chat_res = (
        sb.table("ai_chat_messages")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at")
        .execute()
    )

    return {
        "patient": patient,
        "study": study,
        "analysis": bundle["analysis"],
        "metrics": bundle["metrics"],
        "findings": bundle["findings"],
        "report": bundle["report"],
        "ai": bundle["ai"],
        "chat_messages": chat_res.data,
    }


@router.get("/{study_id}/signal-preview")
def get_study_signal_preview(
    study_id: str,
    max_points: int = Query(default=1200, ge=100, le=5000),
):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    preview = get_signal_preview_for_study(sb, study_id, max_points=max_points)
    return {"preview": preview}


@router.get("/{study_id}/image-results")
def get_study_image_results(study_id: str):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    qc = sb.table("ecg_image_qc").select("*").eq("study_id", study_id).order("created_at", desc=True).limit(1).execute().data
    seg = sb.table("ecg_image_segmentation").select("*").eq("study_id", study_id).order("created_at", desc=True).limit(1).execute().data
    rec = sb.table("ecg_image_reconstruction").select("*").eq("study_id", study_id).order("created_at", desc=True).limit(1).execute().data
    cls = sb.table("ecg_image_classification").select("*").eq("study_id", study_id).order("created_at", desc=True).limit(1).execute().data
    fus = sb.table("ecg_fusion_results").select("*").eq("study_id", study_id).order("created_at", desc=True).limit(1).execute().data

    return {
        "qc": qc[0] if qc else None,
        "segmentation": seg[0] if seg else None,
        "reconstruction": rec[0] if rec else None,
        "classification": cls[0] if cls else None,
        "fusion": fus[0] if fus else None,
    }


@router.get("/{study_id}/image-classification")
def get_study_image_classification(study_id: str):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    result = (
        sb.table("ecg_image_classification")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    return result.data[0] if result.data else None