from fastapi import APIRouter, HTTPException, Query
from app.db.supabase_client import get_supabase
from app.schemas.patient import PatientCreate
from app.services.patient_helpers import get_patient_with_studies
from app.services.study_helpers import get_latest_analysis_bundle
from app.services.compare_helpers import compare_studies_payload
from app.services.timeline_helpers import get_patient_timeline
 
router = APIRouter()
 
 
@router.post("")
def create_patient(payload: PatientCreate):
    sb = get_supabase()
    result = sb.table("patients").insert(payload.model_dump(mode="json")).execute()
    return result.data[0]
 
 
@router.get("")
def list_patients():
    sb = get_supabase()
    result = sb.table("patients").select("*").order("created_at", desc=True).execute()
    return result.data
 
 
@router.get("/{patient_id}")
def get_patient(patient_id: str):
    sb = get_supabase()
    result = sb.table("patients").select("*").eq("id", patient_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return result.data[0]
 
 
@router.get("/{patient_id}/studies")
def get_patient_studies(patient_id: str):
    sb = get_supabase()
 
    patient_res = sb.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_res.data:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
 
    result = (
        sb.table("ecg_studies")
        .select("*")
        .eq("patient_id", patient_id)
        .order("study_datetime", desc=True)
        .execute()
    )
    return result.data
 
 
@router.get("/{patient_id}/full")
def get_patient_full(patient_id: str):
    sb = get_supabase()
    payload = get_patient_with_studies(sb, patient_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return payload
 
 
@router.get("/{patient_id}/timeline")
def get_patient_timeline_route(patient_id: str):
    sb = get_supabase()
    payload = get_patient_timeline(sb, patient_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return payload
 
 
@router.get("/{patient_id}/compare")
def compare_patient_studies(
    patient_id: str,
    study_a: str = Query(...),
    study_b: str = Query(...)
):
    sb = get_supabase()
 
    patient_res = sb.table("patients").select("*").eq("id", patient_id).execute()
    if not patient_res.data:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
 
    study_a_res = sb.table("ecg_studies").select("*").eq("id", study_a).eq("patient_id", patient_id).execute()
    study_b_res = sb.table("ecg_studies").select("*").eq("id", study_b).eq("patient_id", patient_id).execute()
 
    if not study_a_res.data or not study_b_res.data:
        raise HTTPException(status_code=404, detail="Uno o ambos estudios no pertenecen al paciente")
 
    study_a_row = study_a_res.data[0]
    study_b_row = study_b_res.data[0]
 
    bundle_a = get_latest_analysis_bundle(sb, study_a)
    bundle_b = get_latest_analysis_bundle(sb, study_b)
 
    comparison = compare_studies_payload(bundle_a, bundle_b)
 
    return {
        "patient": patient_res.data[0],
        "study_a": {
            "study": study_a_row,
            "analysis": bundle_a["analysis"],
            "metrics": bundle_a["metrics"],
            "findings": bundle_a["findings"],
            "report": bundle_a["report"],
            "ai": bundle_a["ai"],
        },
        "study_b": {
            "study": study_b_row,
            "analysis": bundle_b["analysis"],
            "metrics": bundle_b["metrics"],
            "findings": bundle_b["findings"],
            "report": bundle_b["report"],
            "ai": bundle_b["ai"],
        },
        "comparison": comparison,
    }