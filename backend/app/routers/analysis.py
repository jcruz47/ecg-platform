from fastapi import APIRouter, HTTPException
from app.db.supabase_client import get_supabase

router = APIRouter()


def _detect_job_type(sb, study_id: str, source_type: str) -> str:
    """
    Detecta el job_type real según los archivos que existen en la DB,
    no según el source_type declarado al crear el estudio.
    """
    has_signal = sb.table("ecg_signal_files") \
        .select("id").eq("study_id", study_id) \
        .limit(1).execute().data

    has_image = sb.table("ecg_image_files") \
        .select("id").eq("study_id", study_id) \
        .limit(1).execute().data

    if has_signal and has_image:
        return "both"
    if has_signal:
        return "signal"
    if has_image:
        return "image"

    # Ningún archivo subido — falla temprano con mensaje claro
    raise HTTPException(
        status_code=422,
        detail=(
            "No se encontró ningún archivo asociado al estudio. "
            "Sube al menos una imagen ECG o un archivo de señal antes de ejecutar el análisis."
        )
    )


@router.post("/studies/{study_id}/run")
def enqueue_analysis(study_id: str):
    sb = get_supabase()

    study_result = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_result.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    study = study_result.data[0]

    # Detecta job_type real según archivos existentes
    job_type = _detect_job_type(sb, study_id, study["source_type"])

    job = sb.table("analysis_jobs").insert({
        "study_id": study_id,
        "job_type": job_type,
        "status": "queued"
    }).execute()

    sb.table("ecg_studies").update({"status": "queued"}).eq("id", study_id).execute()
    return job.data[0]


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    sb = get_supabase()
    result = sb.table("analysis_jobs").select("*").eq("id", job_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return result.data[0]


@router.get("/studies/{study_id}/latest")
def get_latest_analysis(study_id: str):
    sb = get_supabase()

    analysis_res = (
        sb.table("ecg_analysis")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not analysis_res.data:
        return {"analysis": None, "metrics": None, "findings": [], "report": None, "ai": None}

    analysis = analysis_res.data[0]
    analysis_id = analysis["id"]

    metrics  = sb.table("ecg_metrics").select("*").eq("analysis_id", analysis_id).execute().data
    findings = sb.table("ecg_findings").select("*").eq("analysis_id", analysis_id).execute().data
    report   = sb.table("reports").select("*").eq("analysis_id", analysis_id).execute().data
    ai = (
        sb.table("ai_interpretations")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )

    return {
        "analysis": analysis,
        "metrics": metrics[0] if metrics else None,
        "findings": findings,
        "report": report[0] if report else None,
        "ai": ai[0] if ai else None,
    }