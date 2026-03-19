from fastapi import APIRouter, HTTPException
from app.db.supabase_client import get_supabase
from app.schemas.ai import AskAIRequest
from app.services.llm_lmstudio import chat_complete, get_llm_backend_info
from app.services.timeline_helpers import get_patient_timeline

router = APIRouter()


def _effective_sampling_rate(study: dict, analysis: dict | None) -> float | None:
    """
    Decide qué frecuencia de muestreo mostrar al LLM.

    Reglas:
    - Si el estudio/análisis viene de imagen reconstruida, usar la frecuencia
      calculada en analysis.summary_json.sampling_rate.
    - Si el estudio viene de señal directa, usar study.sampling_rate_hz.
    - Si falta una, usar la otra como fallback.
    """
    source_type = str(study.get("source_type") or "").lower()
    analyzer_type = str(analysis.get("analyzer_type") or "").lower() if analysis else ""

    analysis_summary = (analysis or {}).get("summary_json") or {}
    analysis_fs = analysis_summary.get("sampling_rate")
    study_fs = study.get("sampling_rate_hz")

    is_image_based = ("image" in source_type) or ("image" in analyzer_type)

    # Caso 1: análisis basado en imagen reconstruida
    if is_image_based:
        if analysis_fs is not None:
            try:
                return float(analysis_fs)
            except (TypeError, ValueError):
                pass

        if study_fs is not None:
            try:
                return float(study_fs)
            except (TypeError, ValueError):
                pass

        return None

    # Caso 2: señal directa
    if study_fs is not None:
        try:
            return float(study_fs)
        except (TypeError, ValueError):
            pass

    # Fallback final
    if analysis_fs is not None:
        try:
            return float(analysis_fs)
        except (TypeError, ValueError):
            pass

    return None


def build_context_text(
    study: dict,
    analysis: dict,
    metrics: dict | None,
    findings: list[dict]
) -> str:
    effective_fs = _effective_sampling_rate(study, analysis)

    return f"""
Estudio:
- ID: {study.get('id')}
- Tipo de fuente: {study.get('source_type')}
- Estado: {study.get('status')}
- Frecuencia de muestreo: {effective_fs}
- Derivaciones: {study.get('lead_count')}

Resumen del análisis:
{analysis.get('summary_json', {}) if analysis else {}}

Métricas:
{metrics or {}}

Hallazgos:
{findings}
""".strip()


@router.get("/backend-info")
def backend_info():
    return get_llm_backend_info()


@router.get("/studies/{study_id}/messages")
def get_study_messages(study_id: str):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("id").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    result = (
        sb.table("ai_chat_messages")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at")
        .execute()
    )
    return result.data


@router.post("/studies/{study_id}/interpret")
def interpret_study(study_id: str):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    study = study_res.data[0]

    analysis_res = (
        sb.table("ecg_analysis")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not analysis_res.data:
        raise HTTPException(status_code=404, detail="No hay análisis para este estudio")

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

    metrics = metrics_res.data[0] if metrics_res.data else {}
    findings = findings_res.data

    context = build_context_text(study, analysis, metrics, findings)

    prompt = f"""
Genera una interpretación preliminar prudente de este ECG.
No inventes hallazgos.
No des diagnóstico definitivo.
Responde en español.

CONTEXTO:
{context}
""".strip()

    interpretation = chat_complete([{"role": "user", "content": prompt}])

    saved = sb.table("ai_interpretations").insert({
        "study_id": study_id,
        "interpretation_text": interpretation
    }).execute()

    return saved.data[0]


@router.post("/studies/{study_id}/ask")
def ask_about_study(study_id: str, payload: AskAIRequest):
    sb = get_supabase()

    study_res = sb.table("ecg_studies").select("*").eq("id", study_id).execute()
    if not study_res.data:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    study = study_res.data[0]

    analysis_res = (
        sb.table("ecg_analysis")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not analysis_res.data:
        raise HTTPException(status_code=404, detail="No hay análisis para este estudio")

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

    metrics = metrics_res.data[0] if metrics_res.data else {}
    findings = findings_res.data

    history_res = (
        sb.table("ai_chat_messages")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at")
        .execute()
    )

    history_messages = [
        {"role": row["role"], "content": row["content"]}
        for row in history_res.data[-8:]
    ]

    context = build_context_text(study, analysis, metrics, findings)

    final_user_prompt = f"""
Responde la pregunta SOLO con base en este contexto del estudio ECG.
Si no hay suficiente información, dilo claramente.
No inventes diagnóstico definitivo.
Responde en español.

CONTEXTO:
{context}

PREGUNTA:
{payload.question}
""".strip()

    answer = chat_complete(history_messages + [{"role": "user", "content": final_user_prompt}])

    sb.table("ai_chat_messages").insert([
        {
            "study_id": study_id,
            "role": "user",
            "content": payload.question
        },
        {
            "study_id": study_id,
            "role": "assistant",
            "content": answer
        }
    ]).execute()

    return {"answer": answer}


@router.post("/patients/{patient_id}/longitudinal-summary")
def longitudinal_summary(patient_id: str):
    sb = get_supabase()

    payload = get_patient_timeline(sb, patient_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    prompt = f"""
Genera un resumen longitudinal prudente del historial ECG de este paciente.

Reglas:
- Resume cambios relevantes entre estudios.
- Menciona tendencias de frecuencia cardíaca, SDNN, RMSSD y calidad de señal si existen.
- Si hay pocos estudios o faltan métricas, dilo claramente.
- No inventes diagnósticos definitivos.
- Responde en español.

DATOS:
{payload}
""".strip()

    summary = chat_complete(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return {"summary": summary}