import time
from datetime import datetime, timezone

from app.db.supabase_client import get_supabase
from app.core.config import settings
from app.services.signal_pipeline import load_signal_from_bytes, analyze_signal
from app.services.image_full_pipeline import run_full_image_pipeline


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_latest_signal_file(sb, study_id: str):
    result = (
        sb.table("ecg_signal_files")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def get_latest_image_file(sb, study_id: str):
    result = (
        sb.table("ecg_image_files")
        .select("*")
        .eq("study_id", study_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def create_analysis_bundle(sb, study_id: str, job_id: str, analyzer_type: str, payload: dict):
    analysis = sb.table("ecg_analysis").insert({
        "study_id": study_id,
        "job_id": job_id,
        "analyzer_type": analyzer_type,
        "qc_score": payload.get("qc_score"),
        "summary_json": payload.get("summary", {})
    }).execute().data[0]

    metrics = dict(payload.get("metrics") or {})

    # Si el payload trae qc_score y no se guardó en métricas, lo copiamos aquí
    if metrics.get("signal_quality_score") is None and payload.get("qc_score") is not None:
        metrics["signal_quality_score"] = payload.get("qc_score")

    sb.table("ecg_metrics").insert({
        "analysis_id": analysis["id"],
        **metrics
    }).execute()

    for finding in payload.get("findings", []):
        sb.table("ecg_findings").insert({
            "analysis_id": analysis["id"],
            "label": finding["label"],
            "severity": finding["severity"],
            "confidence": finding.get("confidence"),
            "details_json": finding.get("details_json", {})
        }).execute()

    sb.table("reports").insert({
        "analysis_id": analysis["id"],
        "report_json": {
            "study_id": study_id,
            "summary": payload.get("summary", {}),
            "metrics": metrics,
            "findings": payload.get("findings", [])
        }
    }).execute()


def process_signal_job(sb, job: dict):
    study_id = job["study_id"]
    study = sb.table("ecg_studies").select("*").eq("id", study_id).execute().data[0]

    file_row = get_latest_signal_file(sb, study_id)
    if not file_row:
        raise ValueError(
            "No existe archivo de señal asociado al estudio. "
            "Sube un archivo CSV, TXT, NPY, XLSX o MAT antes de ejecutar el análisis."
        )

    raw_bytes = sb.storage.from_(file_row["bucket_name"]).download(file_row["object_path"])

    filename_hint = (
        file_row.get("original_filename")
        or file_row.get("file_name")
        or file_row.get("object_path")
        or file_row.get("file_format")
    )

    signal = load_signal_from_bytes(raw_bytes, filename_hint)

    fs = float(study.get("sampling_rate_hz") or settings.default_sampling_rate)
    payload = analyze_signal(signal, fs)

    create_analysis_bundle(sb, study_id, job["id"], "classic", payload)


def process_image_job(sb, job: dict):
    study_id = job["study_id"]

    file_row = get_latest_image_file(sb, study_id)
    if not file_row:
        raise ValueError(
            "No existe imagen asociada al estudio. "
            "Sube una imagen PNG, JPG o WEBP antes de ejecutar el análisis."
        )

    raw_bytes = sb.storage.from_(file_row["bucket_name"]).download(file_row["object_path"])

    payload = run_full_image_pipeline(sb, study_id, raw_bytes)

    qc = payload["qc"]
    image_cls = payload["image_classification"]
    fusion = payload["fusion"]

    sb.table("ecg_image_qc").insert({
        "study_id": study_id,
        "usable": qc["usable"],
        "quality_label": qc["quality_label"],
        "sharpness": qc["sharpness"],
        "brightness": qc["brightness"],
        "contrast": qc["contrast"],
    }).execute()

    sb.table("ecg_image_segmentation").insert({
        "study_id": study_id,
        "mask_bucket_name": "ecg-image-masks",
        "mask_object_path": payload["mask_path"],
    }).execute()

    if payload["reconstruction"] is not None:
        sb.table("ecg_image_reconstruction").insert({
            "study_id": study_id,
            "derived_bucket_name": "ecg-image-derived",
            "derived_object_path": payload["derived_signal_path"],
            "reconstruction_quality_score": payload["reconstruction"]["reconstruction_quality_score"],
            "estimated_sampling_rate_hz": payload["reconstruction"]["estimated_sampling_rate_hz"],
        }).execute()

    if image_cls is not None:
        sb.table("ecg_image_classification").insert({
            "study_id": study_id,
            "model_name": image_cls["model_name"],
            "predicted_label": image_cls["predicted_label"],
            "confidence": image_cls["confidence"],
            "probabilities_json": image_cls["probabilities"],
        }).execute()

    sb.table("ecg_fusion_results").insert({
        "study_id": study_id,
        "fusion_label": fusion["fusion_label"],
        "fusion_confidence": fusion["fusion_confidence"],
        "image_classifier_label": image_cls["predicted_label"] if image_cls else None,
        "signal_pipeline_summary": payload["signal_payload"]["summary"] if payload["signal_payload"] else {},
    }).execute()

    if payload["signal_payload"] is not None:
        create_analysis_bundle(
            sb,
            study_id,
            job["id"],
            "image_reconstructed_signal",
            payload["signal_payload"]
        )
    else:
        create_analysis_bundle(
            sb,
            study_id,
            job["id"],
            "image_qc_only",
            {
                "summary": {
                    "note": "No se pudo reconstruir señal con suficiente calidad"
                },
                "metrics": {
                    # No guardar sharpness como signal_quality_score
                    "signal_quality_score": None
                },
                "findings": [{
                    "label": "Imagen no utilizable para reconstrucción robusta",
                    "severity": "medium",
                    "confidence": 0.8,
                    "details_json": {
                        "quality_label": qc["quality_label"],
                        "issues": qc.get("issues", []),
                        "classifier_label": image_cls["predicted_label"] if image_cls else None
                    }
                }],
                # Puedes guardar un score de QC en analysis.qc_score si quieres
                "qc_score": None,
            }
        )


def process_auto_job(sb, job: dict):
    """
    job_type = 'auto' o source_type ambiguo:
    corre el pipeline correcto según qué archivos existen en la DB.
    Prioridad: si hay imagen -> image pipeline; si solo señal -> signal pipeline.
    """
    study_id = job["study_id"]
    has_image = get_latest_image_file(sb, study_id) is not None
    has_signal = get_latest_signal_file(sb, study_id) is not None

    if not has_image and not has_signal:
        raise ValueError(
            "No se encontró ningún archivo asociado al estudio. "
            "Sube al menos una imagen o un archivo de señal antes de ejecutar el análisis."
        )

    if has_image:
        process_image_job(sb, job)

    if has_signal:
        process_signal_job(sb, job)


def main():
    sb = get_supabase()
    print("Worker iniciado...")

    while True:
        result = (
            sb.table("analysis_jobs")
            .select("*")
            .eq("status", "queued")
            .order("created_at")
            .limit(1)
            .execute()
        )

        if not result.data:
            time.sleep(3)
            continue

        job = result.data[0]

        sb.table("analysis_jobs").update({
            "status": "running",
            "started_at": now_iso()
        }).eq("id", job["id"]).execute()

        sb.table("ecg_studies").update({
            "status": "processing"
        }).eq("id", job["study_id"]).execute()

        try:
            job_type = job["job_type"]

            if job_type == "signal":
                process_signal_job(sb, job)
            elif job_type == "image":
                process_image_job(sb, job)
            elif job_type in ("auto", "both"):
                process_auto_job(sb, job)
            else:
                process_auto_job(sb, job)

            sb.table("analysis_jobs").update({
                "status": "succeeded",
                "finished_at": now_iso()
            }).eq("id", job["id"]).execute()

            sb.table("ecg_studies").update({
                "status": "completed"
            }).eq("id", job["study_id"]).execute()

            print(f"Job {job['id']} completado")

        except Exception as e:
            sb.table("analysis_jobs").update({
                "status": "failed",
                "error_message": str(e),
                "finished_at": now_iso()
            }).eq("id", job["id"]).execute()

            sb.table("ecg_studies").update({
                "status": "failed"
            }).eq("id", job["study_id"]).execute()

            print(f"Error en job {job['id']}: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()