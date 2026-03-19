import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.db.supabase_client import get_supabase
from app.core.config import settings

router = APIRouter()


@router.post("/{study_id}/signal")
async def upload_signal(study_id: str, file: UploadFile = File(...)):
    allowed = {"csv", "txt", "xlsx", "npy", "mat"}
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo sin nombre")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Formato no soportado")

    content = await file.read()
    object_path = f"{study_id}/{uuid.uuid4()}_{file.filename}"

    sb = get_supabase()
    sb.storage.from_(settings.signal_bucket).upload(
        path=object_path,
        file=content,
        file_options={
            "content-type": file.content_type or "application/octet-stream",
            "upsert": "false"
        }
    )

    result = sb.table("ecg_signal_files").insert({
        "study_id": study_id,
        "bucket_name": settings.signal_bucket,
        "object_path": object_path,
        "original_filename": file.filename,
        "file_format": ext,
        "mime_type": file.content_type,
        "file_size_bytes": len(content)
    }).execute()

    return result.data[0]


@router.post("/{study_id}/image")
async def upload_image(study_id: str, file: UploadFile = File(...)):
    allowed = {"png", "jpg", "jpeg", "webp"}
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo sin nombre")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

    content = await file.read()
    object_path = f"{study_id}/{uuid.uuid4()}_{file.filename}"

    sb = get_supabase()
    sb.storage.from_(settings.image_bucket).upload(
        path=object_path,
        file=content,
        file_options={
            "content-type": file.content_type or "image/png",
            "upsert": "false"
        }
    )

    result = sb.table("ecg_image_files").insert({
        "study_id": study_id,
        "bucket_name": settings.image_bucket,
        "object_path": object_path,
        "original_filename": file.filename,
        "mime_type": file.content_type,
        "file_size_bytes": len(content)
    }).execute()

    return result.data[0]