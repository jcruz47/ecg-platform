from fastapi import APIRouter
from app.db.supabase_client import get_supabase

router = APIRouter()


@router.get("")
def list_doctors():
    sb = get_supabase()
    result = sb.table("doctors").select("*").order("created_at").execute()
    return result.data