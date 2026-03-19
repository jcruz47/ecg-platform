from datetime import date
from typing import Optional
from pydantic import BaseModel


class PatientCreate(BaseModel):
    doctor_id: str
    first_name: str
    last_name: str
    sex: Optional[str] = None
    birth_date: Optional[date] = None
    notes: Optional[str] = None