from typing import Optional
from pydantic import BaseModel


class StudyCreate(BaseModel):
    patient_id: str
    doctor_id: str
    source_type: str
    sampling_rate_hz: Optional[float] = 360
    lead_count: Optional[int] = 1