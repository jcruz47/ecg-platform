from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from app.routers import doctors, patients, studies, uploads, analysis, ai
 
app = FastAPI(
    title="ECG Multimodal Platform API",
    version="0.1.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(doctors.router, prefix="/doctors", tags=["doctors"])
app.include_router(patients.router, prefix="/patients", tags=["patients"])
app.include_router(studies.router, prefix="/studies", tags=["studies"])
app.include_router(uploads.router, prefix="/uploads", tags=["uploads"])
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(ai.router, prefix="/ai", tags=["ai"])
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 