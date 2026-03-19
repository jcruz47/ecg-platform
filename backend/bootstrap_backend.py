from pathlib import Path

dirs = [
    "app/core",
    "app/db",
    "app/schemas",
    "app/routers",
    "app/services",
]

files = [
    "app/__init__.py",
    "app/main.py",
    "app/core/__init__.py",
    "app/core/config.py",
    "app/db/__init__.py",
    "app/db/supabase_client.py",
    "app/schemas/__init__.py",
    "app/schemas/patient.py",
    "app/schemas/study.py",
    "app/schemas/ai.py",
    "app/routers/__init__.py",
    "app/routers/doctors.py",
    "app/routers/patients.py",
    "app/routers/studies.py",
    "app/routers/uploads.py",
    "app/routers/analysis.py",
    "app/routers/ai.py",
    "app/services/__init__.py",
    "app/services/signal_pipeline.py",
    "app/services/image_pipeline.py",
    "app/services/llm_lmstudio.py",
    ".env",
    "worker.py",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

for f in files:
    p = Path(f)
    if not p.exists():
        p.write_text("", encoding="utf-8")

print("Estructura backend creada.")