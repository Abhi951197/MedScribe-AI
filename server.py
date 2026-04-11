"""
MedScribe AI — FastAPI backend.

Endpoints:
  GET  /              → serves the single-page UI (static/index.html)
  GET  /samples       → list available dataset samples (grouped by specialty)
  GET  /sample-audio  → stream a dataset audio file (for the in-browser player)
  GET  /sample-transcript → ground-truth transcript for a sample
  POST /process       → run the full agent pipeline on uploaded audio OR pasted transcript
  POST /process-sample→ run the full agent pipeline on a dataset sample by name
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from medical_agents import MedicalPipeline, load_whisper_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WHISPER_MODEL_NAME = "base"  # fixed per product decision

PROJECT_ROOT = Path(__file__).parent
STATIC_DIR = PROJECT_ROOT / "static"
AUDIO_DIR = PROJECT_ROOT / "audio_recordings" / "Audio_Recordings"
TRANSCRIPT_DIR = PROJECT_ROOT / "audio_recordings" / "Clean_Transcripts"

CATEGORY_LABELS = {
    "CAR": "Cardiology",
    "DER": "Dermatology",
    "GAS": "Gastroenterology",
    "GEN": "General Medicine",
    "MSK": "Musculoskeletal",
    "NEU": "Neurology",
    "RES": "Respiratory",
    "URO": "Urology",
    "OBG": "Obstetrics & Gynae",
    "PSY": "Psychiatry",
    "PED": "Pediatrics",
    "END": "Endocrinology",
    "OPH": "Ophthalmology",
    "ENT": "ENT",
    "HEM": "Hematology",
    "INF": "Infectious Disease",
}

# ---------------------------------------------------------------------------
# Heavy resources — loaded once on startup
# ---------------------------------------------------------------------------

print(f"[startup] loading Whisper model '{WHISPER_MODEL_NAME}' …")
_whisper = load_whisper_model(WHISPER_MODEL_NAME)
print("[startup] whisper ready")

if not GEMINI_API_KEY:
    print("[startup] WARNING: GEMINI_API_KEY is not set in .env")

_pipeline = MedicalPipeline(_whisper, GEMINI_API_KEY) if GEMINI_API_KEY else None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="MedScribe AI", version="1.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "static/index.html missing")
    return FileResponse(index)


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@app.get("/samples")
def list_samples():
    if not AUDIO_DIR.exists():
        return {"samples": [], "categories": {}}
    files = sorted(p.name for p in AUDIO_DIR.glob("*.mp3"))
    grouped: dict[str, list[str]] = {}
    for f in files:
        prefix = f[:3]
        grouped.setdefault(prefix, []).append(f)
    return {
        "samples": files,
        "categories": {p: CATEGORY_LABELS.get(p, p) for p in grouped.keys()},
        "grouped": grouped,
    }


def _safe_sample(name: str) -> Path:
    p = (AUDIO_DIR / name).resolve()
    if not str(p).startswith(str(AUDIO_DIR.resolve())) or not p.exists():
        raise HTTPException(404, f"sample '{name}' not found")
    return p


@app.get("/sample-audio")
def sample_audio(name: str):
    p = _safe_sample(name)
    return FileResponse(p, media_type="audio/mpeg")


@app.get("/sample-transcript")
def sample_transcript(name: str):
    stem = Path(name).stem
    t = TRANSCRIPT_DIR / f"{stem}.txt"
    if not t.exists():
        return {"transcript": None}
    return {"transcript": t.read_text(encoding="utf-8", errors="ignore")}


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------

def _ctx_to_dict(ctx) -> dict:
    return {
        "language": ctx.language,
        "transcript": ctx.transcript,
        "entities": ctx.entities,
        "emergency": ctx.emergency,
        "soap_note": ctx.soap_note,
        "treatment_plan": ctx.treatment_plan,
        "patient_instructions": ctx.patient_instructions,
        "timings": {k: round(v, 2) for k, v in ctx.timings.items()},
        "trace": ctx.trace,
    }


def _require_pipeline():
    if _pipeline is None:
        raise HTTPException(500, "GEMINI_API_KEY not configured in .env")


@app.post("/process")
async def process(
    audio: UploadFile | None = File(None),
    transcript: str | None = Form(None),
):
    _require_pipeline()
    if not audio and not (transcript and transcript.strip()):
        raise HTTPException(400, "Provide either an audio file or a transcript")

    audio_path: str | None = None
    if audio is not None:
        suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            audio_path = tmp.name
        size = os.path.getsize(audio_path)
        if size < 1024:
            raise HTTPException(
                400,
                f"Audio file is too small ({size} bytes). "
                "Record for at least 1–2 seconds.",
            )

    try:
        ctx = _pipeline.run(audio_path=audio_path, transcript=transcript)
    except Exception as e:
        raise HTTPException(500, f"pipeline error: {e}")
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

    return JSONResponse(_ctx_to_dict(ctx))


@app.post("/process-sample")
def process_sample(name: str = Form(...)):
    _require_pipeline()
    p = _safe_sample(name)
    try:
        ctx = _pipeline.run(audio_path=str(p))
    except Exception as e:
        raise HTTPException(500, f"pipeline error: {e}")
    return JSONResponse(_ctx_to_dict(ctx))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
