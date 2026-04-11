"""
Medical Transcription & Emergency Alert AI - Streamlit app (agent-based).

Three input modes:
  1. Live microphone recording (st.audio_input)
  2. Upload audio file (mp3/wav/m4a/ogg/webm)
  3. Dataset sample picker (from audio_recordings/Audio_Recordings)

The pipeline is a chain of specialist agents defined in medical_agents.py.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from medical_agents import AgentContext, MedicalPipeline, load_whisper_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")

PROJECT_ROOT = Path(__file__).parent
AUDIO_DIR = PROJECT_ROOT / "audio_recordings" / "Audio_Recordings"
TRANSCRIPT_DIR = PROJECT_ROOT / "audio_recordings" / "Clean_Transcripts"

CATEGORY_LABELS = {
    "CAR": "Cardiology",
    "DER": "Dermatology",
    "GAS": "Gastroenterology",
    "GEN": "General",
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

st.set_page_config(
    page_title="MedScribe AI - Agent Pipeline",
    page_icon="🏥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=DEFAULT_API_KEY,
    help="Get a free key at https://aistudio.google.com/app/apikey",
)

whisper_size = st.sidebar.selectbox(
    "Whisper model",
    ["tiny", "base", "small"],
    index=1,
    help="'tiny' is fastest. 'small' is more accurate but heavier.",
)

st.sidebar.divider()
st.sidebar.markdown(
    "### Agent Pipeline\n"
    "1. **TranscriptionAgent** (Whisper)\n"
    "2. **EntityExtractionAgent** (LLM)\n"
    "3. **EmergencyDetectionAgent** (Rules + LLM)\n"
    "4. **ClinicalSummaryAgent** (SOAP)\n"
    "5. **TreatmentAdvisorAgent**\n"
    "6. **PatientEducationAgent**"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🏥 MedScribe AI")
st.caption(
    "Agent-based medical transcription, triage, and clinical note generator. "
    "Whisper ASR + Gemini LLM agents, orchestrated in a pipeline."
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Whisper model…")
def get_whisper(size: str):
    return load_whisper_model(size)


def get_pipeline(size: str, key: str) -> MedicalPipeline:
    return MedicalPipeline(get_whisper(size), key)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

@st.cache_data
def list_dataset_samples() -> list[str]:
    if not AUDIO_DIR.exists():
        return []
    return sorted(p.name for p in AUDIO_DIR.glob("*.mp3"))


def sample_label(name: str) -> str:
    prefix = name[:3]
    return f"{name}  —  {CATEGORY_LABELS.get(prefix, 'Unknown')}"


# ---------------------------------------------------------------------------
# Input selection
# ---------------------------------------------------------------------------

st.subheader("1️⃣  Choose an input")
tab_mic, tab_upload, tab_dataset, tab_text = st.tabs(
    ["🎙️ Live Recording", "📁 Upload File", "📚 Dataset Sample", "📝 Paste Transcript"]
)

audio_path: str | None = None
transcript_override: str | None = None

with tab_mic:
    st.write("Press record, speak, then press stop. Works on desktop and mobile.")
    mic_audio = st.audio_input("Record consultation audio")
    if mic_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(mic_audio.getbuffer())
            audio_path = tmp.name
        st.audio(audio_path)
        st.success("Recording captured ✔️")

with tab_upload:
    uploaded = st.file_uploader(
        "Upload consultation audio",
        type=["mp3", "wav", "m4a", "ogg", "webm", "flac"],
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name
        st.audio(audio_path)

with tab_dataset:
    samples = list_dataset_samples()
    if not samples:
        st.info(f"No dataset found at {AUDIO_DIR}")
    else:
        st.write(f"**{len(samples)} samples** available across specialties.")
        categories = sorted({s[:3] for s in samples})
        cat_filter = st.multiselect(
            "Filter by specialty",
            options=categories,
            format_func=lambda c: f"{c} - {CATEGORY_LABELS.get(c, c)}",
        )
        filtered = [s for s in samples if not cat_filter or s[:3] in cat_filter]
        pick = st.selectbox(
            "Pick a sample",
            options=filtered,
            format_func=sample_label,
            key="dataset_pick",
        )
        if pick:
            audio_path = str(AUDIO_DIR / pick)
            st.audio(audio_path)
            # Show the ground-truth transcript if present
            ref_txt = TRANSCRIPT_DIR / f"{Path(pick).stem}.txt"
            if ref_txt.exists():
                with st.expander("📄 Ground-truth transcript (reference)"):
                    st.text(ref_txt.read_text(encoding="utf-8", errors="ignore"))

with tab_text:
    pasted = st.text_area(
        "Paste or type a transcript to skip ASR entirely",
        height=240,
        placeholder="D: What brought you in today?\nP: I've been having chest pain for 2 hours...",
    )
    if pasted.strip():
        transcript_override = pasted.strip()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

st.divider()
col_run, col_clear = st.columns([3, 1])
run_clicked = col_run.button("🚀 Run agent pipeline", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear result", use_container_width=True):
    st.session_state.pop("result_ctx", None)

if run_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not audio_path and not transcript_override:
        st.error("Pick an input first (mic, upload, dataset, or paste transcript).")
    else:
        try:
            pipeline = get_pipeline(whisper_size, api_key)
        except Exception as e:
            st.error(f"Failed to initialise pipeline: {e}")
            st.stop()

        status_box = st.empty()
        progress = st.progress(0.0, text="Starting…")

        def on_progress(agent_name: str, idx: int, total: int) -> None:
            progress.progress(idx / total, text=f"Running {agent_name} ({idx}/{total})")

        try:
            ctx = pipeline.run(
                audio_path=audio_path,
                transcript=transcript_override,
                progress_cb=on_progress,
            )
            progress.progress(1.0, text="Done")
            status_box.success("Pipeline complete ✔️")
            st.session_state["result_ctx"] = ctx
        except Exception as e:
            status_box.error(f"Pipeline error: {e}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

ctx: AgentContext | None = st.session_state.get("result_ctx")
if ctx is not None:
    st.divider()
    st.subheader("2️⃣  Results")

    # --- Emergency banner ---
    emg = ctx.emergency or {}
    if emg.get("is_emergency"):
        level = emg.get("urgency_level", "HIGH")
        color = {"CRITICAL": "#b00020", "HIGH": "#d32f2f", "MODERATE": "#f57c00"}.get(
            level, "#d32f2f"
        )
        flags = emg.get("red_flags") or []
        st.markdown(
            f"""
            <div style="background:{color};color:white;padding:18px;border-radius:10px;
                        font-size:18px;font-weight:600;">
              🚨 EMERGENCY DETECTED — {level}<br/>
              <span style="font-size:15px;font-weight:400;">
                Red flags: {", ".join(flags) or "—"}<br/>
                Reason: {emg.get("reason", "")}<br/>
                Action: {emg.get("recommended_action", "")}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ No emergency detected by rule-set or LLM triage.")

    # --- Tabs for agent outputs ---
    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "📝 Transcript",
            "🧬 Entities",
            "🩺 SOAP Note",
            "💊 Treatment Plan",
            "🧑‍⚕️ Patient Instructions",
            "🔍 Agent Trace",
        ]
    )

    with t1:
        st.markdown(f"**Detected language:** `{ctx.language}`")
        st.text_area("Transcript", ctx.transcript, height=300)

    with t2:
        ents = ctx.entities or {}
        if "_raw" in ents:
            st.warning("LLM did not return parseable JSON. Raw output below.")
            st.code(ents["_raw"])
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Chief complaint**")
                st.write(ents.get("chief_complaint") or "—")
                st.markdown("**Symptoms**")
                for s in ents.get("symptoms") or []:
                    st.markdown(f"- {s}")
                st.markdown("**Diagnoses (working)**")
                for d in ents.get("diagnoses") or []:
                    st.markdown(f"- {d}")
                st.markdown("**Procedures**")
                for p in ents.get("procedures") or []:
                    st.markdown(f"- {p}")
            with c2:
                st.markdown("**Vitals**")
                vitals = ents.get("vitals") or {}
                if any(vitals.values()):
                    for k, v in vitals.items():
                        if v:
                            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")
                else:
                    st.write("— none extracted —")
                st.markdown("**Medications**")
                for m in ents.get("medications") or []:
                    if isinstance(m, dict):
                        parts = [m.get("name"), m.get("dose"), m.get("frequency")]
                        st.markdown("- " + " ".join(p for p in parts if p))
                    else:
                        st.markdown(f"- {m}")
                st.markdown("**Allergies**")
                for a in ents.get("allergies") or []:
                    st.markdown(f"- {a}")
            with st.expander("Raw JSON"):
                st.json(ents)

    with t3:
        st.markdown(ctx.soap_note or "_No note generated._")

    with t4:
        st.markdown(ctx.treatment_plan or "_No treatment plan generated._")
        st.caption("⚠️ Decision-support only. Always review with a clinician.")

    with t5:
        st.markdown(ctx.patient_instructions or "_No patient instructions generated._")

    with t6:
        st.markdown("**Timings (seconds)**")
        if ctx.timings:
            st.json({k: round(v, 2) for k, v in ctx.timings.items()})
        st.markdown("**Trace**")
        st.code("\n".join(ctx.trace) or "—")

    # --- Download full report ---
    report = {
        "language": ctx.language,
        "transcript": ctx.transcript,
        "entities": ctx.entities,
        "emergency": ctx.emergency,
        "soap_note": ctx.soap_note,
        "treatment_plan": ctx.treatment_plan,
        "patient_instructions": ctx.patient_instructions,
        "timings": ctx.timings,
    }
    st.download_button(
        "⬇️ Download full report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="medscribe_report.json",
        mime="application/json",
    )
