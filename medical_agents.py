"""
Agent-based medical transcription & triage pipeline.

Each agent has a single responsibility. The Orchestrator runs them in sequence
and returns a structured report that the Streamlit UI renders.

Agents:
  1. TranscriptionAgent      - Whisper ASR (supports file + mic recording)
  2. EntityExtractionAgent   - Extract symptoms / vitals / meds / diagnoses (LLM)
  3. EmergencyDetectionAgent - Rule-based keywords + LLM verification
  4. ClinicalSummaryAgent    - SOAP-format clinical note (LLM)
  5. TreatmentAdvisorAgent   - Suggested management / next steps (LLM)
  6. PatientEducationAgent   - Plain-language patient instructions (LLM)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests
import whisper

# Gemini REST endpoint (matches the quickstart in .env)
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Critical keywords (rule-based first pass). Tuned for recall, not precision.
EMERGENCY_KEYWORDS: list[str] = [
    "chest pain", "crushing chest", "pressure in chest",
    "shortness of breath", "trouble breathing", "difficulty breathing",
    "can't breathe", "cannot breathe", "gasping",
    "heavy bleeding", "bleeding heavily", "hemorrhage",
    "unconscious", "unresponsive", "passed out", "loss of consciousness",
    "overdose", "poisoning", "suicidal", "suicide",
    "heart attack", "cardiac arrest",
    "stroke", "slurred speech", "face drooping", "facial droop",
    "seizure", "convulsion",
    "severe pain", "worst headache", "thunderclap headache",
    "anaphylaxis", "allergic reaction", "throat closing", "can't swallow",
    "blue lips", "cyanosis",
    "choking",
    "severe burn",
    "head injury", "hit my head",
    "broken bone", "compound fracture",
    "vomiting blood", "coughing blood", "blood in stool",
]

# Very light-weight regex for vitals (used as a deterministic backup).
VITAL_REGEX = {
    "blood_pressure": re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b"),
    "heart_rate":     re.compile(r"\b(?:heart rate|pulse|hr)\s*(?:of|is|=)?\s*(\d{2,3})\b", re.I),
    "temperature":    re.compile(r"\b(\d{2,3}(?:\.\d)?)\s*(?:°\s*)?(?:f|fahrenheit|c|celsius)\b", re.I),
    "spo2":           re.compile(r"\b(?:spo2|oxygen|o2\s*sat)\s*(?:of|is|=)?\s*(\d{2,3})\s*%?\b", re.I),
    "respiratory_rate": re.compile(r"\b(?:respiratory rate|resp rate|rr)\s*(?:of|is|=)?\s*(\d{1,2})\b", re.I),
}


# ---------------------------------------------------------------------------
# Shared context passed between agents
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    audio_path: str | None = None
    transcript: str = ""
    language: str = "en"
    entities: dict[str, Any] = field(default_factory=dict)
    emergency: dict[str, Any] = field(default_factory=dict)
    soap_note: str = ""
    treatment_plan: str = ""
    patient_instructions: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    def log(self, agent: str, msg: str) -> None:
        self.trace.append(f"[{agent}] {msg}")


# ---------------------------------------------------------------------------
# LLM client (Gemini REST — matches the quickstart in .env)
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around the Gemini REST API.

    Mirrors the quickstart:
        curl "https://generativelanguage.googleapis.com/v1beta/models/\
              gemini-flash-latest:generateContent" \
             -H 'Content-Type: application/json' \
             -H 'X-goog-api-key: $KEY' \
             -d '{"contents":[{"parts":[{"text":"..."}]}]}'
    """

    def __init__(self, api_key: str, timeout: int = 60) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Gemini {r.status_code}: {r.text[:400]}")
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response: {str(data)[:400]}") from e


def _configure_llm(api_key: str) -> GeminiClient:
    return GeminiClient(api_key)


def _llm_json(client: GeminiClient, prompt: str) -> dict[str, Any]:
    """Call the LLM and try hard to parse a JSON object out of the response."""
    text = (client.generate(prompt) or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"_raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"_raw": text}


# ---------------------------------------------------------------------------
# Agent 1: Transcription
# ---------------------------------------------------------------------------

class TranscriptionAgent:
    name = "TranscriptionAgent"

    def __init__(self, whisper_model: Any) -> None:
        self.model = whisper_model

    def run(self, ctx: AgentContext) -> None:
        import os
        if not ctx.audio_path:
            raise ValueError("No audio path supplied to TranscriptionAgent")
        size = os.path.getsize(ctx.audio_path) if os.path.exists(ctx.audio_path) else 0
        if size < 1024:
            raise ValueError(
                f"Audio file is empty or too small ({size} bytes). "
                "Did the microphone actually capture anything? Record for at "
                "least 1–2 seconds before stopping."
            )
        t0 = time.time()
        result = self.model.transcribe(ctx.audio_path, fp16=False)
        ctx.transcript = (result.get("text") or "").strip()
        ctx.language = result.get("language", "en")
        ctx.timings[self.name] = time.time() - t0
        ctx.log(self.name, f"lang={ctx.language}, chars={len(ctx.transcript)}")
        if not ctx.transcript:
            raise ValueError("Whisper produced an empty transcript.")


# ---------------------------------------------------------------------------
# Agent 2: Entity extraction
# ---------------------------------------------------------------------------

ENTITY_PROMPT = """You are a clinical information extraction engine.
From the doctor-patient transcript below, extract structured entities.

Return ONLY a valid JSON object with this exact schema:
{{
  "chief_complaint": "string",
  "symptoms":    ["string", ...],
  "vitals":      {{"blood_pressure": "string|null", "heart_rate": "string|null",
                  "temperature": "string|null", "spo2": "string|null",
                  "respiratory_rate": "string|null"}},
  "medications": [{{"name": "string", "dose": "string|null", "frequency": "string|null"}}],
  "diagnoses":   ["string", ...],
  "procedures":  ["string", ...],
  "allergies":   ["string", ...],
  "history":     ["string", ...],
  "patient_demographics": {{"age": "string|null", "sex": "string|null"}}
}}

Use empty lists / nulls when unknown. No commentary, only the JSON object.

Transcript:
\"\"\"{transcript}\"\"\"
"""


class EntityExtractionAgent:
    name = "EntityExtractionAgent"

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    def run(self, ctx: AgentContext) -> None:
        t0 = time.time()
        data = _llm_json(self.llm, ENTITY_PROMPT.format(transcript=ctx.transcript))

        # Regex backup for vitals if LLM missed any
        vitals = data.get("vitals") or {}
        for key, rx in VITAL_REGEX.items():
            if not vitals.get(key):
                m = rx.search(ctx.transcript)
                if m:
                    vitals[key] = m.group(0)
        data["vitals"] = vitals

        ctx.entities = data
        ctx.timings[self.name] = time.time() - t0
        ctx.log(self.name, f"symptoms={len(data.get('symptoms', []))}, "
                           f"meds={len(data.get('medications', []))}")


# ---------------------------------------------------------------------------
# Agent 3: Emergency detection
# ---------------------------------------------------------------------------

EMERGENCY_PROMPT = """You are a clinical triage classifier. Given a doctor-patient
transcript, decide whether the patient has a life-threatening or urgent condition.

Return ONLY a valid JSON object with this schema:
{{
  "is_emergency":     true|false,
  "urgency_level":    "CRITICAL"|"HIGH"|"MODERATE"|"LOW",
  "red_flags":        ["string", ...],
  "reason":           "string (one sentence)",
  "recommended_action":"string (one sentence)"
}}

Err on the side of flagging true emergencies (high recall).

Transcript:
\"\"\"{transcript}\"\"\"
"""


class EmergencyDetectionAgent:
    name = "EmergencyDetectionAgent"

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    def run(self, ctx: AgentContext) -> None:
        t0 = time.time()
        text_lower = ctx.transcript.lower()

        # Rule pass
        rule_hits = [kw for kw in EMERGENCY_KEYWORDS if kw in text_lower]

        # LLM pass
        llm_result: dict[str, Any] = {}
        try:
            llm_result = _llm_json(self.llm, EMERGENCY_PROMPT.format(transcript=ctx.transcript))
        except Exception as e:
            llm_result = {"_error": str(e)}

        is_emergency = bool(rule_hits) or bool(llm_result.get("is_emergency"))
        urgency = llm_result.get("urgency_level") or ("HIGH" if rule_hits else "LOW")
        red_flags = list({*rule_hits, *(llm_result.get("red_flags") or [])})

        ctx.emergency = {
            "is_emergency": is_emergency,
            "urgency_level": urgency,
            "red_flags": red_flags,
            "rule_hits": rule_hits,
            "reason": llm_result.get("reason", ""),
            "recommended_action": llm_result.get("recommended_action", ""),
        }
        ctx.timings[self.name] = time.time() - t0
        ctx.log(self.name, f"emergency={is_emergency}, level={urgency}, hits={len(red_flags)}")


# ---------------------------------------------------------------------------
# Agent 4: SOAP summary
# ---------------------------------------------------------------------------

SOAP_PROMPT = """You are an experienced clinical scribe. Convert the doctor-patient
transcript below into a concise SOAP note in markdown.

Use these sections (use bold headers):
**S: Subjective** - chief complaint, HPI, pertinent positives/negatives, PMH, social.
**O: Objective** - vitals and exam findings actually mentioned.
**A: Assessment** - working diagnosis and differential.
**P: Plan** - investigations, medications (with dose/route/frequency), follow-up.

Write only facts present or strongly implied in the transcript. Do not invent vitals.

Transcript:
\"\"\"{transcript}\"\"\"

Extracted entities (for reference, may be empty):
{entities}
"""


class ClinicalSummaryAgent:
    name = "ClinicalSummaryAgent"

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    def run(self, ctx: AgentContext) -> None:
        t0 = time.time()
        prompt = SOAP_PROMPT.format(
            transcript=ctx.transcript,
            entities=json.dumps(ctx.entities, indent=2)[:2000],
        )
        ctx.soap_note = (self.llm.generate(prompt) or "").strip()
        ctx.timings[self.name] = time.time() - t0
        ctx.log(self.name, f"note_chars={len(ctx.soap_note)}")


# ---------------------------------------------------------------------------
# Agent 5: Treatment advisor
# ---------------------------------------------------------------------------

TREATMENT_PROMPT = """You are a medical decision-support assistant. Based on the
transcript and extracted entities, suggest reasonable next steps for the clinician.

Return markdown with these sections:
### Recommended Investigations
- ...

### Suggested Medications / Management
- ...

### Red Flags to Watch For
- ...

### When to Escalate
- ...

Important: These are decision-support suggestions only, not prescriptions. Keep it
evidence-aligned and conservative. If information is insufficient, say so.

Transcript:
\"\"\"{transcript}\"\"\"
"""


class TreatmentAdvisorAgent:
    name = "TreatmentAdvisorAgent"

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    def run(self, ctx: AgentContext) -> None:
        t0 = time.time()
        ctx.treatment_plan = (
            self.llm.generate(TREATMENT_PROMPT.format(transcript=ctx.transcript)) or ""
        ).strip()
        ctx.timings[self.name] = time.time() - t0
        ctx.log(self.name, f"plan_chars={len(ctx.treatment_plan)}")


# ---------------------------------------------------------------------------
# Agent 6: Patient education
# ---------------------------------------------------------------------------

PATIENT_PROMPT = """Write plain-language patient instructions (6th-grade reading
level) based on the consultation. Use short bullet points covering:
- What we think is going on
- What to do at home
- Medications (if any) - how and when
- Warning signs that mean "go to the ER"
- When to follow up

Be warm but concise. Do not invent diagnoses.

Transcript:
\"\"\"{transcript}\"\"\"
"""


class PatientEducationAgent:
    name = "PatientEducationAgent"

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    def run(self, ctx: AgentContext) -> None:
        t0 = time.time()
        ctx.patient_instructions = (
            self.llm.generate(PATIENT_PROMPT.format(transcript=ctx.transcript)) or ""
        ).strip()
        ctx.timings[self.name] = time.time() - t0
        ctx.log(self.name, f"instructions_chars={len(ctx.patient_instructions)}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MedicalPipeline:
    """Runs the full agent pipeline. Accepts an optional progress callback
    that receives (agent_name, index, total) for UI updates."""

    def __init__(self, whisper_model: Any, api_key: str) -> None:
        llm = _configure_llm(api_key)
        self.agents = [
            TranscriptionAgent(whisper_model),
            EntityExtractionAgent(llm),
            EmergencyDetectionAgent(llm),
            ClinicalSummaryAgent(llm),
            TreatmentAdvisorAgent(llm),
            PatientEducationAgent(llm),
        ]

    def run(
        self,
        audio_path: str | None = None,
        transcript: str | None = None,
        progress_cb: Callable[[str, int, int], None] | None = None,
    ) -> AgentContext:
        ctx = AgentContext(audio_path=audio_path, transcript=(transcript or "").strip())
        total = len(self.agents)
        for idx, agent in enumerate(self.agents, start=1):
            # Skip transcription when a transcript is supplied directly
            if isinstance(agent, TranscriptionAgent) and transcript:
                ctx.log(agent.name, "skipped (transcript provided)")
                if progress_cb:
                    progress_cb(agent.name, idx, total)
                continue
            if progress_cb:
                progress_cb(agent.name, idx, total)
            try:
                agent.run(ctx)
            except Exception as e:
                ctx.log(agent.name, f"ERROR: {e}")
                # If the first agent (transcription) fails, abort — nothing
                # downstream can work without a transcript.
                if isinstance(agent, TranscriptionAgent) and not ctx.transcript:
                    ctx.log("Pipeline", "aborted: no transcript available")
                    break
        return ctx


# ---------------------------------------------------------------------------
# Whisper loader (cached at module level so the UI layer can wrap in cache)
# ---------------------------------------------------------------------------

def load_whisper_model(model_name: str = "base") -> Any:
    return whisper.load_model(model_name)
