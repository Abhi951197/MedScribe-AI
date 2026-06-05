# 🏥 MedScribe AI

> **Agent-based medical transcription, triage & clinical note generator.**
> Converts doctor–patient audio into structured SOAP notes, flags emergencies
> in real time, and suggests decision-support in under a minute.

Built from the scope in [`deep-research-report.md`](deep-research-report.md) —
an AI-based medical transcription + emergency-alert system for Indian
healthcare settings (clinics, wards, teleconsults, ambulance/EMS).

---

## ✨ Features

- **🎙️ Live microphone recording** (browser MediaRecorder API)
- **📁 Audio file upload** — mp3, wav, m4a, ogg, webm, flac
- **📚 Dataset sample picker** — browse 272+ clinical dialogues by specialty
- **📝 Paste transcript** — skip ASR and run only the LLM agents
- **🚨 Real-time emergency detection** — rule-based red-flag keywords OR an LLM triage classifier, with urgency levels (CRITICAL / HIGH / MODERATE / LOW)
- **🩺 Structured SOAP note** generation
- **💊 Treatment plan** decision-support (investigations, meds, red flags, when to escalate)
- **🧑‍⚕️ Plain-language patient instructions**
- **📊 Agent trace + per-agent timings**
- **⬇️ JSON report download**

---

## 🧠 Architecture — Agent Pipeline

Six specialist agents run in sequence. Each one has a single responsibility.

| # | Agent | Tech | Role |
|---|---|---|---|
| 1 | **TranscriptionAgent** | OpenAI Whisper (`base`) | Audio → text + language detection |
| 2 | **EntityExtractionAgent** | Gemini Flash + regex | Symptoms, vitals, meds, diagnoses, history |
| 3 | **EmergencyDetectionAgent** | 40+ rules + LLM | Hybrid triage with urgency level |
| 4 | **ClinicalSummaryAgent** | Gemini Flash | SOAP-format clinical note |
| 5 | **TreatmentAdvisorAgent** | Gemini Flash | Decision-support plan |
| 6 | **PatientEducationAgent** | Gemini Flash | Plain-language patient instructions |

The orchestrator (`MedicalPipeline` in [`medical_agents.py`](medical_agents.py))
runs them on a shared `AgentContext` and aborts cleanly if transcription fails.

```
┌───────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐   ┌─────────┐
│ 🎙️ Audio  │ → │Transcribe│ → │ Entities │ → │Emergency│ → │  SOAP  │ → │ Patient │
└───────────┘   └──────────┘   └──────────┘   └─────────┘   └────────┘   └─────────┘
                                                     ↓
                                              🚨 Red Alert
```

---

## 🛠️ Tech Stack

**Backend** — FastAPI + Uvicorn · OpenAI Whisper (local ASR) · Google Gemini Flash (LLM, via REST)
**Frontend** — Single-page HTML/CSS/JS (no build step), Inter font, dark modern UI
**Runtime** — Python 3.10+

---

## 🚀 Quick Start

### 1. Install system dependency (ffmpeg)

Whisper requires `ffmpeg` on PATH to decode audio.

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt update && sudo apt install ffmpeg
```

Restart your terminal after install so PATH is refreshed, then verify:
```bash
ffmpeg -version
```

### 2. Clone & install Python dependencies

```bash
git clone https://github.com/Abhi951197/MedScribe-AI medscribe-ai
cd medscribe-ai

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure your Gemini API key

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey),
then copy the example env file:

```bash
cp .env.example .env
```

Edit `.env` and paste your key:
```
GEMINI_API_KEY=your-key-here
```

### 4. (Optional) Download the dataset

The repo **does not include** the ~1 GB audio dataset — it's listed in
`.gitignore`. You only need it if you want to use the **📚 Dataset Samples**
tab in the UI. The app works fine with live mic / file upload / pasted
transcripts without it.

The dataset is a collection of simulated doctor–patient consultations across
specialties (cardiology, dermatology, gastroenterology, MSK, etc.) with clean
ground-truth transcripts. Download and place it so the folder tree looks like:

```
GENAI/
├── audio_recordings/
│   ├── Audio_Recordings/     # CAR0001.mp3, DER0001.mp3, GAS0001.mp3, ...
│   └── Clean_Transcripts/    # CAR0001.txt, DER0001.txt, GAS0001.txt, ...
```

> **Dataset source:** [Fareez et al., Figshare — "A dataset of simulated patient-physician medical interviews"](https://www.kaggle.com/datasets/najamahmed97/audio-recording-whisper)

### 5. Run the app

```bash
python server.py
```

Open **http://127.0.0.1:8000** in your browser. That's it.

---

## 📂 Project Structure

```
GENAI/
├── server.py               # FastAPI backend — endpoints + pipeline wiring
├── medical_agents.py       # The 6 agents + orchestrator + Gemini REST client
├── static/
│   └── index.html          # Modern single-page UI (no build step)
├── requirements.txt
├── .env.example
├── .gitignore
├── deep-research-report.md # Original product spec
└── README.md
```

---

## 🖼️ Screenshots

### Input screen — modern dark UI with 4 input modes

![Input screen](docs/01-input.png)

### Live microphone recording with pulsing animation

![Live recording](docs/02-recording.png)

### Dataset sample picker — filter by specialty

![Dataset picker](docs/03-dataset.png)

### Emergency detection — animated critical banner

![Emergency banner](docs/04-emergency.png)

### SOAP note + treatment plan + patient instructions (tabbed)

![SOAP note](docs/05-soap.png)

> 📸 **Taking your own screenshots:** Run the app, work through a sample, and
> drop PNGs into [`docs/`](docs/) using the filenames above.

---

## 📋 Sample Output (from real run)

Running the full pipeline on `CAR0001.mp3` (39-year-old male, left-sided chest pain):

### 🚨 Emergency Detection

```json
{
  "is_emergency": true,
  "urgency_level": "CRITICAL",
  "rule_hits": ["chest pain", "trouble breathing", "difficulty breathing",
                "loss of consciousness", "heart attack", "stroke"],
  "reason": "The patient is a smoker with a strong family history of early
             cardiac events presenting with severe chest pain, respiratory
             distress, and lightheadedness, which are highly suggestive of
             acute coronary syndrome or pulmonary embolism.",
  "recommended_action": "The patient requires immediate emergency medical
                         evaluation, including an ECG and cardiac biomarkers,
                         to rule out life-threatening cardiac or pulmonary
                         conditions."
}
```

### 🧬 Extracted Entities

```json
{
  "chief_complaint": "Left-sided chest pain",
  "symptoms": ["left-sided chest pain", "shortness of breath",
               "lightheadedness", "palpitations", "sweating", "neck swelling"],
  "patient_demographics": {"age": "39", "sex": "male"},
  "history": [
    "Smoker (1 pack per day for 10-15 years)",
    "Alcohol use (10 drinks per week)",
    "Family history of heart attack (father at age 45)",
    "Family history of cholesterol problems"
  ]
}
```

### 🩺 SOAP Note (excerpt)

```markdown
**S: Subjective**
* Chief Complaint: Left-sided chest pain.
* HPI: 39-year-old male presents with sharp left-sided chest pain that began
  approximately 8 hours ago. Pain is 7-8/10, constant, worse when lying down
  and with deep inspiration, improves when sitting up. Associated symptoms:
  shortness of breath, lightheadedness, palpitations, diaphoresis.
* Social: 1 PPD smoker × 10-15 years, 10 drinks/week, occasional cannabis.
* Family: Father had MI at age 45; paternal cholesterol problems.

**O: Objective**
* Vitals not reported in transcript.
* Neck appears "slightly swollen" per patient.

**A: Assessment**
* Acute chest pain — differential includes acute coronary syndrome,
  pulmonary embolism, pericarditis, pneumothorax.
* High cardiac risk: family history of early MI, active smoker.

**P: Plan**
* Immediate ECG and cardiac troponin.
* CXR, CBC, BMP, D-dimer.
* IV access, continuous cardiac monitoring.
* Cardiology consult on positive findings.
```

### ⏱️ Typical Timings

| Stage | Time |
|---|---|
| TranscriptionAgent (Whisper base, ~90s audio) | ~8–15 s |
| EntityExtractionAgent | ~2–4 s |
| EmergencyDetectionAgent (rules + LLM) | ~2–3 s |
| ClinicalSummaryAgent | ~4–8 s |
| TreatmentAdvisorAgent | ~4–8 s |
| PatientEducationAgent | ~3–6 s |
| **Total** | **~25–50 s** |

---

## ⚠️ Known Limits

- **Gemini free-tier rate limit:** ~20 requests/minute. Running back-to-back
  samples may hit HTTP 429. Wait a minute or upgrade your key.
- **Whisper `base` accuracy:** good but not great on noisy audio or heavy
  accents. Switch to `small` in [`server.py`](server.py) (`WHISPER_MODEL_NAME`)
  for better accuracy at the cost of speed.
- **Decision-support is NOT a prescription.** Always review with a clinician.
- **No PHI safeguards in this demo.** Don't point it at real patient data
  without adding the privacy/audit layer from [`deep-research-report.md`](deep-research-report.md).

---

## 🧪 API Endpoints

The FastAPI server exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Serves the single-page UI |
| `GET`  | `/samples` | Lists dataset samples grouped by specialty |
| `GET`  | `/sample-audio?name=CAR0001.mp3` | Stream a dataset audio file |
| `GET`  | `/sample-transcript?name=CAR0001.mp3` | Ground-truth transcript for a sample |
| `POST` | `/process` | Run the full pipeline on uploaded audio or pasted transcript |
| `POST` | `/process-sample` | Run the pipeline on a dataset sample by name |

Example curl:
```bash
curl -X POST http://127.0.0.1:8000/process-sample \
  -F "name=CAR0001.mp3" | jq .
```

---

## 🗺️ Roadmap (from the research report)

- [ ] Multilingual support (Hindi, Marathi, Tamil, Telugu)
- [ ] On-device edge deployment for ambulance/EMS use
- [ ] EHR / ABHA integration
- [ ] Clinician feedback loop for continuous retraining
- [ ] DPDP Act / NDHM compliance hardening (audit logs, consent, encryption)

---

## 📄 License

MIT — see `LICENSE` (or add one before publishing).

---

**Built with** FastAPI · OpenAI Whisper · Google Gemini · Inter · ❤️
