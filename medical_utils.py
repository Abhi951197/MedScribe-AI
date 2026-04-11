import whisper
import google.generativeai as genai

# Emergency keywords based on the report
EMERGENCY_KEYWORDS = [
    "chest pain", "shortness of breath", "trouble breathing",
    "heavy bleeding", "unconscious", "overdose", "dizzy",
    "severe pain", "heart attack", "stroke"
]

def load_whisper_model(model_name="base"):
    """Load the Whisper model. 'base' or 'small' is good for fast MVP."""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    return model

def transcribe_audio(model, audio_path):
    """Transcribe audio using Whisper."""
    result = model.transcribe(audio_path)
    return result["text"]

def detect_emergencies(transcript):
    """Rule-based emergency detection."""
    transcript_lower = transcript.lower()
    flags = []
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in transcript_lower:
            flags.append(keyword)
    return flags

def generate_medical_summary(transcript, api_key):
    """Use Gemini API for summarization and NER."""
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as it's typically the fastest/free tier model currently
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an AI medical assistant. Please process the following doctor-patient conversation transcript.
    
    1. Extract clinical entities (Symptoms, Vitals, Diagnoses, Medications).
    2. Summarize the conversation into a structured clinical note using the SOAP format (Subjective, Objective, Assessment, Plan).
    3. Suggest any initial management steps.
    
    Transcript:
    "{transcript}"
    
    Format your response clearly using markdown.
    """
    
    response = model.generate_content(prompt)
    return response.text
