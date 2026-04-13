from models.whisper_model import transcribe
import re
import os

def analyze_audio(video_path: str, audio_content: bytes | None = None):
    # If audio_content was provided separately, we could write it and parse
    # For now, whisper can ingest the video container directly.
    result = transcribe(video_path)
    text = result["text"]
    segments = result["segments"]
    
    # Calculate simplistic audio features
    speech_rate_wpm = 0
    words = text.split()
    total_words = len(words)
    total_duration = segments[-1]["end"] if segments else 0
    if total_duration > 0:
        speech_rate_wpm = (total_words / (total_duration / 60.0))
        
    filler_words = {"um": len(re.findall(r'\bum\b', text, re.IGNORECASE)), 
                    "like": len(re.findall(r'\blike\b', text, re.IGNORECASE))}
    
    # Simple proxies
    pause_count = len([s for i, s in enumerate(segments[1:]) if s['start'] - segments[i]['end'] > 1.0])
    avg_pause_duration = 0.5 # proxy
    
    transcript = {
        "full_text": text.strip(),
        "segments": [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in segments]
    }
    
    audio_features = {
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "pause_count": pause_count,
        "avg_pause_duration": round(avg_pause_duration, 2),
        "filler_words": filler_words,
        "confidence_score": 0.85 # baseline proxy
    }
    
    return audio_features, transcript
