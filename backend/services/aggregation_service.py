import uuid
from datetime import datetime
from services.audio_service import analyze_audio
from services.nlp_service import analyze_text
from services.vision_service import analyze_vision
from services.scoring_service import calculate_scores
import json
import os
import asyncio

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sessions')

async def process_interview(video_content: bytes, audio_content: bytes | None = None) -> dict:
    session_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    # Save the bytes to temporary files for processing
    tmp_video_path = f"tmp_{session_id}.webm"
    with open(tmp_video_path, "wb") as f:
        f.write(video_content)
        
    try:
        # Run processing sequentially or asynchronously
        print(f"Processing session {session_id}...")
        
        # Audio & Transcript Extraction
        audio_features, transcript = await asyncio.to_thread(analyze_audio, tmp_video_path, audio_content)
        
        # NLP Processing
        nlp_analysis = await asyncio.to_thread(analyze_text, transcript["full_text"])
        
        # Vision Processing
        vision_features = await asyncio.to_thread(analyze_vision, tmp_video_path)
        
        # Generate Scores
        final_scores, feedback = await asyncio.to_thread(
            calculate_scores, 
            audio_features, 
            vision_features, 
            nlp_analysis
        )
        
        session_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "transcript": transcript,
            "audio_features": audio_features,
            "vision_features": vision_features,
            "nlp_analysis": nlp_analysis,
            "final_scores": final_scores,
            "feedback": feedback
        }
        
        # Save to JSON
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = f"session_{timestamp.replace(':', '-')}.json"
        with open(os.path.join(DATA_DIR, filename), "w") as f:
            json.dump(session_data, f, indent=2)
            
        return session_data
        
    finally:
        # Cleanup
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
