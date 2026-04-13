from fastapi import APIRouter, HTTPException
import os
import json

router = APIRouter()
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sessions')

@router.get("/sessions")
def get_sessions():
    sessions = []
    if not os.path.exists(DATA_DIR):
        return sessions
        
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "final_score": data.get("final_scores", {}).get("overall", 0)
                })
    return sessions

@router.get("/session/{session_id}")
def get_session(session_id: str):
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=404, detail="Session not found")
        
    for filename in os.listdir(DATA_DIR):
        if filename.startswith(f"session_{session_id}") or filename == f"{session_id}.json":
            # Wait, our spec says format is session_<timestamp>.json and session_id inside
            # Let's just find the file. Better yet, we can name the file session_{session_id}.json for direct lookup.
            pass
            
    # For robust lookup
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get("session_id") == session_id:
                    return data
                    
    raise HTTPException(status_code=404, detail="Session not found")
