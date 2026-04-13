from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Optional
from services.aggregation_service import process_interview

router = APIRouter()

@router.post("/analyze")
async def analyze_interview(
    video_file: UploadFile = File(...),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Endpoint to receive interview video/audio and process it.
    Returns JSON session object with structured analytics.
    """
    if not video_file:
        raise HTTPException(status_code=400, detail="Video file is required.")
        
    try:
        # Save files temporarily and process
        video_content = await video_file.read()
        audio_content = await audio_file.read() if audio_file else None
        
        # We will dispatch processing out to our aggregator
        session_data = await process_interview(video_content, audio_content)
        
        return session_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process: {str(e)}")
