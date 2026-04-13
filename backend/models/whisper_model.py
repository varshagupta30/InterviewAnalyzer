import whisper
import os

_model = None

def get_model():
    global _model
    if _model is None:
        print("==================================================================")
        print("Loading Whisper Model... (First load WILL take 1-3 minutes to download weights)")
        print("Please do NOT close the server. Downloading...")
        print("==================================================================")
        try:
            _model = whisper.load_model("base")
            print("Whisper Model successfully downloaded and loaded into memory! ✅")
        except Exception as e:
            print(f"Failed to load Whisper Model: {e}")
            raise e
    return _model

def transcribe(audio_path: str) -> dict:
    model = get_model()
    result = model.transcribe(audio_path)
    return result
