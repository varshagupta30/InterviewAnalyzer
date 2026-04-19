from transformers import pipeline
import torch

_nlp_pipeline = None

def get_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        print("==================================================================")
        print("Loading Zero-Shot Classification Model (facebook/bart-large-mnli)...")
        print("This is a 1.6GB model and will take a moment on the first run.")
        print("==================================================================")
        
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print("CUDA detected! Running AI model on GPU 🚀")
        else:
            print("CUDA not detected. Running AI model on CPU.")
            
        _nlp_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    return _nlp_pipeline

def evaluate_zero_shot(text: str, candidate_labels: list[str]) -> dict:
    if not text:
        return {"labels": candidate_labels, "scores": [1.0 / len(candidate_labels)] * len(candidate_labels)}
    
    # BART can handle longer text, up to 1024 tokens usually, but we keep it reasonable
    if len(text) > 2000:
        text = text[:2000] 
    
    pipe = get_pipeline()
    result = pipe(text, candidate_labels=candidate_labels)
    return result
