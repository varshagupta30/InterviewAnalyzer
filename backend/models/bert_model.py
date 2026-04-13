from transformers import pipeline

_nlp_pipeline = None

def get_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        print("Loading BERT Model... (This will take a moment on first run)")
        _nlp_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _nlp_pipeline

def extract_sentiment(text: str) -> dict:
    if not text:
        return {"label": "NEUTRAL", "score": 0.5}
    if len(text) > 500:
        text = text[:500] 
    
    pipe = get_pipeline()
    result = pipe(text)[0]
    return result
