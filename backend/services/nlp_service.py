from models.bert_model import extract_sentiment

def analyze_text(transcript: str) -> dict:
    sentiment = extract_sentiment(transcript)
    
    # Calculate simplistic NLP features
    words = transcript.split()
    answer_length_score = min(1.0, len(words) / 100.0) # Up to 100 words is a "good" length score proxy
    
    keyword_match_score = 0.7 # We'd need the question for this, assume flat 0.7 for demo
    
    # Use sentiment score as coherence/relevance proxy
    val = sentiment["score"] if sentiment["label"] == "POSITIVE" else (1.0 - sentiment["score"])
    
    relevance_score = val
    coherence_score = min(1.0, val + 0.1) # Boost coherence slightly over raw sentiment
    
    return {
        "relevance_score": round(relevance_score, 2),
        "coherence_score": round(coherence_score, 2),
        "keyword_match_score": round(keyword_match_score, 2),
        "answer_length_score": round(answer_length_score, 2)
    }
