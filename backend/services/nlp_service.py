from models.bert_model import evaluate_zero_shot

def analyze_text(transcript: str) -> dict:
    relevance_labels = ["highly relevant and specific answer", "off-topic or generic answer"]
    coherence_labels = ["clear and coherent explanation", "confusing or rambling explanation"]
    
    relevance_result = evaluate_zero_shot(transcript, candidate_labels=relevance_labels)
    coherence_result = evaluate_zero_shot(transcript, candidate_labels=coherence_labels)
    
    rel_idx = relevance_result["labels"].index("highly relevant and specific answer")
    relevance_score = relevance_result["scores"][rel_idx]
    
    coh_idx = coherence_result["labels"].index("clear and coherent explanation")
    coherence_score = coherence_result["scores"][coh_idx]
    
    # Calculate simplistic NLP features
    words = transcript.split()
    answer_length_score = min(1.0, len(words) / 100.0) # Up to 100 words is a "good" length score proxy
    
    keyword_match_score = 0.7 # We'd need the question for this, assume flat 0.7 for demo
    
    return {
        "relevance_score": round(relevance_score, 2),
        "coherence_score": round(coherence_score, 2),
        "keyword_match_score": round(keyword_match_score, 2),
        "answer_length_score": round(answer_length_score, 2)
    }
