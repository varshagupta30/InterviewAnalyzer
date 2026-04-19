import re
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
    
    # Calculate stricter NLP features
    words = transcript.split()
    word_count = len(words)
    
    # Stricter answer length penalty. Too short or too long is penalized.
    if word_count < 20:
        answer_length_score = word_count / 40.0
    elif word_count > 250:
        answer_length_score = max(0.0, 1.0 - ((word_count - 250) / 100.0))
    else:
        answer_length_score = min(1.0, word_count / 100.0)
    
    # Generic tech keywords
    tech_keywords = [
        "api", "database", "python", "javascript", "cloud", "aws", "azure", "gcp", "docker", "kubernetes", 
        "architecture", "scalability", "frontend", "backend", "framework", "agile", "sprint", "ci/cd", 
        "testing", "performance", "optimization", "security", "git", "version control", "deployment",
        "react", "angular", "vue", "node", "sql", "nosql", "microservices", "algorithm", "data structure"
    ]
    
    matched_keywords = sum(1 for kw in tech_keywords if re.search(r'\b' + re.escape(kw) + r'\b', transcript, re.IGNORECASE))
    
    # Score based on keyword hits, but heavily dependent on overall relevance to prevent keyword stuffing
    raw_keyword_score = min(1.0, matched_keywords / 4.0) 
    keyword_match_score = raw_keyword_score * relevance_score
    
    return {
        "relevance_score": round(relevance_score, 2),
        "coherence_score": round(coherence_score, 2),
        "keyword_match_score": round(keyword_match_score, 2),
        "answer_length_score": round(answer_length_score, 2)
    }
