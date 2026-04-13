import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'weights.json')

def load_weights():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {
        "confidence": {"audio": 0.4, "vision": 0.4, "nlp": 0.2},
        "communication": {"audio": 0.5, "nlp": 0.5},
        "technical": {"nlp": 1.0}
    }

def calculate_scores(audio_features: dict, vision_features: dict, nlp_analysis: dict) -> tuple[dict, list]:
    weights = load_weights()
    
    # Normalize features to 0-1
    # Audio norm
    audio_score = audio_features.get("confidence_score", 0.0)
    audio_comm_score = max(0.0, 1.0 - (audio_features.get("pause_count", 0) * 0.05))
    
    # Vision norm
    vision_score = vision_features.get("eye_contact_score", 0.0)
    
    # NLP norm
    nlp_conf_score = nlp_analysis.get("coherence_score", 0.0)
    nlp_comm_score = nlp_analysis.get("relevance_score", 0.0)
    nlp_tech_score = nlp_analysis.get("keyword_match_score", 0.0)
    
    # Apply weights
    conf_cfg = weights.get("confidence", {})
    confidence = (
        audio_score * conf_cfg.get("audio", 0) +
        vision_score * conf_cfg.get("vision", 0) +
        nlp_conf_score * conf_cfg.get("nlp", 0)
    )
    
    comm_cfg = weights.get("communication", {})
    communication = (
        audio_comm_score * comm_cfg.get("audio", 0) +
        nlp_comm_score * comm_cfg.get("nlp", 0)
    )
    
    tech_cfg = weights.get("technical", {})
    technical = nlp_tech_score * tech_cfg.get("nlp", 0)
    
    overall = (confidence + communication + technical) / 3.0
    
    final_scores = {
        "confidence": round(confidence, 2),
        "communication": round(communication, 2),
        "technical": round(technical, 2),
        "overall": round(overall, 2)
    }
    
    feedback = []
    if audio_features.get("pause_count", 0) > 5:
        feedback.append("Try to reduce the number of long pauses.")
    if vision_features.get("eye_contact_score", 1.0) < 0.6:
        feedback.append("Maintain better eye contact with the camera.")
    if nlp_analysis.get("relevance_score", 1.0) < 0.6:
        feedback.append("Make sure your answers stay relevant to the topic.")
    
    if not feedback:
        feedback.append("Great job! Your performance was solid across the board.")
        
    return final_scores, feedback
