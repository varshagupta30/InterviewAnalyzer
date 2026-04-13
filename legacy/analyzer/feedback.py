"""Feedback generator for strengths, improvements, and mistakes."""

from __future__ import annotations

from typing import Dict, List


def generate_feedback(feature_vector: Dict[str, float], scores: Dict[str, float]) -> Dict[str, List[str]]:
    """Generate actionable interview feedback sections."""
    strengths: List[str] = []
    improvements: List[str] = []
    mistakes: List[str] = []

    # Positive observations.
    if feature_vector.get("upright_posture_ratio", 0.0) >= 0.65:
        strengths.append("Maintained upright posture for most of the response.")
    if feature_vector.get("eye_contact_ratio", 0.0) >= 0.60:
        strengths.append("Maintained strong eye-contact behavior.")
    if feature_vector.get("gesture_variety_mean", 0.0) >= 0.40:
        strengths.append("Used hand gestures with good variety and openness.")
    if feature_vector.get("hand_symmetry_mean", 0.0) >= 0.55:
        strengths.append("Balanced use of both hands while speaking.")

    # Improvement suggestions.
    if feature_vector.get("head_tilt_mean", 0.0) > 0.08:
        improvements.append("Reduce repeated head tilting to project a steadier presence.")
    if feature_vector.get("forward_lean_mean", 0.0) > 0.09:
        improvements.append("Avoid leaning forward too much; keep your spine vertically aligned.")
    if feature_vector.get("tense_expression_ratio", 0.0) > 0.25:
        improvements.append("Relax facial tension and maintain a more neutral-to-positive expression.")
    if feature_vector.get("hand_movement_frequency_mean", 0.0) < 0.30:
        improvements.append("Use slightly more purposeful hand gestures to improve engagement.")

    # Clear mistakes to rectify.
    if feature_vector.get("slouch_ratio", 0.0) > 0.35:
        mistakes.append("Frequent slouching detected during the answer.")
    if feature_vector.get("eye_contact_ratio", 0.0) < 0.45:
        mistakes.append("Avoiding eye-contact was detected in many frames.")
    if feature_vector.get("face_touching_ratio", 0.0) > 0.20:
        mistakes.append("Frequent face-touching detected (possible nervous gesture).")
    if feature_vector.get("nervous_gesture_ratio", 0.0) > 0.30:
        mistakes.append("High frequency micro-gestures suggest fidgeting behavior.")

    # Ensure all sections are populated with beginner-friendly defaults.
    if not strengths:
        strengths.append("Completed the full response session successfully.")
    if not improvements:
        improvements.append("Keep practicing with varied question difficulty to improve consistency.")
    if not mistakes:
        mistakes.append("No major critical body-language mistakes were detected.")

    return {
        "strengths": strengths,
        "improvements": improvements,
        "mistakes": mistakes,
        "score_snapshot": [
            f"Confidence: {scores.get('confidence_score', 0):.2f}/10",
            f"Posture: {scores.get('posture_score', 0):.2f}/10",
            f"Gesture: {scores.get('gesture_score', 0):.2f}/10",
            f"Body language: {scores.get('body_language_score', 0):.2f}/10",
            f"Overall: {scores.get('overall_score', 0):.2f}/10",
        ],
    }
