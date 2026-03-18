"""Rule-based scoring engine with ML-ready interface contract."""

from __future__ import annotations

from typing import Dict


# Posture thresholds
POSTURE_UPRIGHT_GOOD = 0.70
SLOUCH_RATIO_BAD = 0.35
SHOULDER_ALIGNMENT_GOOD = 0.70

# Gesture thresholds
GESTURE_FREQUENCY_MIN = 0.30
GESTURE_FREQUENCY_MAX = 1.80
GESTURE_VARIETY_GOOD = 0.45
NERVOUS_GESTURE_BAD = 0.30
FACE_TOUCH_BAD = 0.20

# Face thresholds
EYE_CONTACT_GOOD = 0.65
TENSE_EXPRESSION_BAD = 0.35

# Global movement thresholds
GLOBAL_MOTION_BAD = 2.0

# Weighted final aggregation
WEIGHT_CONFIDENCE = 0.30
WEIGHT_POSTURE = 0.25
WEIGHT_GESTURE = 0.25
WEIGHT_BODY_LANGUAGE = 0.20


def compute_scores(feature_vector: dict) -> dict:
    """
    Input: Dictionary of averaged frame-level features.
    Output: Dictionary of scores (1-10) per dimension.

    TODO: Replace rule-based logic with trained ML model when dataset is ready.
    The feature_vector format should remain stable as the ML interface contract.
    """
    confidence_score = _score_confidence(feature_vector)
    posture_score = _score_posture(feature_vector)
    gesture_score = _score_gesture(feature_vector)
    body_language_score = _score_body_language(feature_vector)

    overall = (
        confidence_score * WEIGHT_CONFIDENCE
        + posture_score * WEIGHT_POSTURE
        + gesture_score * WEIGHT_GESTURE
        + body_language_score * WEIGHT_BODY_LANGUAGE
    )

    return {
        "confidence_score": round(confidence_score, 2),
        "posture_score": round(posture_score, 2),
        "gesture_score": round(gesture_score, 2),
        "body_language_score": round(body_language_score, 2),
        "overall_score": round(_clip_1_to_10(overall), 2),
    }


def _score_confidence(fv: Dict[str, float]) -> float:
    """Score confidence from eye contact, posture steadiness, and controlled motion."""
    eye_contact = fv.get("eye_contact_ratio", 0.0)
    upright = fv.get("upright_posture_ratio", 0.0)
    global_motion = fv.get("global_motion_mean", 0.0)

    eye_part = _scale_ratio(eye_contact, EYE_CONTACT_GOOD)
    upright_part = _scale_ratio(upright, POSTURE_UPRIGHT_GOOD)
    motion_part = 1.0 - min(global_motion / GLOBAL_MOTION_BAD, 1.0)

    return _map_unit_to_score((eye_part * 0.45) + (upright_part * 0.35) + (motion_part * 0.20))


def _score_posture(fv: Dict[str, float]) -> float:
    """Score posture from slouch ratio, shoulder alignment, and forward lean."""
    slouch_ratio = fv.get("slouch_ratio", 1.0)
    shoulder = fv.get("shoulder_alignment_mean", 0.0)
    forward_lean = fv.get("forward_lean_mean", 0.2)

    slouch_part = 1.0 - min(slouch_ratio / SLOUCH_RATIO_BAD, 1.0)
    shoulder_part = _scale_ratio(shoulder, SHOULDER_ALIGNMENT_GOOD)
    lean_part = 1.0 - min(forward_lean / 0.20, 1.0)

    return _map_unit_to_score((slouch_part * 0.45) + (shoulder_part * 0.35) + (lean_part * 0.20))


def _score_gesture(fv: Dict[str, float]) -> float:
    """Score gestures from frequency, variety, symmetry, and nervous penalties."""
    freq = fv.get("hand_movement_frequency_mean", 0.0)
    variety = fv.get("gesture_variety_mean", 0.0)
    symmetry = fv.get("hand_symmetry_mean", 0.0)
    nervous = fv.get("nervous_gesture_ratio", 0.0)
    face_touch = fv.get("face_touching_ratio", 0.0)

    # Frequency is best in a moderate band.
    if freq < GESTURE_FREQUENCY_MIN:
        freq_part = freq / max(GESTURE_FREQUENCY_MIN, 1e-6)
    elif freq > GESTURE_FREQUENCY_MAX:
        freq_part = max(0.0, 1.0 - ((freq - GESTURE_FREQUENCY_MAX) / GESTURE_FREQUENCY_MAX))
    else:
        freq_part = 1.0

    variety_part = _scale_ratio(variety, GESTURE_VARIETY_GOOD)
    symmetry_part = symmetry
    nervous_penalty = min(nervous / NERVOUS_GESTURE_BAD, 1.0)
    touch_penalty = min(face_touch / FACE_TOUCH_BAD, 1.0)

    unit_score = (
        (freq_part * 0.30)
        + (variety_part * 0.25)
        + (symmetry_part * 0.20)
        + ((1.0 - nervous_penalty) * 0.15)
        + ((1.0 - touch_penalty) * 0.10)
    )
    return _map_unit_to_score(unit_score)


def _score_body_language(fv: Dict[str, float]) -> float:
    """Score openness and engagement from posture/face/gesture blended signals."""
    upright = fv.get("upright_posture_ratio", 0.0)
    eye_contact = fv.get("eye_contact_ratio", 0.0)
    eyebrow_activity = fv.get("eyebrow_activity_mean", 0.0)
    tense_expression = fv.get("tense_expression_ratio", 0.0)
    gesture_variety = fv.get("gesture_variety_mean", 0.0)

    unit_score = (
        _scale_ratio(upright, POSTURE_UPRIGHT_GOOD) * 0.25
        + _scale_ratio(eye_contact, EYE_CONTACT_GOOD) * 0.30
        + min(1.0, eyebrow_activity / 0.50) * 0.15
        + (1.0 - min(tense_expression / TENSE_EXPRESSION_BAD, 1.0)) * 0.15
        + _scale_ratio(gesture_variety, GESTURE_VARIETY_GOOD) * 0.15
    )
    return _map_unit_to_score(unit_score)


def _scale_ratio(value: float, good_threshold: float) -> float:
    """Scale ratio-like metric where >= threshold maps near 1.0."""
    if good_threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, value / good_threshold))


def _map_unit_to_score(unit_value: float) -> float:
    """Map [0,1] domain to [1,10] score."""
    return _clip_1_to_10(1.0 + (9.0 * max(0.0, min(1.0, unit_value))))


def _clip_1_to_10(value: float) -> float:
    """Clip final score to interview score range."""
    return max(1.0, min(10.0, value))
