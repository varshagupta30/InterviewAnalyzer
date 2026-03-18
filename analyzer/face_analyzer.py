"""Face and eye-contact behavior feature extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


EYE_CONTACT_THRESHOLD = 0.65
SMILE_RATIO_THRESHOLD = 0.38
TENSE_MOUTH_THRESHOLD = 0.20
NOD_MOVEMENT_THRESHOLD = 0.01


def analyze_face_features(
    face_landmarks: Optional[List[Any]],
    state: Dict[str, Any],
    dt: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract eye-contact, expression, and head movement features."""
    if not face_landmarks:
        return {
            "eye_contact_proxy": 0.0,
            "smile_signal": 0.0,
            "tense_expression_signal": 0.0,
            "eyebrow_activity": 0.0,
            "head_nod_frequency": 0.0,
        }, state

    eye_contact_proxy = _estimate_eye_contact(face_landmarks)
    smile_signal, tense_signal = _estimate_expression(face_landmarks)
    eyebrow_activity = _estimate_eyebrow_activity(face_landmarks)

    nose = face_landmarks[1]
    prev_nose_y = state.get("prev_nose_y")
    if prev_nose_y is None:
        head_nod_frequency = 0.0
    else:
        nod_speed = abs(nose.y - prev_nose_y) / max(dt, 1e-6)
        head_nod_frequency = 1.0 if nod_speed > NOD_MOVEMENT_THRESHOLD else 0.0

    state["prev_nose_y"] = nose.y

    return {
        "eye_contact_proxy": eye_contact_proxy,
        "smile_signal": smile_signal,
        "tense_expression_signal": tense_signal,
        "eyebrow_activity": eyebrow_activity,
        "head_nod_frequency": head_nod_frequency,
    }, state


def summarize_face_features(frame_features: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate face features over sampled frames."""
    return {
        "eye_contact_ratio": _mean([f["eye_contact_proxy"] for f in frame_features]),
        "smile_ratio": _mean([f["smile_signal"] for f in frame_features]),
        "tense_expression_ratio": _mean([f["tense_expression_signal"] for f in frame_features]),
        "eyebrow_activity_mean": _mean([f["eyebrow_activity"] for f in frame_features]),
        "head_nod_frequency_mean": _mean([f["head_nod_frequency"] for f in frame_features]),
    }


def _estimate_eye_contact(face_landmarks: List[Any]) -> float:
    """Estimate eye-contact proxy via iris position centered in each eye."""
    # Eye corners and iris centers from refined face mesh.
    left_outer, left_inner = face_landmarks[33], face_landmarks[133]
    right_inner, right_outer = face_landmarks[362], face_landmarks[263]

    # Iris landmarks exist when refine_face_landmarks=True.
    left_iris = face_landmarks[468] if len(face_landmarks) > 468 else None
    right_iris = face_landmarks[473] if len(face_landmarks) > 473 else None
    if left_iris is None or right_iris is None:
        return 0.0

    left_eye_width = math.dist((left_outer.x, left_outer.y), (left_inner.x, left_inner.y))
    right_eye_width = math.dist((right_outer.x, right_outer.y), (right_inner.x, right_inner.y))
    if left_eye_width == 0.0 or right_eye_width == 0.0:
        return 0.0

    left_center_x = (left_outer.x + left_inner.x) / 2.0
    right_center_x = (right_outer.x + right_inner.x) / 2.0

    left_dev = abs(left_iris.x - left_center_x) / left_eye_width
    right_dev = abs(right_iris.x - right_center_x) / right_eye_width

    mean_deviation = (left_dev + right_dev) / 2.0
    return max(0.0, min(1.0, 1.0 - (mean_deviation / 0.20)))


def _estimate_expression(face_landmarks: List[Any]) -> Tuple[float, float]:
    """Estimate smile and tense expression using mouth geometry ratios."""
    mouth_left = face_landmarks[61]
    mouth_right = face_landmarks[291]
    upper_lip = face_landmarks[13]
    lower_lip = face_landmarks[14]

    mouth_width = math.dist((mouth_left.x, mouth_left.y), (mouth_right.x, mouth_right.y))
    mouth_height = math.dist((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))
    if mouth_width == 0.0:
        return 0.0, 0.0

    ratio = mouth_height / mouth_width

    smile_signal = 1.0 if ratio > SMILE_RATIO_THRESHOLD else 0.0
    tense_signal = 1.0 if ratio < TENSE_MOUTH_THRESHOLD else 0.0
    return smile_signal, tense_signal


def _estimate_eyebrow_activity(face_landmarks: List[Any]) -> float:
    """Estimate eyebrow activity from brow-eye vertical distances."""
    left_brow = face_landmarks[70]
    left_eye_top = face_landmarks[159]
    right_brow = face_landmarks[300]
    right_eye_top = face_landmarks[386]

    left_gap = abs(left_brow.y - left_eye_top.y)
    right_gap = abs(right_brow.y - right_eye_top.y)
    raw = (left_gap + right_gap) / 2.0

    return max(0.0, min(1.0, raw / 0.08))


def _mean(values: List[float]) -> float:
    """Safe mean helper."""
    return float(statistics.fmean(values)) if values else 0.0
