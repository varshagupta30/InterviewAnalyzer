"""Hand gesture feature extraction and behavioral scoring signals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


GESTURE_FREQUENCY_MIN = 0.3
NERVOUSE_GESTURE_SPEED_MAX = 0.03
FACE_TOUCH_DISTANCE_THRESHOLD = 0.05


def analyze_gesture_features(
    left_hand: Optional[List[Any]],
    right_hand: Optional[List[Any]],
    face_landmarks: Optional[List[Any]],
    state: Dict[str, Any],
    dt: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract gesture features from both hands and face proximity signals."""
    left_center = _hand_center(left_hand)
    right_center = _hand_center(right_hand)

    prev_left = state.get("prev_left_center")
    prev_right = state.get("prev_right_center")

    left_speed = _speed(prev_left, left_center, dt)
    right_speed = _speed(prev_right, right_center, dt)

    hand_movement_frequency = (left_speed + right_speed) / 2.0
    gesture_variety = _gesture_variety(left_hand, right_hand)
    hand_symmetry = _symmetry_score(left_speed, right_speed)
    nervous_gesture = _nervous_signal(left_speed, right_speed)
    face_touching = _face_touch_signal(left_hand, right_hand, face_landmarks)

    state["prev_left_center"] = left_center
    state["prev_right_center"] = right_center

    return {
        "left_hand_speed": left_speed,
        "right_hand_speed": right_speed,
        "hand_movement_frequency": hand_movement_frequency,
        "gesture_variety": gesture_variety,
        "hand_symmetry": hand_symmetry,
        "nervous_gesture_signal": nervous_gesture,
        "face_touching_signal": face_touching,
    }, state


def summarize_gesture_features(frame_features: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate gesture frame metrics to summary features."""
    return {
        "hand_movement_frequency_mean": _mean([f["hand_movement_frequency"] for f in frame_features]),
        "gesture_variety_mean": _mean([f["gesture_variety"] for f in frame_features]),
        "hand_symmetry_mean": _mean([f["hand_symmetry"] for f in frame_features]),
        "nervous_gesture_ratio": _mean([f["nervous_gesture_signal"] for f in frame_features]),
        "face_touching_ratio": _mean([f["face_touching_signal"] for f in frame_features]),
        "left_hand_speed_mean": _mean([f["left_hand_speed"] for f in frame_features]),
        "right_hand_speed_mean": _mean([f["right_hand_speed"] for f in frame_features]),
        "hand_movement_frequency_std": _std([f["hand_movement_frequency"] for f in frame_features]),
    }


def _hand_center(hand_landmarks: Optional[List[Any]]) -> Optional[Tuple[float, float]]:
    """Return mean hand center for 21 landmarks if hand is detected."""
    if not hand_landmarks:
        return None

    x = sum(p.x for p in hand_landmarks) / len(hand_landmarks)
    y = sum(p.y for p in hand_landmarks) / len(hand_landmarks)
    return (x, y)


def _speed(prev: Optional[Tuple[float, float]], curr: Optional[Tuple[float, float]], dt: float) -> float:
    """Compute hand center speed in normalized frame units/sec."""
    if prev is None or curr is None:
        return 0.0
    return math.dist(prev, curr) / max(dt, 1e-6)


def _gesture_variety(left_hand: Optional[List[Any]], right_hand: Optional[List[Any]]) -> float:
    """Estimate gesture variety from hand openness and two-hand usage."""
    if not left_hand and not right_hand:
        return 0.0

    active_signals: List[float] = []
    for hand in [left_hand, right_hand]:
        if hand:
            # Open palm proxy: average fingertip spread around wrist.
            wrist = hand[0]
            tips = [hand[4], hand[8], hand[12], hand[16], hand[20]]
            spread = sum(math.dist((wrist.x, wrist.y), (tip.x, tip.y)) for tip in tips) / len(tips)
            active_signals.append(spread)

    if not active_signals:
        return 0.0

    # Normalize to 0..1 practical range.
    raw = statistics.fmean(active_signals)
    return max(0.0, min(1.0, raw / 0.25))


def _symmetry_score(left_speed: float, right_speed: float) -> float:
    """Higher score means both hands are used with similar intensity."""
    denom = max(left_speed + right_speed, 1e-6)
    asymmetry = abs(left_speed - right_speed) / denom
    return max(0.0, min(1.0, 1.0 - asymmetry))


def _nervous_signal(left_speed: float, right_speed: float) -> float:
    """Detect nervous micro-motions from frequent low-amplitude hand speed."""
    mean_speed = (left_speed + right_speed) / 2.0
    return 1.0 if 0.0 < mean_speed < NERVOUSE_GESTURE_SPEED_MAX else 0.0


def _face_touch_signal(
    left_hand: Optional[List[Any]],
    right_hand: Optional[List[Any]],
    face_landmarks: Optional[List[Any]],
) -> float:
    """Detect likely face touching when fingertip is close to nose/cheeks."""
    if not face_landmarks:
        return 0.0

    nose = face_landmarks[1]
    left_cheek = face_landmarks[234]
    right_cheek = face_landmarks[454]
    face_points = [(nose.x, nose.y), (left_cheek.x, left_cheek.y), (right_cheek.x, right_cheek.y)]

    for hand in [left_hand, right_hand]:
        if not hand:
            continue
        fingertips = [hand[4], hand[8], hand[12], hand[16], hand[20]]
        for tip in fingertips:
            for face_point in face_points:
                if math.dist((tip.x, tip.y), face_point) < FACE_TOUCH_DISTANCE_THRESHOLD:
                    return 1.0
    return 0.0


def _mean(values: List[float]) -> float:
    """Safe mean helper."""
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    """Safe std helper."""
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
