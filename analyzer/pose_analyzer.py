"""Posture and body-language feature extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import statistics


SHOULDER_TILT_THRESHOLD = 0.04
SLOUCH_ANGLE_THRESHOLD = 160.0
FORWARD_LEAN_THRESHOLD = 0.08
UPRIGHT_POSTURE_MIN_RATIO = 0.65


def analyze_pose_features(
    pose_landmarks: Optional[List[Any]],
    frame_width: int,
    frame_height: int,
    state: Dict[str, Any],
    dt: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract posture features from pose landmarks for one frame."""
    if not pose_landmarks:
        return {
            "shoulder_alignment": 0.0,
            "spine_straightness": 0.0,
            "head_tilt": 0.0,
            "forward_lean": 0.0,
            "slouch_detected": 1.0,
            "upright_posture_signal": 0.0,
            "body_restlessness": 0.0,
        }, state

    # Required key landmarks from MediaPipe pose model.
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]
    nose = pose_landmarks[0]

    shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2.0, (left_shoulder.y + right_shoulder.y) / 2.0)
    hip_center = ((left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0)

    # Shoulder alignment quality: 1 means shoulders nearly level.
    shoulder_delta = abs(left_shoulder.y - right_shoulder.y)
    shoulder_alignment = max(0.0, 1.0 - (shoulder_delta / SHOULDER_TILT_THRESHOLD))
    shoulder_alignment = min(1.0, shoulder_alignment)

    # Spine angle relative to vertical axis (180 means perfectly vertical in this formulation).
    spine_straightness = _compute_spine_straightness(shoulder_center, hip_center)

    # Head tilt proxy from horizontal nose offset around shoulder center.
    head_tilt = abs(nose.x - shoulder_center[0])

    # Forward lean proxy: nose too far ahead of shoulder center on x-axis.
    forward_lean = abs(nose.x - shoulder_center[0])

    slouch_detected = 1.0 if spine_straightness < SLOUCH_ANGLE_THRESHOLD else 0.0
    upright_posture_signal = 1.0 if (slouch_detected == 0.0 and shoulder_delta < SHOULDER_TILT_THRESHOLD) else 0.0

    # Restlessness from center shift across sampled frames.
    prev_center = state.get("prev_shoulder_center")
    if prev_center is None:
        body_restlessness = 0.0
    else:
        body_restlessness = math.dist(prev_center, shoulder_center) / max(dt, 1e-6)

    state["prev_shoulder_center"] = shoulder_center

    return {
        "shoulder_alignment": shoulder_alignment,
        "spine_straightness": spine_straightness,
        "head_tilt": head_tilt,
        "forward_lean": forward_lean,
        "slouch_detected": slouch_detected,
        "upright_posture_signal": upright_posture_signal,
        "body_restlessness": body_restlessness,
    }, state


def summarize_pose_features(frame_features: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate pose features over all sampled frames."""
    return {
        "shoulder_alignment_mean": _mean([f["shoulder_alignment"] for f in frame_features]),
        "spine_straightness_mean": _mean([f["spine_straightness"] for f in frame_features]),
        "head_tilt_mean": _mean([f["head_tilt"] for f in frame_features]),
        "forward_lean_mean": _mean([f["forward_lean"] for f in frame_features]),
        "slouch_ratio": _mean([f["slouch_detected"] for f in frame_features]),
        "upright_posture_ratio": _mean([f["upright_posture_signal"] for f in frame_features]),
        "body_restlessness_mean": _mean([f["body_restlessness"] for f in frame_features]),
        "body_restlessness_std": _std([f["body_restlessness"] for f in frame_features]),
    }


def _compute_spine_straightness(shoulder_center: Tuple[float, float], hip_center: Tuple[float, float]) -> float:
    """Return spine angle (degrees) where larger values indicate straighter posture."""
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    # Compare torso vector to vertical vector.
    torso_angle = math.degrees(math.atan2(dy, dx))
    vertical_angle = -90.0
    diff = abs(torso_angle - vertical_angle)
    diff = min(diff, 360.0 - diff)

    # Convert to straightness-style score around 180 ideal (per project threshold style).
    return max(0.0, 180.0 - diff)


def _mean(values: List[float]) -> float:
    """Safe mean helper."""
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    """Safe std helper."""
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
