"""Sample dataset helper for early testing without a real labeled dataset."""

from __future__ import annotations

from typing import Dict, List


def get_sample_feature_vectors() -> List[Dict[str, float]]:
    """Return a tiny rule-test dataset for local experiments."""
    return [
        {
            "eye_contact_ratio": 0.78,
            "upright_posture_ratio": 0.82,
            "slouch_ratio": 0.10,
            "gesture_variety_mean": 0.52,
            "nervous_gesture_ratio": 0.08,
            "face_touching_ratio": 0.02,
        },
        {
            "eye_contact_ratio": 0.40,
            "upright_posture_ratio": 0.35,
            "slouch_ratio": 0.55,
            "gesture_variety_mean": 0.18,
            "nervous_gesture_ratio": 0.42,
            "face_touching_ratio": 0.30,
        },
    ]
