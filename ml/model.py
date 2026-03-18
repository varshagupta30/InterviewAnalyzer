"""ML model interface placeholder for future trained scoring model."""

from __future__ import annotations

from typing import Dict


class InterviewScoreModel:
    """Placeholder class that can be swapped with a trained model later."""

    def __init__(self) -> None:
        """Initialize model resources.

        TODO: Load trained model artifacts when dataset is available.
        """
        self.ready = False

    def predict(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        """Return predicted scores for provided feature vector.

        TODO: Replace this fallback with real model inference.
        """
        _ = feature_vector
        return {
            "confidence_score": 5.0,
            "posture_score": 5.0,
            "gesture_score": 5.0,
            "body_language_score": 5.0,
            "overall_score": 5.0,
        }
