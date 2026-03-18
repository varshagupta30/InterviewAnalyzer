"""Core video analysis pipeline using OpenCV + MediaPipe Holistic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics
import time

import cv2
import mediapipe as mp
import numpy as np

from .face_analyzer import analyze_face_features, summarize_face_features
from .feedback import generate_feedback
from .gesture_analyzer import analyze_gesture_features, summarize_gesture_features
from .pose_analyzer import analyze_pose_features, summarize_pose_features
from .scorer import compute_scores


PROCESS_FPS = 10
MAX_ANALYSIS_SECONDS = 60
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


@dataclass
class FramePacket:
    """Container for per-frame raw analyzer outputs."""

    pose: Dict[str, float]
    gesture: Dict[str, float]
    face: Dict[str, float]
    global_motion: float


class VideoAnalyzer:
    """Run full interview analysis and return scores + feedback."""

    def __init__(self) -> None:
        """Initialize MediaPipe references and run-state holders."""
        self.has_mediapipe_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")
        self.mp_holistic = mp.solutions.holistic if self.has_mediapipe_solutions else None

    def analyze_video(self, video_path: str, max_seconds: int = MAX_ANALYSIS_SECONDS) -> Dict[str, Any]:
        """Analyze one interview response video and return report payload."""
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Primary path: MediaPipe Holistic when available.
        if self.has_mediapipe_solutions:
            frame_packets, metadata = self._extract_frame_packets(str(source), max_seconds=max_seconds)
            if not frame_packets:
                raise RuntimeError("No valid frames were analyzed. Check lighting and camera framing.")

            # Aggregate frame-level features into stable feature vector for rule-based/ML scoring.
            feature_vector = self._build_feature_vector(frame_packets)
            metadata["analyzer_mode"] = "mediapipe_holistic"
        else:
            # Fallback path: OpenCV-only coarse features so the app remains runnable.
            feature_vector, metadata = self._extract_fallback_feature_vector(str(source), max_seconds=max_seconds)
            metadata["analyzer_mode"] = "opencv_fallback"
            metadata["analyzer_note"] = (
                "mediapipe.solutions not available in current environment; "
                "using fallback analyzer."
            )

        # Compute per-dimension interview scores.
        scores = compute_scores(feature_vector)

        # Produce candidate-friendly recommendations.
        feedback = generate_feedback(feature_vector, scores)

        return {
            "video_path": str(source),
            "metadata": metadata,
            "feature_vector": feature_vector,
            "scores": scores,
            "feedback": feedback,
        }

    def _extract_fallback_feature_vector(self, video_path: str, max_seconds: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Fallback analyzer using OpenCV face detection + motion proxies."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_duration = total_frames / fps if total_frames else 0.0
        analysis_duration = min(float(max_seconds), total_duration if total_duration else float(max_seconds))

        frame_interval = max(1, int(round(fps / PROCESS_FPS)))
        max_frames = int(analysis_duration * fps)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        face_detected_values: List[float] = []
        eye_contact_values: List[float] = []
        motion_values: List[float] = []
        pose_stability_values: List[float] = []

        frame_index = 0
        prev_gray: Optional[np.ndarray] = None
        analysis_start = time.time()

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            if max_frames and frame_index > max_frames:
                break
            if frame_index % frame_interval != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(faces) > 0:
                face_detected_values.append(1.0)
                x, y, w, h = faces[0]
                face_cx = (x + (w / 2.0)) / frame.shape[1]
                face_cy = (y + (h / 2.0)) / frame.shape[0]

                # Eye-contact proxy: centered face is treated as camera-facing.
                eye_contact_proxy = max(0.0, 1.0 - ((abs(face_cx - 0.5) + abs(face_cy - 0.35)) / 0.8))
                eye_contact_values.append(min(1.0, eye_contact_proxy))

                # Posture proxy: stable face center over time.
                if len(pose_stability_values) == 0:
                    pose_stability_values.append(1.0)
                else:
                    pose_stability_values.append(max(0.0, 1.0 - (abs(face_cx - 0.5) / 0.5)))
            else:
                face_detected_values.append(0.0)
                eye_contact_values.append(0.0)
                pose_stability_values.append(0.0)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_values.append(float(np.mean(magnitude)))
            else:
                motion_values.append(0.0)

            prev_gray = gray

        analysis_end = time.time()
        capture.release()

        # Build a stable feature vector contract used by scorer + feedback.
        eye_contact_ratio = self._mean(eye_contact_values)
        upright_ratio = self._mean(pose_stability_values)
        slouch_ratio = max(0.0, 1.0 - upright_ratio)
        global_motion_mean = self._mean(motion_values)

        feature_vector: Dict[str, float] = {
            "shoulder_alignment_mean": upright_ratio,
            "spine_straightness_mean": 160.0 + (20.0 * upright_ratio),
            "head_tilt_mean": max(0.0, 0.12 - (0.10 * upright_ratio)),
            "forward_lean_mean": max(0.0, 0.12 - (0.10 * upright_ratio)),
            "slouch_ratio": slouch_ratio,
            "upright_posture_ratio": upright_ratio,
            "body_restlessness_mean": global_motion_mean,
            "body_restlessness_std": self._std(motion_values),
            "hand_movement_frequency_mean": global_motion_mean,
            "gesture_variety_mean": min(1.0, global_motion_mean / 2.0),
            "hand_symmetry_mean": 0.5,
            "nervous_gesture_ratio": 0.15 if global_motion_mean < 0.8 else 0.30,
            "face_touching_ratio": 0.0,
            "left_hand_speed_mean": global_motion_mean / 2.0,
            "right_hand_speed_mean": global_motion_mean / 2.0,
            "hand_movement_frequency_std": self._std(motion_values),
            "eye_contact_ratio": eye_contact_ratio,
            "smile_ratio": 0.25,
            "tense_expression_ratio": 0.20,
            "eyebrow_activity_mean": 0.30,
            "head_nod_frequency_mean": 0.10,
            "global_motion_mean": global_motion_mean,
            "global_motion_std": self._std(motion_values),
            "frame_count": float(len(motion_values)),
        }

        metadata = {
            "source_fps": fps,
            "total_video_duration_seconds": total_duration,
            "analysis_duration_seconds": analysis_duration,
            "processed_frames": len(motion_values),
            "processing_wall_time_seconds": round(analysis_end - analysis_start, 2),
            "face_detect_ratio": self._mean(face_detected_values),
        }
        return feature_vector, metadata

    def _extract_frame_packets(self, video_path: str, max_seconds: int) -> Tuple[List[FramePacket], Dict[str, Any]]:
        """Read frames, run landmark detection, and compute per-frame feature packets."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_duration = total_frames / fps if total_frames else 0.0
        analysis_duration = min(float(max_seconds), total_duration if total_duration else float(max_seconds))

        frame_interval = max(1, int(round(fps / PROCESS_FPS)))
        max_frames = int(analysis_duration * fps)

        packets: List[FramePacket] = []
        frame_index = 0

        # State objects let sub-analyzers compute velocities/frequencies between frames.
        pose_state: Dict[str, Any] = {}
        gesture_state: Dict[str, Any] = {}
        face_state: Dict[str, Any] = {}

        with self.mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            refine_face_landmarks=True,
        ) as holistic:
            analysis_start = time.time()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                if max_frames and frame_index > max_frames:
                    break
                if frame_index % frame_interval != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                frame_height, frame_width = frame.shape[:2]
                dt = 1.0 / PROCESS_FPS

                pose_features, pose_state = analyze_pose_features(
                    result.pose_landmarks.landmark if result.pose_landmarks else None,
                    frame_width,
                    frame_height,
                    pose_state,
                    dt,
                )
                gesture_features, gesture_state = analyze_gesture_features(
                    result.left_hand_landmarks.landmark if result.left_hand_landmarks else None,
                    result.right_hand_landmarks.landmark if result.right_hand_landmarks else None,
                    result.face_landmarks.landmark if result.face_landmarks else None,
                    gesture_state,
                    dt,
                )
                face_features, face_state = analyze_face_features(
                    result.face_landmarks.landmark if result.face_landmarks else None,
                    face_state,
                    dt,
                )

                global_motion = (
                    pose_features.get("body_restlessness", 0.0)
                    + gesture_features.get("hand_movement_frequency", 0.0)
                    + face_features.get("head_nod_frequency", 0.0)
                ) / 3.0

                packets.append(
                    FramePacket(
                        pose=pose_features,
                        gesture=gesture_features,
                        face=face_features,
                        global_motion=global_motion,
                    )
                )

            analysis_end = time.time()

        capture.release()

        metadata = {
            "source_fps": fps,
            "total_video_duration_seconds": total_duration,
            "analysis_duration_seconds": analysis_duration,
            "processed_frames": len(packets),
            "processing_wall_time_seconds": round(analysis_end - analysis_start, 2),
        }
        return packets, metadata

    def _build_feature_vector(self, packets: List[FramePacket]) -> Dict[str, float]:
        """Merge pose/gesture/face summaries into one ML-ready feature vector."""
        pose_features = [packet.pose for packet in packets]
        gesture_features = [packet.gesture for packet in packets]
        face_features = [packet.face for packet in packets]

        pose_summary = summarize_pose_features(pose_features)
        gesture_summary = summarize_gesture_features(gesture_features)
        face_summary = summarize_face_features(face_features)

        global_motion_values = [packet.global_motion for packet in packets]

        feature_vector: Dict[str, float] = {}
        feature_vector.update(pose_summary)
        feature_vector.update(gesture_summary)
        feature_vector.update(face_summary)

        # Global confidence signal from combined movement behavior.
        feature_vector["global_motion_mean"] = self._mean(global_motion_values)
        feature_vector["global_motion_std"] = self._std(global_motion_values)

        # Stable contract keys useful for model handoff later.
        feature_vector["frame_count"] = float(len(packets))
        return feature_vector

    @staticmethod
    def _mean(values: List[float]) -> float:
        """Return arithmetic mean with empty-list safety."""
        return float(statistics.fmean(values)) if values else 0.0

    @staticmethod
    def _std(values: List[float]) -> float:
        """Return population std-dev with small-list safety."""
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
