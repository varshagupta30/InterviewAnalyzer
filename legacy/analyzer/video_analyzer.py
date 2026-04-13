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
MIN_VALID_FPS = 5.0
MAX_VALID_FPS = 120.0
MAX_REASONABLE_FRAME_COUNT = 10_000_000
FLOW_NORMALIZER = 6.0

# Quality gate thresholds
MIN_PROCESSED_FRAMES_HARD = 12
MIN_FACE_DETECT_RATIO_HARD = 0.05
MIN_EYE_CONTACT_RATIO_HARD = 0.05
MIN_GLOBAL_MOTION_HARD = 0.02
MIN_FACE_STREAK_HARD = 3

MIN_FACE_AREA_RATIO = 0.015
MAX_FACE_AREA_RATIO = 0.45
MIN_FACE_TEXTURE_VAR = 45.0

MIN_PROCESSED_FRAMES_WARN = 80
MIN_FACE_DETECT_RATIO_WARN = 0.20


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

        # Validate analysis quality and reject unusable recordings.
        quality_warnings = self._assess_quality(feature_vector, metadata)
        metadata["quality_warnings"] = quality_warnings

        return {
            "video_path": str(source),
            "metadata": metadata,
            "feature_vector": feature_vector,
            "scores": scores,
            "feedback": feedback,
        }

    def _assess_quality(self, feature_vector: Dict[str, float], metadata: Dict[str, Any]) -> List[str]:
        """Assess recording quality, raise for unusable input, and return warning messages."""
        warnings: List[str] = []

        frame_count = int(feature_vector.get("frame_count", 0.0))
        face_detect_ratio = float(metadata.get("face_detect_ratio", feature_vector.get("eye_contact_ratio", 0.0)))
        max_face_streak = int(metadata.get("max_face_streak", 0))
        eye_contact_ratio = float(feature_vector.get("eye_contact_ratio", 0.0))
        global_motion = float(feature_vector.get("global_motion_mean", 0.0))

        # Hard-reject clearly invalid/empty submissions.
        if frame_count < MIN_PROCESSED_FRAMES_HARD:
            raise RuntimeError(
                "Recording is too short or unreadable. Please record again with camera on and stable lighting."
            )

        # Reject if a visible face is not detected for a minimum fraction of the session.
        if face_detect_ratio < MIN_FACE_DETECT_RATIO_HARD:
            raise RuntimeError(
                "No visible person detected in the recording. Please keep your face and upper body in frame and retry."
            )

        if max_face_streak < MIN_FACE_STREAK_HARD:
            raise RuntimeError(
                "No stable face/person detected in the recording. Please keep your face clearly visible and retry."
            )

        # Reject if face is weakly detected and there is almost no camera-facing engagement signal.
        if face_detect_ratio < 0.18 and eye_contact_ratio < MIN_EYE_CONTACT_RATIO_HARD and global_motion < MIN_GLOBAL_MOTION_HARD:
            raise RuntimeError(
                "No visible person detected in the recording. Please keep your face and upper body in frame and retry."
            )

        # Non-fatal warnings for low-confidence analyses.
        if frame_count < MIN_PROCESSED_FRAMES_WARN:
            warnings.append(
                "Low frame coverage detected. Results may be less reliable; consider re-recording in better conditions."
            )

        if face_detect_ratio < MIN_FACE_DETECT_RATIO_WARN:
            warnings.append(
                "Face visibility was low for much of the recording. Keep your face centered and well lit for better accuracy."
            )

        if global_motion < 0.03:
            warnings.append(
                "Very low motion was detected. Natural speaking gestures improve analysis confidence."
            )

        return warnings

    def _extract_fallback_feature_vector(self, video_path: str, max_seconds: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Fallback analyzer using OpenCV face detection + motion proxies."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps, total_frames, total_duration, analysis_duration, max_frames = self._resolve_video_timing(capture, max_seconds)

        frame_interval = max(1, int(round(fps / PROCESS_FPS)))

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        face_detected_values: List[float] = []
        eye_contact_values: List[float] = []
        global_motion_values: List[float] = []
        pose_stability_values: List[float] = []
        shoulder_alignment_values: List[float] = []
        head_tilt_values: List[float] = []
        forward_lean_values: List[float] = []
        slouch_values: List[float] = []

        hand_activity_values: List[float] = []
        hand_symmetry_values: List[float] = []
        gesture_variety_values: List[float] = []
        nervous_gesture_values: List[float] = []
        face_touch_values: List[float] = []

        smile_values: List[float] = []
        tense_values: List[float] = []
        eyebrow_values: List[float] = []
        nod_values: List[float] = []

        frame_index = 0
        prev_gray: Optional[np.ndarray] = None
        prev_face_center_y: Optional[float] = None
        baseline_face_area: Optional[float] = None
        face_streak = 0
        max_face_streak = 0
        analysis_start = time.time()

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            if max_frames is not None and frame_index > max_frames:
                break
            if frame_index % frame_interval != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            frame_h, frame_w = frame.shape[:2]
            frame_area = float(max(1, frame_h * frame_w))

            flow_magnitude = None
            global_motion = 0.0
            left_hand_activity = 0.0
            right_hand_activity = 0.0
            face_region_motion = 0.0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                global_motion = min(2.0, float(np.mean(flow_magnitude) / FLOW_NORMALIZER))
                upper_cut = int(frame_h * 0.75)
                left_slice = flow_magnitude[:upper_cut, : frame_w // 2]
                right_slice = flow_magnitude[:upper_cut, frame_w // 2 :]

                if left_slice.size > 0:
                    left_hand_activity = min(1.0, float(np.mean(left_slice) / FLOW_NORMALIZER))
                if right_slice.size > 0:
                    right_hand_activity = min(1.0, float(np.mean(right_slice) / FLOW_NORMALIZER))

            hand_activity = (left_hand_activity + right_hand_activity) / 2.0
            hand_activity_values.append(hand_activity)
            global_motion_values.append(global_motion)

            symmetry = 1.0 - (abs(left_hand_activity - right_hand_activity) / max(left_hand_activity + right_hand_activity, 1e-6))
            hand_symmetry_values.append(max(0.0, min(1.0, symmetry)))

            if len(hand_activity_values) > 1:
                recent = hand_activity_values[-min(20, len(hand_activity_values)) :]
                gesture_variety = min(1.0, (self._std(recent) * 4.0) + ((max(recent) - min(recent)) * 1.5))
                gesture_variety_values.append(gesture_variety)
            else:
                gesture_variety_values.append(0.0)

            nervous_gesture_values.append(1.0 if 0.02 < hand_activity < 0.09 else 0.0)

            if len(faces) > 0:
                # Use largest face for stability.
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, w, h = faces[0]
                x = int(max(0, x))
                y = int(max(0, y))
                w = int(max(1, w))
                h = int(max(1, h))

                face_area_ratio = (w * h) / frame_area

                # Validate face candidate to reduce false positives on empty/noisy frames.
                face_roi = gray[y : min(frame_h, y + h), x : min(frame_w, x + w)]
                eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4) if face_roi.size > 0 else []
                texture_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var()) if face_roi.size > 0 else 0.0
                valid_face = (
                    (MIN_FACE_AREA_RATIO <= face_area_ratio <= MAX_FACE_AREA_RATIO)
                    and (len(eyes) >= 1)
                    and (texture_var >= MIN_FACE_TEXTURE_VAR)
                )

                if not valid_face:
                    face_detected_values.append(0.0)
                    face_streak = 0

                    eye_contact_values.append(0.0)
                    pose_stability_values.append(0.0)
                    shoulder_alignment_values.append(0.0)
                    head_tilt_values.append(0.18)
                    forward_lean_values.append(0.18)
                    slouch_values.append(1.0)
                    smile_values.append(0.0)
                    tense_values.append(1.0)
                    eyebrow_values.append(0.0)
                    nod_values.append(0.0)
                    face_touch_values.append(0.0)

                    prev_gray = gray
                    continue

                face_detected_values.append(1.0)
                face_streak += 1
                max_face_streak = max(max_face_streak, face_streak)

                face_cx = (x + (w / 2.0)) / frame_w
                face_cy = (y + (h / 2.0)) / frame_h
                if baseline_face_area is None:
                    baseline_face_area = face_area_ratio

                # Eye-contact proxy: centered face is treated as camera-facing.
                center_score = max(0.0, 1.0 - ((abs(face_cx - 0.5) / 0.5 + abs(face_cy - 0.37) / 0.63) / 2.0))
                face_size_score = min(1.0, face_area_ratio / 0.10)
                eye_contact_proxy = (0.75 * center_score) + (0.25 * face_size_score)
                eye_contact_values.append(min(1.0, eye_contact_proxy))

                # Posture proxies from face stability and scale drift.
                pose_stability = max(0.0, 1.0 - min(1.0, abs(face_cx - 0.5) * 1.6))
                pose_stability_values.append(pose_stability)
                shoulder_alignment_values.append(pose_stability)

                lean_delta = abs(face_area_ratio - baseline_face_area)
                forward_lean = min(0.25, lean_delta * 2.8)
                forward_lean_values.append(forward_lean)

                # Slouch proxy: face too low in frame is likely hunching.
                slouch_values.append(1.0 if face_cy > 0.50 else 0.0)

                # Head tilt proxy using eye alignment when eyes are detectable.
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
                    e1, e2 = eyes[0], eyes[1]
                    e1_center = (e1[0] + e1[2] / 2.0, e1[1] + e1[3] / 2.0)
                    e2_center = (e2[0] + e2[2] / 2.0, e2[1] + e2[3] / 2.0)
                    dx = max(1e-6, abs(e1_center[0] - e2_center[0]))
                    dy = abs(e1_center[1] - e2_center[1])
                    head_tilt_values.append(min(0.20, dy / dx))
                else:
                    head_tilt_values.append(max(0.0, min(0.20, abs(face_cx - 0.5) * 0.35)))

                # Expression proxies from mouth and brow regions.
                mouth_x1 = x + int(w * 0.20)
                mouth_x2 = x + int(w * 0.80)
                mouth_y1 = y + int(h * 0.58)
                mouth_y2 = y + int(h * 0.95)
                mouth_roi = gray[max(0, mouth_y1) : min(frame_h, mouth_y2), max(0, mouth_x1) : min(frame_w, mouth_x2)]
                if mouth_roi.size > 0:
                    mouth_edges = cv2.Canny(mouth_roi, 40, 120)
                    edge_density = float(np.count_nonzero(mouth_edges)) / float(mouth_edges.size)
                    darkness = 1.0 - (float(np.mean(mouth_roi)) / 255.0)
                    expression_energy = min(1.0, (edge_density * 1.8) + (darkness * 0.6))
                    smile_values.append(max(0.0, min(1.0, (expression_energy - 0.22) / 0.38)))
                    tense_values.append(max(0.0, min(1.0, (0.20 - expression_energy) / 0.20)))
                else:
                    smile_values.append(0.0)
                    tense_values.append(0.0)

                brow_x1 = x + int(w * 0.20)
                brow_x2 = x + int(w * 0.80)
                brow_y1 = y + int(h * 0.12)
                brow_y2 = y + int(h * 0.40)
                brow_roi = gray[max(0, brow_y1) : min(frame_h, brow_y2), max(0, brow_x1) : min(frame_w, brow_x2)]
                if brow_roi.size > 0:
                    sobel_y = cv2.Sobel(brow_roi, cv2.CV_32F, 0, 1, ksize=3)
                    brow_activity = min(1.0, float(np.mean(np.abs(sobel_y))) / 45.0)
                    eyebrow_values.append(brow_activity)
                else:
                    eyebrow_values.append(0.0)

                # Nodding proxy from face vertical movement.
                if prev_face_center_y is None:
                    nod_values.append(0.0)
                else:
                    nod_values.append(1.0 if abs(face_cy - prev_face_center_y) > 0.012 else 0.0)
                prev_face_center_y = face_cy

                # Face-touch proxy from motion concentrated in expanded face region.
                if flow_magnitude is not None:
                    pad_x = int(w * 0.25)
                    pad_y = int(h * 0.25)
                    fx1 = max(0, x - pad_x)
                    fy1 = max(0, y - pad_y)
                    fx2 = min(frame_w, x + w + pad_x)
                    fy2 = min(frame_h, y + h + pad_y)
                    face_flow_region = flow_magnitude[fy1:fy2, fx1:fx2]
                    if face_flow_region.size > 0:
                        face_region_motion = min(1.0, float(np.mean(face_flow_region)) / FLOW_NORMALIZER)
                face_touch_values.append(1.0 if (face_region_motion > (hand_activity * 0.8) and hand_activity > 0.05) else 0.0)
            else:
                face_detected_values.append(0.0)
                face_streak = 0
                eye_contact_values.append(0.0)
                pose_stability_values.append(0.0)
                shoulder_alignment_values.append(0.0)
                head_tilt_values.append(0.18)
                forward_lean_values.append(0.18)
                slouch_values.append(1.0)
                smile_values.append(0.0)
                tense_values.append(1.0)
                eyebrow_values.append(0.0)
                nod_values.append(0.0)
                face_touch_values.append(0.0)

            prev_gray = gray

        analysis_end = time.time()
        capture.release()

        sampled_frames = len(global_motion_values)
        if sampled_frames == 0:
            raise RuntimeError("No frames were sampled from uploaded video. Please retry recording.")

        observed_duration = frame_index / max(fps, 1e-6)
        resolved_total_duration = total_duration if total_duration > 0 else observed_duration
        resolved_analysis_duration = min(analysis_duration, resolved_total_duration) if resolved_total_duration > 0 else analysis_duration

        # Build a stable feature vector contract used by scorer + feedback.
        eye_contact_ratio = self._mean(eye_contact_values)
        upright_ratio = self._mean(pose_stability_values)
        slouch_ratio = self._mean(slouch_values)
        global_motion_mean = self._mean(global_motion_values)
        hand_frequency_mean = self._mean(hand_activity_values)
        hand_symmetry_mean = self._mean(hand_symmetry_values)
        gesture_variety_mean = self._mean(gesture_variety_values)
        nervous_ratio = self._mean(nervous_gesture_values)

        # Dynamic shoulder/spine proxies from stability + lean behavior.
        shoulder_alignment_mean = self._mean(shoulder_alignment_values)
        forward_lean_mean = self._mean(forward_lean_values)
        spine_straightness = max(120.0, 180.0 - ((forward_lean_mean * 180.0) + (slouch_ratio * 35.0)))

        feature_vector: Dict[str, float] = {
            "shoulder_alignment_mean": shoulder_alignment_mean,
            "spine_straightness_mean": spine_straightness,
            "head_tilt_mean": self._mean(head_tilt_values),
            "forward_lean_mean": forward_lean_mean,
            "slouch_ratio": slouch_ratio,
            "upright_posture_ratio": upright_ratio,
            "body_restlessness_mean": global_motion_mean,
            "body_restlessness_std": self._std(global_motion_values),
            "hand_movement_frequency_mean": hand_frequency_mean,
            "gesture_variety_mean": gesture_variety_mean,
            "hand_symmetry_mean": hand_symmetry_mean,
            "nervous_gesture_ratio": nervous_ratio,
            "face_touching_ratio": self._mean(face_touch_values),
            "left_hand_speed_mean": self._mean([max(0.0, v - 0.02) for v in hand_activity_values]),
            "right_hand_speed_mean": self._mean([max(0.0, v - 0.02) for v in hand_activity_values]),
            "hand_movement_frequency_std": self._std(hand_activity_values),
            "eye_contact_ratio": eye_contact_ratio,
            "smile_ratio": self._mean(smile_values),
            "tense_expression_ratio": self._mean(tense_values),
            "eyebrow_activity_mean": self._mean(eyebrow_values),
            "head_nod_frequency_mean": self._mean(nod_values),
            "global_motion_mean": global_motion_mean,
            "global_motion_std": self._std(global_motion_values),
            "frame_count": float(sampled_frames),
        }

        metadata = {
            "source_fps": fps,
            "total_video_duration_seconds": resolved_total_duration,
            "analysis_duration_seconds": resolved_analysis_duration,
            "processed_frames": sampled_frames,
            "processing_wall_time_seconds": round(analysis_end - analysis_start, 2),
            "face_detect_ratio": self._mean(face_detected_values),
            "max_face_streak": max_face_streak,
        }
        return feature_vector, metadata

    def _extract_frame_packets(self, video_path: str, max_seconds: int) -> Tuple[List[FramePacket], Dict[str, Any]]:
        """Read frames, run landmark detection, and compute per-frame feature packets."""
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps, total_frames, total_duration, analysis_duration, max_frames = self._resolve_video_timing(capture, max_seconds)

        frame_interval = max(1, int(round(fps / PROCESS_FPS)))

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
                if max_frames is not None and frame_index > max_frames:
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
            "total_video_duration_seconds": total_duration if total_duration > 0 else (frame_index / max(fps, 1e-6)),
            "analysis_duration_seconds": analysis_duration,
            "processed_frames": len(packets),
            "processing_wall_time_seconds": round(analysis_end - analysis_start, 2),
        }
        return packets, metadata

    def _resolve_video_timing(self, capture: cv2.VideoCapture, max_seconds: int) -> Tuple[float, int, float, float, Optional[int]]:
        """Resolve robust timing parameters across unstable codec metadata."""
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = raw_fps if MIN_VALID_FPS <= raw_fps <= MAX_VALID_FPS else 30.0

        raw_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        valid_frame_count = 0 < raw_total_frames < MAX_REASONABLE_FRAME_COUNT

        total_duration = (raw_total_frames / fps) if valid_frame_count else 0.0
        analysis_duration = min(float(max_seconds), total_duration) if total_duration > 0 else float(max_seconds)
        max_frames = int(analysis_duration * fps) if valid_frame_count else None

        return fps, (raw_total_frames if valid_frame_count else 0), total_duration, analysis_duration, max_frames

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
