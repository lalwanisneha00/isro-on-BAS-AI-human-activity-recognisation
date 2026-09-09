"""MediaPipe pose landmark extraction and skeleton rendering.

MediaPipe 1.x dropped the old `mp.solutions.pose` helper, so this module uses
the current Tasks API (`PoseLandmarker`) in VIDEO mode. The model file lives in
models/ and is loaded from disk, so nothing touches the network at runtime.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

from . import config

# 35 bone segments linking the 33 body landmarks.
CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# Landmark indices worth emphasising: wrists, elbows, shoulders, hips.
KEY_JOINTS = {11, 12, 13, 14, 15, 16, 23, 24}

MODEL_ROOT = Path(__file__).resolve().parent.parent


class PoseTracker:
    """Wraps PoseLandmarker; returns landmarks or None, and never raises."""

    def __init__(self):
        self.available = False
        self.error = ""
        self._landmarker = None
        self._last_timestamp_ms = -1
        self._load()

    def _load(self) -> None:
        model_path = MODEL_ROOT / config.POSE_MODEL_PATH
        if not model_path.exists():
            self.error = f"Pose model missing at {config.POSE_MODEL_PATH}"
            return
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=config.POSE_DETECTION_BUDGET,
                min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                output_segmentation_masks=False,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self.available = True
        except Exception as exc:                      # pragma: no cover - env dependent
            self.error = f"Pose model failed to load: {exc}"

    def detect(self, frame_bgr: np.ndarray):
        """Run pose detection on one BGR frame.

        Returns a list of poses, each a list of 33 landmarks - one entry per
        crew member visible. Empty when nobody is in frame or the model is
        unavailable. MediaPipe gives no stable identity across frames, so
        matching a pose to a particular crew member is the tracker's job.
        """
        if not self.available:
            return []

        # VIDEO mode demands strictly increasing timestamps.
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        except Exception:
            return []

        return list(result.pose_landmarks or [])

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass


def draw_skeleton(frame: np.ndarray, landmarks, colour=None) -> np.ndarray:
    """Draw one crew member's 33-point skeleton.

    The skeleton is drawn in the colour of that member's current activity, so
    a glance at the video says who is doing what without reading any text.
    """
    if not landmarks:
        return frame
    if colour is None:
        colour = config.COLOR_ACCENT
    soft = tuple(int(c * 0.55) for c in colour)

    height, width = frame.shape[:2]
    points = []
    for lm in landmarks:
        visible = getattr(lm, "visibility", 1.0) >= config.LANDMARK_VISIBILITY_THRESHOLD
        points.append((int(lm.x * width), int(lm.y * height), visible))

    overlay = frame.copy()

    # Bones first, so joints sit on top of them.
    for start, end in CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        x1, y1, v1 = points[start]
        x2, y2, v2 = points[end]
        bone = colour if (v1 and v2) else soft
        thickness = 2 if (v1 and v2) else 1
        cv2.line(overlay, (x1, y1), (x2, y2), bone, thickness, cv2.LINE_AA)

    for index, (x, y, visible) in enumerate(points):
        if not visible:
            continue
        if index in KEY_JOINTS:
            cv2.circle(overlay, (x, y), 6, colour, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 6, config.COLOR_JOINT, 1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (x, y), 3, config.COLOR_JOINT, -1, cv2.LINE_AA)

    # Blend so the skeleton reads as a HUD layer rather than paint on the video.
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def draw_subject_tag(frame, landmarks, activity: str, confidence: float,
                     colour, ignored: int = 0, posture: str = "") -> None:
    """Label the monitored subject on the video, above their head.

    The tag says who is being followed and what they are doing, so it is
    obvious at a glance that one person was chosen deliberately rather than
    one person happening to be all the detector found.
    """
    if not landmarks:
        return

    height, width = frame.shape[:2]
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    if not xs:
        return

    centre = int(sum(xs) / len(xs))
    top = int(min(ys))

    # Posture first, then the task: "SEATED - LAPTOP WORK".
    described = f"{posture.upper()} - {activity.upper()}" if posture \
        else activity.upper()
    text = f"SUBJECT LOCKED - {described}  {confidence:.0%}"
    scale, thickness = 0.46, 1
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          scale, thickness)

    pad = 8
    x1 = max(4, min(centre - text_w // 2 - pad, width - text_w - 2 * pad - 4))
    y2 = max(text_h + 2 * pad + 4, top - 12)
    y1 = y2 - text_h - 2 * pad
    x2 = x1 + text_w + 2 * pad

    panel = frame.copy()
    cv2.rectangle(panel, (x1, y1), (x2, y2), (14, 10, 8), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad, y2 - pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Say plainly when other people are present but deliberately not followed.
    if ignored:
        note = f"{ignored} other{'s' if ignored > 1 else ''} in frame - not tracked"
        cv2.putText(frame, note, (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, config.COLOR_ACCENT, 1, cv2.LINE_AA)


def draw_objects(frame, detections) -> None:
    """Outline the whitelisted objects, named for what they stand in for.

    Only objects on the task whitelist ever reach here, so nothing outside
    the mission's vocabulary is ever drawn on the console.
    """
    if not detections:
        return

    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection.box)
        # An object in a hand is what actually changes the classification, so
        # it is the one drawn brightly; everything else stays quiet.
        colour = config.COLOR_ACCENT if detection.in_hand else (120, 110, 96)
        thickness = 2 if detection.in_hand else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

        label = detection.name.upper()
        if detection.in_hand:
            label += "  IN HAND"
        scale = 0.42
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

        ty = max(th + 8, y1 - 4)
        cv2.rectangle(frame, (x1, ty - th - 7), (x1 + tw + 10, ty + 3),
                      (14, 10, 8), -1)
        cv2.putText(frame, label, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 1, cv2.LINE_AA)
