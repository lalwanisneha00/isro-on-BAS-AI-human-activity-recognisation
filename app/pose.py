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
                num_poses=config.MAX_CREW,
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


def draw_crew_tag(frame, landmarks, name: str, label: str, colour) -> None:
    """Label one crew member on the video, beside their own skeleton.

    With several people in frame a single readout cannot say who is doing
    what, so the name and activity ride next to each body.
    """
    height, width = frame.shape[:2]

    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    if not xs:
        return

    # Anchor above the head, kept inside the frame.
    cx = int(min(max(sum(xs) / len(xs), 60), width - 60))
    top = int(min(ys))
    text = f"{name}  {label.upper()}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)

    x0 = int(min(max(cx - tw // 2 - 9, 6), width - tw - 20))
    y1 = max(th + 14, top - 10)
    y0 = y1 - th - 11

    panel = frame[y0:y1, x0:x0 + tw + 18]
    if panel.size:
        dark = panel.copy()
        dark[:] = (18, 12, 9)
        cv2.addWeighted(dark, 0.8, panel, 0.2, 0, panel)

    cv2.rectangle(frame, (x0, y0), (x0 + tw + 18, y1), colour, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x0 + 3, y1), colour, -1)
    cv2.putText(frame, text, (x0 + 9, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)


def draw_crew_summary(frame, tracks) -> None:
    """Draw the crew count, and an alert strip if anyone has stopped moving."""
    height, width = frame.shape[:2]

    count = len(tracks)
    text = "NO CREW IN FRAME" if count == 0 else (
        f"{count} CREW MEMBER{'S' if count != 1 else ''} TRACKED")
    cv2.putText(frame, text, (18, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                config.COLOR_ACCENT if count else (140, 128, 110), 1, cv2.LINE_AA)

    alerts = [t for t in tracks if t.is_anomaly]
    if not alerts:
        return

    names = ", ".join(t.name for t in alerts)
    warning = f"ANOMALY - NO MOTION: {names}"
    (tw, th), _ = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    x0, y0 = 16, 16
    strip = frame[y0:y0 + th + 18, x0:x0 + tw + 26]
    if strip.size:
        dark = strip.copy()
        dark[:] = (18, 12, 9)
        cv2.addWeighted(dark, 0.82, strip, 0.18, 0, strip)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 26, y0 + th + 18),
                  config.COLOR_ALERT, 2, cv2.LINE_AA)
    cv2.putText(frame, warning, (x0 + 13, y0 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_ALERT, 2, cv2.LINE_AA)
