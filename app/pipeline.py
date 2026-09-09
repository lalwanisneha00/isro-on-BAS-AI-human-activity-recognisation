"""The processing thread that turns raw camera frames into annotated output.

Running this once in its own thread (rather than inside each HTTP stream) means
pose detection happens a single time per frame no matter how many browser tabs
are watching, and later stages - normalisation, classification, logging - all
read from the same frame-by-frame result.
"""

import threading
import time

import numpy as np

from . import config
from .camera import CameraSource
from .demo import DemoSource
from .logbook import ActivityLogger
from .objects import ObjectDetector, roles_present
from .subject import SubjectLock
from .normalize import PoseWindow, normalise
from .pose import (PoseTracker, draw_objects, draw_skeleton,
                   draw_subject_tag)


class ProcessingPipeline:
    """Camera -> pose -> overlay, published as the latest annotated frame."""

    def __init__(self, camera: CameraSource):
        self.camera = camera
        self.tracker = PoseTracker()
        self.demo = DemoSource()
        # Starting straight into Demo Mode means a demo never has to touch
        # the webcam at all, not even to fail opening it.
        self.mode = "demo" if getattr(config, "START_IN_DEMO_MODE", False) else "live"

        self._output = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Exactly one person is followed. See app/subject.py for why.
        self.subject = SubjectLock()
        self.objects = ObjectDetector()
        self.detections = []
        self.logger = ActivityLogger()
        self.pose_detected = False
        self.landmarks = None
        self.normalised = None
        self.process_fps = 0.0
        self.latency_ms = 0.0

    # ------------------------------------------------------------ lifecycle --
    def start(self) -> "ProcessingPipeline":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.tracker.close()
        self.demo.close()
        self.logger.close()

    # --------------------------------------------------------------- output --
    def read(self) -> np.ndarray:
        """Latest annotated frame; falls back to raw camera output."""
        with self._lock:
            if self._output is not None:
                return self._output
        return self.camera.read()

    def set_mode(self, mode: str) -> dict:
        """Switch between the live camera and the recorded demo clip."""
        mode = "demo" if str(mode).lower() == "demo" else "live"
        if mode != self.mode:
            # Close the open log segment so a mode change cannot invent one
            # long activity spanning both sources.
            self.logger.flush()
            self.subject.reset()
            if mode == "demo":
                self.demo.rewind()
            self.mode = mode
        return self.mode_state()

    def mode_state(self) -> dict:
        return {
            "mode": self.mode,
            "demo_available": self.demo.available,
            "demo_detail": self.demo.detail,
            "demo_position": round(self.demo.position, 4),
            "demo_elapsed": round(self.demo.elapsed, 1),
            "demo_duration": round(self.demo.duration, 1),
        }

    @property
    def pose_status(self) -> str:
        if self.mode == "demo":
            return "tracking" if self.pose_detected else "searching"
        if not self.tracker.available:
            return "unavailable"
        return "tracking" if self.pose_detected else "searching"

    @property
    def pose_detail(self) -> str:
        if self.mode == "demo":
            return ("Replaying recorded crew feed" if self.pose_detected
                    else self.demo.detail)
        if not self.tracker.available:
            return self.tracker.error
        if self.pose_detected:
            if self.subject.ignored:
                return (f"Subject locked - ignoring {self.subject.ignored} "
                        f"other{'s' if self.subject.ignored > 1 else ''} in frame")
            return "Subject locked - 33 landmarks"
        if self.subject.locked:
            return "Subject briefly out of view - holding lock"
        return "No crew member in frame"

    def view_state(self) -> dict:
        """How the video is presented, and what the object model is doing."""
        return {
            "video_mode": config.VIDEO_MODE,
            "show_skeleton": config.SHOW_SKELETON,
            "show_objects": config.SHOW_OBJECTS,
            "objects": self.objects.status(),
        }

    def set_view(self, show_skeleton=None, video_mode=None,
                 show_objects=None) -> dict:
        """Change how the video is presented. Never touches classification.

        Toggling the overlay must not disturb the pipeline: the same frames
        are analysed and the same rows are logged either way. Only the drawing
        changes, which is why these flags are read at draw time and nowhere
        else.
        """
        if show_skeleton is not None:
            config.SHOW_SKELETON = bool(show_skeleton)
        if show_objects is not None:
            config.SHOW_OBJECTS = bool(show_objects)
        if video_mode in ("normal", "privacy"):
            config.VIDEO_MODE = video_mode
        return self.view_state()

    def activity(self) -> dict:
        """The monitored subject's activity, and the alert state."""
        state = self.subject.state()
        state["objects"] = [d.as_dict() for d in self.detections]
        state["held"] = sorted({d.name for d in self.detections if d.in_hand})
        state.update({
            "subject_name": config.SUBJECT_NAME,
            "still_seconds": state["still_seconds"] if state["is_anomaly"] else 0.0,
        })
        return state

    def log_view(self, limit: int = 25) -> dict:
        """Recent activity segments plus cumulative time per activity."""
        return {
            "events": self.logger.recent(limit),
            "totals": self.logger.totals(),
            "timeline": self.logger.timeline(),
            "status": self.logger.status(),
            "order": config.ACTIVITY_ORDER,
            "colours": config.ACTIVITY_HEX,
        }

    def telemetry(self) -> dict:
        """Numbers proving normalisation holds steady while raw values move."""
        window = self.subject.window
        pose = window.latest if window else None
        data = {
            "buffer_count": window.count if window else 0,
            "buffer_size": window.size if window else config.WINDOW_SIZE,
            "buffer_span": round(window.span_seconds, 2) if window else 0.0,
            "buffer_ready": window.is_full if window else False,
            "has_pose": pose is not None,
            "subject_name": config.SUBJECT_NAME,
        }
        if pose is None:
            return data

        shoulder = (pose.points[config.LM_LEFT_SHOULDER]
                    + pose.points[config.LM_RIGHT_SHOULDER]) / 2.0
        data.update({
            "torso_length": round(pose.torso_length, 4),
            "hip_x": round(pose.hip_center[0], 4),
            "hip_y": round(pose.hip_center[1], 4),
            "spine_angle": round(pose.spine_angle, 1),
            "confidence": round(pose.confidence, 3),
            "norm_shoulder_x": round(float(shoulder[0]), 4),
            "norm_shoulder_y": round(float(shoulder[1]), 4),
            "norm_points": [[round(float(x), 3), round(float(y), 3)]
                            for x, y in pose.points],
        })
        return data

    # ------------------------------------------------------------ internals --
    def _loop(self) -> None:
        frame_interval = 1.0 / config.TARGET_FPS
        smoothed_fps = 0.0
        last_tick = time.time()

        while self._running:
            started = time.time()

            if self.mode == "demo":
                # Replayed footage carries its own pose track, so a demo never
                # depends on the webcam or on MediaPipe finding a person.
                frame, poses = self.demo.next()
                if frame is None:
                    frame, poses = self.demo.placeholder(), []
            else:
                frame = self.camera.read()
                poses = []
                if self.camera.status == "live":
                    poses = self.tracker.detect(frame)

            # The detector may see several people. Exactly one of them is
            # followed; the rest are discarded here and never reach the
            # window, the classifier or the log.
            height, width = frame.shape[:2]
            candidates, raw_for = [], {}
            for pose_landmarks in poses:
                pose = normalise(pose_landmarks, width, height)
                if pose is None:
                    continue
                raw_for[id(pose)] = pose_landmarks
                candidates.append(pose)

            # Objects are detected on a background thread every few frames;
            # whatever the newest result is gets used. Pose tracking never
            # waits for the slower object model.
            self.objects.submit(frame)
            detections = self.objects.latest()

            lead_landmarks = None
            torso_px = height * 0.25
            if candidates:
                best = max(candidates, key=lambda c: c.torso_length)
                lead_landmarks = raw_for.get(id(best))
                torso_px = best.torso_length * height
            if detections and lead_landmarks is not None:
                detections = self.objects.attach_to_hands(
                    detections, lead_landmarks, width, height, torso_px)
            self.detections = detections

            roles = roles_present(detections)
            held = roles_present(detections, held_only=True)

            chosen = self.subject.update(candidates, roles=roles,
                                         held_roles=held)

            self.pose_detected = chosen is not None
            self.normalised = chosen
            self.landmarks = raw_for.get(id(chosen)) if chosen is not None else None

            self.logger.observe(config.SUBJECT_NAME, self.subject.label,
                                self.subject.confidence)

            # Privacy Mode replaces the video with a plain field, so the
            # skeleton and the analysis stay visible while the crew member
            # does not. Crew privacy is a genuine documented concern in
            # spaceflight, and the monitoring works just as well without
            # anyone being recognisable.
            if config.VIDEO_MODE == "privacy":
                frame = np.full_like(frame, (18, 13, 11))
            elif config.SHOW_OBJECTS:
                draw_objects(frame, self.detections)

            if chosen is not None and self.landmarks is not None:
                colour = config.ACTIVITY_COLORS.get(self.subject.label,
                                                    config.COLOR_ACCENT)
                # In Privacy Mode the skeleton is all there is, so it is drawn
                # whatever the overlay toggle says.
                if config.SHOW_SKELETON or config.VIDEO_MODE == "privacy":
                    frame = draw_skeleton(frame, self.landmarks, colour)
                reading = self.subject.classifier.posture
                posture = ""
                if reading is not None and reading.seated:
                    posture = "Seated"
                draw_subject_tag(frame, self.landmarks, self.subject.label,
                                 self.subject.confidence, colour,
                                 self.subject.ignored, posture)

            with self._lock:
                self._output = frame

            self.latency_ms = round((time.time() - started) * 1000, 1)

            now = time.time()
            delta = now - last_tick
            last_tick = now
            if delta > 0:
                smoothed_fps = (0.9 * smoothed_fps) + (0.1 * (1.0 / delta))
                self.process_fps = round(smoothed_fps, 1)

            remaining = frame_interval - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)
