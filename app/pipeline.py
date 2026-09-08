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
from .tracking import CrewTracker
from .normalize import PoseWindow, normalise
from .pose import (PoseTracker, draw_crew_summary, draw_crew_tag,
                   draw_skeleton)


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

        self.crew = CrewTracker()
        self.logger = ActivityLogger()
        self._known_crew = set()
        self.pose_detected = False
        self.landmarks = None
        self.normalised = None
        self.crew_count = 0
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
            self.crew.reset()
            self._known_crew = set()
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
            return "Crew member locked - 33 landmarks"
        return "No crew member in frame"

    def activity(self) -> dict:
        """Every tracked crew member, with the alert state across the crew."""
        crew = self.crew.as_list()
        anomalies = [c for c in crew if c["is_anomaly"]]
        return {
            "crew": crew,
            "crew_count": len(crew),
            "visible_count": self.crew.visible_count,
            "max_crew": config.MAX_CREW,
            "is_anomaly": bool(anomalies),
            "anomaly_crew": [c["name"] for c in anomalies],
            "still_seconds": max((c["still_seconds"] for c in anomalies), default=0.0),
        }

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
        # Telemetry follows the first tracked crew member, which is the one
        # the normalisation panel is demonstrating.
        lead = next(iter(self.crew.tracks), None)
        window = lead.window if lead else None
        pose = window.latest if window else None
        data = {
            "buffer_count": window.count if window else 0,
            "buffer_size": window.size if window else config.WINDOW_SIZE,
            "buffer_span": round(window.span_seconds, 2) if window else 0.0,
            "buffer_ready": window.is_full if window else False,
            "has_pose": pose is not None,
            "crew_name": lead.name if lead else None,
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

            # `poses` is one entry per crew member visible this frame.
            height, width = frame.shape[:2]
            normalised, raw_by_index = [], {}
            for pose_landmarks in poses:
                pose = normalise(pose_landmarks, width, height)
                if pose is None:
                    continue
                raw_by_index[id(pose)] = pose_landmarks
                normalised.append(pose)

            matched = self.crew.update(normalised)
            self.crew_count = len(matched)
            self.pose_detected = bool(matched)
            self.landmarks = raw_by_index.get(id(normalised[0])) if normalised else None
            self.normalised = normalised[0] if normalised else None

            for track, pose in matched:
                self.logger.observe(track.name, track.label, track.confidence)

            # Close the log segment of anyone whose track has been retired.
            active = {t.name for t in self.crew.tracks}
            for name in self._known_crew - active:
                self.logger.retire(name)
            self._known_crew = active

            for track, pose in matched:
                pose_landmarks = raw_by_index.get(id(pose))
                if pose_landmarks is None:
                    continue
                colour = config.ACTIVITY_COLORS.get(track.label, config.COLOR_ACCENT)
                frame = draw_skeleton(frame, pose_landmarks, colour)
                draw_crew_tag(frame, pose_landmarks, track.name, track.label, colour)

            draw_crew_summary(frame, self.crew.tracks)

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
