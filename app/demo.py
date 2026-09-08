"""Demo Mode: replay a recorded crew feed instead of the live webcam.

A live demo can be ruined by a busy camera, bad lighting, or a laptop that
refuses to hand over the device. Demo Mode removes every one of those risks by
replaying a clip that ships with the project, looping for as long as needed.

The clip carries its own pose track, so playback does not depend on MediaPipe
finding a person in the footage. Those landmarks flow into the ordinary
pipeline, which means normalisation, the rolling window, the classifier and the
logger are all doing genuine work during a demo - the labels on screen are
recognised from the motion, not read from a script.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config

DEMO_ROOT = Path(__file__).resolve().parent.parent / config.DEMO_DIR


class _Landmark:
    """Mirrors the attributes the rest of the pipeline reads off MediaPipe."""

    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, visibility):
        self.x = float(x)
        self.y = float(y)
        self.visibility = float(visibility)


class DemoSource:
    """Replays the demo clip and its pose track, looping forever."""

    def __init__(self, directory: Path = DEMO_ROOT):
        self.directory = Path(directory)
        self.video_path = self.directory / config.DEMO_VIDEO_NAME
        self.pose_path = self.directory / config.DEMO_POSE_NAME

        self.available = False
        self.error = ""
        self.status = "idle"
        self.detail = "Demo Mode standby"
        self.crew_count = 0

        # Frames are streamed from the file rather than decoded up front:
        # holding this clip in memory as raw frames would cost several GB.
        self._capture = None
        self._frame_count = 0
        self._poses = None
        self._fps = float(config.TARGET_FPS)
        self._index = 0
        self._lock = threading.Lock()

        self._load()

    # ------------------------------------------------------------- loading --
    def _load(self) -> None:
        if not self.video_path.exists() or not self.pose_path.exists():
            self.error = ("Demo clip missing - run: python tools/make_demo_clip.py")
            self.detail = self.error
            return
        try:
            data = np.load(self.pose_path, allow_pickle=False)
            poses = data["points"]
            # (frames, 33, 3) is a single-crew clip; (frames, crew, 33, 3) has
            # a whole crew. Normalise both to the multi-crew shape.
            if poses.ndim == 3:
                poses = poses[:, None, :, :]
            self._poses = poses
            self.crew_count = int(poses.shape[1])
            self._fps = float(data["fps"][0]) or float(config.TARGET_FPS)

            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise RuntimeError("video file could not be opened")

            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_frames <= 0:
                raise RuntimeError("clip reported no frames")

            # Trust whichever track is shorter, so an index can never overrun.
            self._frame_count = min(video_frames, len(self._poses))
            self._poses = self._poses[:self._frame_count]
            self._capture = capture
            self.available = True
            self.detail = (f"Demo clip ready - {self._frame_count / self._fps:.0f}s loop, "
                           f"{self.crew_count} crew")
        except Exception as exc:
            self.available = False
            self.error = f"Demo clip could not be loaded: {exc}"
            self.detail = self.error
            self._capture, self._poses = None, None

    # ------------------------------------------------------------ playback --
    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def position(self) -> float:
        """How far through the loop we are, 0.0 to 1.0."""
        if not self.frame_count:
            return 0.0
        return self._index / self.frame_count

    @property
    def elapsed(self) -> float:
        return self._index / self._fps if self._fps else 0.0

    @property
    def duration(self) -> float:
        return self.frame_count / self._fps if self._fps else 0.0

    def rewind(self) -> None:
        with self._lock:
            self._index = 0
            if self._capture is not None:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def next(self):
        """Return (frame, poses) for the next step of the loop.

        `poses` is a list with one entry per crew member in the clip, matching
        what the live pose tracker returns for a frame with several people.
        """
        if not self.available:
            return None, []

        with self._lock:
            capture = self._capture
            if capture is None:
                return None, []

            ok, frame = capture.read()
            if not ok or frame is None or self._index >= self._frame_count:
                # End of clip: rewind and keep playing, so a demo can run for
                # as long as the judges want to watch it.
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._index = 0
                ok, frame = capture.read()
                if not ok or frame is None:
                    self.available = False
                    self.detail = "Demo clip stopped delivering frames"
                    return None, []

            index = self._index
            self._index += 1

        if (frame.shape[1] != config.FRAME_WIDTH
                or frame.shape[0] != config.FRAME_HEIGHT):
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        poses = [[_Landmark(x, y, v) for x, y, v in member]
                 for member in self._poses[index]]
        self.status = "playing"
        return frame, poses

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None

    def placeholder(self) -> np.ndarray:
        """Shown when Demo Mode is selected but the clip is missing."""
        frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = (23, 14, 11)
        cv2.putText(frame, "DEMO CLIP NOT FOUND",
                    (40, config.FRAME_HEIGHT // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (25, 178, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, "Run: python tools/make_demo_clip.py",
                    (40, config.FRAME_HEIGHT // 2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 150, 120), 1, cv2.LINE_AA)
        return frame
