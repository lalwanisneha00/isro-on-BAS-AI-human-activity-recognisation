"""Threaded webcam capture that never raises to the caller.

The grab loop runs on its own thread and always keeps the newest frame in
memory, so the web layer reads current video instead of a stale driver buffer.
If the camera is missing, busy, or unplugged mid-run, `read()` returns a
generated "no signal" frame and the loop keeps retrying in the background.
"""

import threading
import time

import cv2
import numpy as np

from . import config


def _no_signal_frame(message: str) -> np.ndarray:
    """Build a dark placeholder frame used whenever real video is unavailable."""
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (23, 14, 11)  # BGR of the near-black navy background

    # Faint grid so the panel still reads as an active instrument, not a dead box.
    for x in range(0, config.FRAME_WIDTH, 48):
        cv2.line(frame, (x, 0), (x, config.FRAME_HEIGHT), (40, 28, 22), 1)
    for y in range(0, config.FRAME_HEIGHT, 48):
        cv2.line(frame, (0, y), (config.FRAME_WIDTH, y), (40, 28, 22), 1)

    cv2.putText(frame, "NO CAMERA SIGNAL", (40, config.FRAME_HEIGHT // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 217, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (40, config.FRAME_HEIGHT // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 150, 120), 1, cv2.LINE_AA)
    return frame


class CameraSource:
    """Background webcam reader with automatic reconnect."""

    def __init__(self, index: int = config.CAMERA_INDEX):
        self.index = index
        self._capture = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.status = "starting"          # starting | live | reconnecting
        self.detail = "Initialising capture device"
        self.fps = 0.0

    # ------------------------------------------------------------ lifecycle --
    def start(self) -> "CameraSource":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release()

    # --------------------------------------------------------------- reading --
    def read(self) -> np.ndarray:
        """Return the newest frame, or a placeholder if the camera is down."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
        return _no_signal_frame(self.detail)

    # -------------------------------------------------------------- internals --
    def _open(self) -> bool:
        self._release()
        # CAP_DSHOW is the fast, reliable Windows backend; fall back to default.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            try:
                capture = cv2.VideoCapture(self.index, backend)
            except Exception:
                continue
            if capture is not None and capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._capture = capture
                return True
            if capture is not None:
                capture.release()
        return False

    def _release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def _loop(self) -> None:
        frame_interval = 1.0 / config.TARGET_FPS
        last_tick = time.time()
        smoothed_fps = 0.0
        frames_seen = 0
        healthy_since = time.time()

        while self._running:
            if self._capture is None:
                if self._open():
                    self.status = "live"
                    self.detail = f"Camera {self.index} online"
                    smoothed_fps = 0.0
                    frames_seen = 0
                    healthy_since = time.time()
                else:
                    self.status = "reconnecting"
                    self.detail = f"Camera {self.index} not available - retrying"
                    with self._lock:
                        self._frame = None
                    time.sleep(1.5)
                    continue

            try:
                ok, frame = self._capture.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None:
                self.status = "reconnecting"
                self.detail = "Lost video stream - reconnecting"
                self._release()
                with self._lock:
                    self._frame = None
                time.sleep(0.8)
                continue

            frame = cv2.flip(frame, 1)  # mirror: moving right on screen matches you
            if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            with self._lock:
                self._frame = frame
            self.status = "live"

            now = time.time()
            delta = now - last_tick
            last_tick = now
            if delta > 0:
                smoothed_fps = (0.9 * smoothed_fps) + (0.1 * (1.0 / delta))
                self.fps = round(smoothed_fps, 1)

            # Stall watchdog: a device that opens but trickles frames is worse
            # than one that fails outright, because nothing else notices.
            frames_seen += 1
            if smoothed_fps >= config.CAMERA_MIN_HEALTHY_FPS:
                healthy_since = now
            elif (frames_seen > 30
                  and (now - healthy_since) > config.CAMERA_STALL_SECONDS):
                self.status = "reconnecting"
                self.detail = "Camera stalled - reacquiring device"
                self._release()
                with self._lock:
                    self._frame = None
                smoothed_fps, frames_seen = 0.0, 0
                healthy_since = time.time()
                self.fps = 0.0
                time.sleep(0.6)
                continue

            sleep_for = frame_interval - (time.time() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

        self._release()
