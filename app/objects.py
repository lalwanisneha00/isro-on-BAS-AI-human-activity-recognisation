"""Object detection, restricted to the objects our task list actually needs.

Why objects at all
------------------
Pose alone cannot separate some activities that matter. A hand raised to the
mouth is the same skeleton whether the crew member is drinking, eating, or
speaking into a headset. What settles it is what they are holding.

The honest limitation, stated plainly
-------------------------------------
This runs a COCO-pretrained YOLO model. COCO knows 80 everyday object classes
and none of the things actually aboard a station - no sample vials, no torque
tools, no cargo transfer bags. Training a detector that does know them needs a
labelled dataset of spaceflight hardware, which v1 does not have.

So v1 does two things, both deliberately:

* Only the classes on the whitelist below are ever used. Everything else the
  model reports - chairs, plants, people, whatever is behind the crew member -
  is dropped before it reaches the fusion logic, the console, or the log.
* Whitelisted classes are displayed under the name of the spaceflight item
  they stand in for. A COCO "bottle" is shown as a Water Pouch, because in
  this system that is the role it plays.

This is a stand-in, and the README says so. Custom training on real flight
hardware is the obvious next version.

Behaviour
---------
Detection runs on a background thread every few frames, and the pose pipeline
reads whatever the most recent result was. Object detection is slower than
pose tracking, and the video must never wait for it: if the detector is busy,
the last known objects are used and the frame rate is unaffected. If the model
is missing or fails to load, everything downstream carries on from pose alone.
"""

import threading
import time
from pathlib import Path

import numpy as np

from . import config

MODEL_ROOT = Path(__file__).resolve().parent.parent

# COCO class -> what it stands in for, and which tasks it supports. Anything
# not in this table is discarded, whatever the model thinks it saw.
WHITELIST = {
    "bottle":      ("Water Pouch", ("drinking", "holding")),
    "cup":         ("Drink Container", ("drinking",)),
    "bowl":        ("Food Container", ("eating",)),
    "spoon":       ("Utensil", ("eating",)),
    "fork":        ("Utensil", ("eating",)),
    "knife":       ("Utensil / Tool", ("eating", "maintenance")),
    "laptop":      ("Crew Laptop", ("laptop",)),
    "keyboard":    ("Workstation Input", ("laptop",)),
    "mouse":       ("Workstation Input", ("laptop",)),
    "cell phone":  ("Handheld Device", ("tablet", "photography")),
    "book":        ("Procedure Manual", ("reading",)),
    "scissors":    ("Hand Tool", ("maintenance",)),
    "toothbrush":  ("Hygiene Item", ("hygiene", "sample")),
    "remote":      ("Handheld Controller", ("experiment",)),
    "backpack":    ("Stowage Bag", ("stowage",)),
    "clock":       ("Timer", ("experiment",)),
}


class Detection:
    """One whitelisted object seen in the frame."""

    __slots__ = ("coco_class", "name", "roles", "confidence", "box",
                 "centre", "in_hand", "hand_distance")

    def __init__(self, coco_class, name, roles, confidence, box):
        self.coco_class = coco_class
        self.name = name
        self.roles = roles
        self.confidence = confidence
        self.box = box                      # (x1, y1, x2, y2) in pixels
        self.centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.in_hand = False
        self.hand_distance = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "coco_class": self.coco_class,
            "roles": list(self.roles),
            "confidence": round(float(self.confidence), 3),
            "box": [int(v) for v in self.box],
            "in_hand": self.in_hand,
            "hand_distance": (round(float(self.hand_distance), 2)
                              if self.hand_distance is not None else None),
        }


class ObjectDetector:
    """Runs YOLO off the critical path and keeps the latest whitelisted result."""

    def __init__(self):
        self.available = False
        self.error = ""
        self.detail = "Object detection off"
        self.model_name = config.YOLO_MODEL_NAME
        self.inference_ms = 0.0
        self.frames_between = config.YOLO_EVERY_N_FRAMES

        self._model = None
        self._names = {}
        self._latest = []
        self._lock = threading.Lock()
        self._busy = False
        self._counter = 0
        self._enabled = config.ENABLE_OBJECT_DETECTION

        if self._enabled:
            self._load()

    # -------------------------------------------------------------- loading --
    def _load(self) -> None:
        weights = MODEL_ROOT / config.YOLO_WEIGHTS_PATH
        if not weights.exists():
            self.error = (f"YOLO weights not found at {config.YOLO_WEIGHTS_PATH}. "
                          "Run: python tools/fetch_yolo.py")
            self.detail = self.error
            return
        try:
            import numpy as _np
            import torch
            from ultralytics import YOLO

            # The video pipeline already runs several threads. Letting torch
            # take half the cores as well starves them, and object detection
            # that measures 58ms on its own stretched past a second under
            # that contention. A small, fixed budget keeps both healthy.
            torch.set_num_threads(config.YOLO_THREADS)

            self._model = YOLO(str(weights))
            self._names = self._model.names

            # The first inference of any torch model is far slower than the
            # rest. Doing it here means the demo never pays for it.
            self._model.predict(_np.zeros((64, 64, 3), dtype=_np.uint8),
                                imgsz=config.YOLO_IMAGE_SIZE, verbose=False,
                                device="cpu")
            self.available = True
            self.detail = (f"{self.model_name} ready - "
                           f"{len(WHITELIST)} object types in scope")
        except Exception as exc:                       # pragma: no cover
            self.available = False
            self.error = f"Object detection unavailable: {exc}"
            self.detail = self.error

    # ------------------------------------------------------------ detection --
    @property
    def enabled(self) -> bool:
        return self._enabled and self.available

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if self._enabled and self._model is None:
            self._load()
        if not self._enabled:
            with self._lock:
                self._latest = []

    def latest(self) -> list:
        with self._lock:
            return list(self._latest)

    def submit(self, frame) -> None:
        """Offer a frame for detection. Returns immediately, always.

        The video loop calls this every frame; only every Nth actually starts
        a detection, and only if the previous one has finished. Pose tracking
        must never wait on the slower object model.
        """
        if not self.enabled or frame is None:
            return

        self._counter += 1
        if self._counter % self.frames_between:
            return
        if self._busy:
            return

        self._busy = True
        threading.Thread(target=self._detect, args=(frame.copy(),),
                         name="objects", daemon=True).start()

    def _detect(self, frame) -> None:
        started = time.time()
        found = []
        try:
            results = self._model.predict(
                frame, imgsz=config.YOLO_IMAGE_SIZE,
                conf=config.YOLO_CONFIDENCE, verbose=False, device="cpu")

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    coco = self._names.get(int(box.cls[0]), "")
                    entry = WHITELIST.get(coco)
                    if entry is None:
                        # Not on the task list. Dropped here, so it can never
                        # reach the console or the mission log.
                        continue
                    name, roles = entry
                    found.append(Detection(coco, name, roles,
                                           float(box.conf[0]),
                                           tuple(float(v) for v in box.xyxy[0])))
        except Exception as exc:                       # pragma: no cover
            self.detail = f"Object detection failed: {exc}"
            found = []
        finally:
            self.inference_ms = round((time.time() - started) * 1000, 1)
            with self._lock:
                self._latest = found
            self._busy = False

    # --------------------------------------------------------------- fusion --
    def attach_to_hands(self, detections: list, landmarks, width: int,
                        height: int, torso_px: float) -> list:
        """Mark which objects are being held, and how far from a hand.

        Distance is measured in torso lengths so "in hand" means the same
        thing whether the crew member is next to the lens or across the module.
        """
        if not detections or not landmarks:
            return detections

        scale = max(torso_px, 1e-3)
        wrists = []
        for index in (15, 16):
            if index < len(landmarks):
                wrists.append((landmarks[index].x * width,
                               landmarks[index].y * height))
        if not wrists:
            return detections

        for detection in detections:
            cx, cy = detection.centre
            nearest = min(((cx - wx) ** 2 + (cy - wy) ** 2) ** 0.5
                          for wx, wy in wrists)
            detection.hand_distance = nearest / scale
            detection.in_hand = detection.hand_distance <= config.OBJECT_HAND_RADIUS
        return detections

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "available": self.available,
            "detail": self.detail,
            "model": self.model_name,
            "inference_ms": self.inference_ms,
            "every_n_frames": self.frames_between,
            "in_scope": len(WHITELIST),
        }


def roles_present(detections: list, held_only: bool = False) -> set:
    """Which task roles the visible objects support."""
    roles = set()
    for detection in detections:
        if held_only and not detection.in_hand:
            continue
        roles.update(detection.roles)
    return roles
