"""Shared helpers for the test suite.

The classifier and logger both read the wall clock, so tests drive them through
a frozen clock: simulated activity that "lasts" twenty seconds must not take
twenty seconds to test.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.normalize import NormalisedPose  # noqa: E402

FPS = config.TARGET_FPS
DT = 1.0 / FPS
ASPECT = config.FRAME_WIDTH / config.FRAME_HEIGHT


class Landmark:
    """Stands in for a MediaPipe landmark."""

    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = float(x), float(y), float(visibility)


def body_points(wrist_l, wrist_r):
    """A plausible 33-point body in torso units, with the given wrists."""
    p = np.zeros((33, 2), dtype=np.float32)
    p[23] = (-0.15, 0.0); p[24] = (0.15, 0.0)          # hips
    p[11] = (-0.20, -1.0); p[12] = (0.20, -1.0)        # shoulders
    p[0] = (0.0, -1.45)                                 # nose
    p[7] = (-0.13, -1.38); p[8] = (0.13, -1.38)        # ears
    p[9] = (-0.05, -1.33); p[10] = (0.05, -1.33)       # mouth corners
    p[13] = (-0.45, -0.55); p[14] = (0.45, -0.55)      # elbows
    p[15] = wrist_l; p[16] = wrist_r                    # wrists
    p[25] = (-0.17, 0.60); p[26] = (0.17, 0.60)        # knees
    p[27] = (-0.18, 1.05); p[28] = (0.18, 1.05)        # ankles
    return p


def landmarks_at(cx, cy, scale, wrist_l, wrist_r, tilt_deg=0.0):
    """Place a body in the frame and return it as MediaPipe-style landmarks."""
    p = body_points(wrist_l, wrist_r).astype(np.float64)
    if tilt_deg:
        a = math.radians(tilt_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = p @ rot.T
    p = p * scale + np.array([cx * ASPECT, cy])
    return [Landmark(v[0] / ASPECT, v[1]) for v in p]


def pose_at(timestamp, wrist_l, wrist_r, hip=(0.5, 0.5), torso=0.25):
    """Build a NormalisedPose directly, bypassing the normaliser."""
    return NormalisedPose(timestamp, body_points(wrist_l, wrist_r),
                          hip, torso, 0.0, 1.0)


def frozen(now):
    """Context manager pinning time.time() to `now`."""

    class _Frozen:
        def __enter__(self):
            self.real = time.time
            time.time = lambda: now

        def __exit__(self, *exc):
            time.time = self.real

    return _Frozen()


def drive(window, classifier, motion, seconds, base=None, hip_fn=None):
    """Feed simulated motion through a window and classifier on a frozen clock.

    `motion(t)` returns (wrist_l, wrist_r) in torso units.
    """
    base = time.time() if base is None else base
    for i in range(int(seconds * FPS)):
        t = i * DT
        wl, wr = motion(t)
        hip = hip_fn(t) if hip_fn else (0.5, 0.5)
        pose = pose_at(base + t, wl, wr, hip=hip)
        window.push(pose)
        with frozen(base + t):
            classifier.update(window)
    return classifier


class Results:
    """Collects pass/fail lines for one test module."""

    def __init__(self, title):
        self.title = title
        self.passed = 0
        self.failed = 0
        self.lines = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            self.lines.append(f"    PASS  {label}")
        else:
            self.failed += 1
            self.lines.append(f"    FAIL  {label}"
                              + (f"  ({detail})" if detail else ""))
        return ok

    def note(self, text: str) -> None:
        self.lines.append(f"          {text}")

    def report(self) -> None:
        print(f"  {self.title}")
        for line in self.lines:
            print(line)
