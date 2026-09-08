"""Normalisation must erase distance, position and body tilt - and nothing else."""

import numpy as np

from _harness import ASPECT, Landmark, Results, landmarks_at, pose_at
from app import config
from app.normalize import PoseWindow, normalise

W, H = config.FRAME_WIDTH, config.FRAME_HEIGHT
IDLE = ((-0.50, -0.15), (0.50, -0.15))


def run() -> Results:
    r = Results("normalisation")

    # ---- the same body, seen every way a camera could see it ---------------
    cases = [
        ("baseline",             0.50, 0.50, 0.25, 0),
        ("twice as close",       0.50, 0.50, 0.50, 0),
        ("far away",             0.50, 0.50, 0.10, 0),
        ("off in the corner",    0.20, 0.75, 0.25, 0),
        ("tilted 30 deg",        0.50, 0.50, 0.25, 30),
        ("tilted -55 deg",       0.50, 0.50, 0.25, -55),
        ("floating upside down", 0.50, 0.50, 0.25, 180),
    ]

    reference, torsos = None, []
    worst_drift = 0.0
    for name, cx, cy, scale, tilt in cases:
        pose = normalise(landmarks_at(cx, cy, scale, *IDLE, tilt_deg=tilt), W, H)
        if not r.check(pose is not None, f"{name}: pose accepted"):
            continue
        torsos.append(pose.torso_length)
        shoulder = (pose.points[11] + pose.points[12]) / 2.0
        r.check(abs(shoulder[0]) < 1e-3 and abs(shoulder[1] + 1.0) < 1e-3,
                f"{name}: shoulders pinned at (0, -1)",
                f"got ({shoulder[0]:.4f}, {shoulder[1]:.4f})")
        if reference is None:
            reference = pose.points
        else:
            worst_drift = max(worst_drift, float(np.abs(pose.points - reference).max()))

    r.check(worst_drift < 1e-3, "every view produces identical normalised points",
            f"worst drift {worst_drift:.5f}")
    r.check(max(torsos) / min(torsos) > 4.0,
            "raw torso length really did vary across those views",
            f"ratio {max(torsos) / min(torsos):.1f}x")
    r.note(f"raw torso {min(torsos):.3f}-{max(torsos):.3f}, "
           f"normalised drift {worst_drift:.6f}")

    # ---- the sanity gate rejects things that are not bodies ---------------
    rubbish = [
        ("all landmarks invisible",
         [Landmark(0.5, 0.5, 0.0) for _ in range(33)]),
        ("collapsed to a point",
         [Landmark(0.5, 0.5, 1.0) for _ in range(33)]),
        ("far outside the frame",
         landmarks_at(4.0, 4.0, 0.25, *IDLE)),
    ]
    for name, lms in rubbish:
        r.check(normalise(lms, W, H) is None, f"rejects {name}")

    r.check(normalise([], W, H) is None, "rejects an empty landmark list")
    r.check(normalise(None, W, H) is None, "rejects None")

    # ---- the window is bounded by time, not by a frame count --------------
    for fps, label in ((30.0, "30 fps"), (12.0, "12 fps")):
        window = PoseWindow()
        base = 1000.0
        for i in range(int(6 * fps)):
            window.push(pose_at(base + i / fps, *IDLE))
        span = window.span_seconds
        r.check(abs(span - config.WINDOW_SECONDS) < 0.25,
                f"window spans ~{config.WINDOW_SECONDS}s at {label}",
                f"got {span:.2f}s")

    # A gap shorter than the tolerance must not wipe the history.
    window = PoseWindow()
    base = 2000.0
    for i in range(40):
        window.push(pose_at(base + i / 24.0, *IDLE))
    before = window.count
    window.push(None)
    r.check(window.count == before, "a single dropped frame keeps the window")

    return r


if __name__ == "__main__":
    run().report()
