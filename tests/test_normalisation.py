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

    # ---- half-framed subjects, the ordinary webcam view -------------------
    # A crew member at a laptop shows head, torso and arms; the legs are out
    # of shot. MediaPipe still emits leg and hip landmarks, guessed, with
    # near-zero visibility. Requiring the hips is what made detection flicker
    # on and off, and trusting the guessed hips made the unit of scale jump
    # between frames.
    def webcam_pose(hip_y=1.21, hip_vis=0.07, shoulder_vis=0.99,
                    shoulder_y=0.60, nose_y=0.42, jitter=0.0):
        p = np.zeros((33, 2), dtype=np.float64)
        p[0] = (0.50, nose_y)
        p[7] = (0.46, nose_y + 0.02); p[8] = (0.54, nose_y + 0.02)
        p[9] = (0.48, nose_y + 0.05); p[10] = (0.52, nose_y + 0.05)
        p[11] = (0.62, shoulder_y); p[12] = (0.38, shoulder_y)
        p[13] = (0.70, shoulder_y + 0.16); p[14] = (0.30, shoulder_y + 0.16)
        p[15] = (0.68, shoulder_y + 0.30); p[16] = (0.32, shoulder_y + 0.30)
        # The guessed lower body, wherever the model decided to put it.
        for i in (23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
            p[i] = (0.50, hip_y + jitter)
        p[23] = (0.56, hip_y + jitter); p[24] = (0.44, hip_y + jitter)

        vis = np.ones(33, dtype=np.float32)
        vis[[11, 12]] = shoulder_vis
        for i in (23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
            vis[i] = hip_vis
        return [Landmark(v[0], v[1], float(vis[j])) for j, v in enumerate(p)]

    upper = normalise(webcam_pose(), W, H)
    r.check(upper is not None,
            "an upper-body-only crew member is accepted, not discarded")
    if upper is not None:
        r.check(upper.framing == "upper", "the framing is reported as upper body",
                upper.framing)
        r.check(upper.hips_estimated,
                "the hips are reconstructed rather than taken on trust")
        shoulder_span = abs(0.62 - 0.38) * ASPECT
        r.check(abs(upper.torso_length
                    - shoulder_span * config.TORSO_PER_SHOULDER) < 0.02,
                "torso length is built from shoulder width, which is visible",
                f"{upper.torso_length:.3f}")

    # The regression that matters: the model's guess for an unseen hip jumps
    # around between frames. The reconstructed hip must not, because every
    # feature downstream is measured in torso lengths.
    lengths = []
    for wobble in (-0.35, -0.1, 0.0, 0.2, 0.5, 0.9):
        pose = normalise(webcam_pose(jitter=wobble), W, H)
        if pose is not None:
            lengths.append(pose.torso_length)
    r.check(len(lengths) == 6, "every frame stays usable as the guess wanders",
            f"{len(lengths)}/6 accepted")
    if lengths:
        spread = (max(lengths) - min(lengths)) / max(1e-6, np.mean(lengths))
        r.check(spread < 0.02,
                "the unit of scale holds steady while the guessed hip wanders",
                f"spread {spread:.1%}")
        r.note(f"guessed hip moved 1.25 frame-heights; torso length varied "
               f"{spread:.2%}")

    # Barely-visible shoulders: nothing can be built, and nothing is invented.
    r.check(normalise(webcam_pose(shoulder_vis=0.2), W, H) is None,
            "with the shoulders unseen, no body frame is invented")

    # A full-body pose still uses the real hips.
    full = normalise(landmarks_at(0.5, 0.5, 0.25, *IDLE), W, H)
    r.check(full is not None and not full.hips_estimated,
            "with the hips visible they are used, not reconstructed")
    r.check(full is not None and full.framing == "full",
            "and the framing is reported as full body",
            full.framing if full else "none")

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
