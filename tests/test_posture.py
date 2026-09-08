"""Seated detection: facing away, behind a desk, near and far.

Face landmarks are never used, so every case here is also the facing-away
case. What varies is whether the legs can be seen at all.
"""

import numpy as np

from _harness import ASPECT, FPS, Landmark, Results
from app import config
from app.normalize import PoseWindow, normalise
from app.posture import FLOATING, SEATED, STANDING, read

W, H = config.FRAME_WIDTH, config.FRAME_HEIGHT


def build(seated: bool, legs_visible: bool = True):
    """A body in torso units, plus a per-landmark visibility track."""
    p = np.zeros((33, 2), dtype=np.float64)
    p[23] = (-0.15, 0.0); p[24] = (0.15, 0.0)          # hips
    p[11] = (-0.20, -1.0); p[12] = (0.20, -1.0)        # shoulders
    p[0] = (0.0, -1.45)                                 # nose
    p[7] = (-0.13, -1.38); p[8] = (0.13, -1.38)        # ears
    p[9] = (-0.05, -1.33); p[10] = (0.05, -1.33)       # mouth
    p[13] = (-0.42, -0.55); p[14] = (0.42, -0.55)      # elbows
    p[15] = (-0.30, -0.35); p[16] = (0.30, -0.35)      # wrists

    if seated == "profile":
        # Seen from the side the thigh has real length on screen, so the knee
        # angle is measurable.
        p[25] = (0.55, 0.05); p[26] = (0.58, 0.05)
        p[27] = (0.55, 0.62); p[28] = (0.58, 0.62)
    elif seated:
        # Head-on the thigh points at the camera and collapses to almost
        # nothing, so the knees sit just under the hips.
        p[25] = (-0.17, 0.05); p[26] = (0.17, 0.05)
        p[27] = (-0.17, 0.62); p[28] = (0.17, 0.62)
    else:
        p[25] = (-0.17, 0.62); p[26] = (0.17, 0.62)
        p[27] = (-0.18, 1.20); p[28] = (0.18, 1.20)

    visibility = np.ones(33, dtype=np.float32)
    if not legs_visible:
        # A desk hides the legs: the model still guesses at them, but says so.
        for i in (25, 26, 27, 28, 29, 30, 31, 32):
            visibility[i] = 0.15
    return p, visibility


def feed(points, visibility, seconds=3.0, scale=0.25, drift=0.0, tilt=0.0):
    """Run a static posture through the window, at a given apparent size."""
    window = PoseWindow()
    base = 1000.0
    for i in range(int(seconds * FPS)):
        t = i / FPS
        shifted = points.copy()
        if tilt:
            a = np.radians(tilt)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            shifted = shifted @ rot.T
        cx = 0.5 + drift * t
        placed = shifted * scale + np.array([cx * ASPECT, 0.5])
        landmarks = [Landmark(v[0] / ASPECT, v[1], float(visibility[j]))
                     for j, v in enumerate(placed)]
        pose = normalise(landmarks, W, H)
        if pose is not None:
            pose.timestamp = base + t
            window.push(pose)
    return window


def run() -> Results:
    r = Results("seated posture")

    seated_pts, seated_vis = build(seated=True)
    standing_pts, standing_vis = build(seated=False)
    hidden_pts, hidden_vis = build(seated=True, legs_visible=False)

    # ---- the basic distinction, legs in shot ------------------------------
    seated = read(feed(seated_pts, seated_vis))
    standing = read(feed(standing_pts, standing_vis))

    r.check(seated.seated and seated.posture == SEATED,
            "a seated person is recognised as seated",
            f"{seated.posture}, score basis: {seated.basis}")
    r.check(not standing.seated and standing.posture == STANDING,
            "a standing person is not called seated",
            f"{standing.posture}")
    r.check("hip height" in seated.basis,
            "head-on, the decision falls back to hip height",
            seated.basis)

    # Seen from the side the thigh is measurable, and the knee angle is used.
    profile_pts, profile_vis = build(seated="profile")
    profile = read(feed(profile_pts, profile_vis))
    r.check(profile.seated, "seated is recognised from the side too",
            f"{profile.posture}")
    r.check(profile.basis.startswith("knee angle"),
            "in profile the knee angle is trusted and used", profile.basis)
    r.note(f"profile knee {profile.knee_angle:.0f} deg, "
           f"head-on knee {seated.knee_angle:.0f} deg for the same posture")
    r.note(f"seated knee {seated.knee_angle:.0f} deg vs "
           f"standing {standing.knee_angle:.0f} deg")
    r.note(f"seated hips {seated.hip_above_ankle:.2f} above ankles vs "
           f"standing {standing.hip_above_ankle:.2f}")

    # ---- legs hidden behind a desk ----------------------------------------
    hidden = read(feed(hidden_pts, hidden_vis))
    r.check(not hidden.lower_body_visible,
            "hidden legs are recognised as hidden, not as folded")
    r.check(hidden.seated, "a seated person behind a desk is still seated",
            f"{hidden.posture}, {hidden.basis}")
    r.check(hidden.confidence < seated.confidence,
            "confidence is lower when the legs cannot be seen",
            f"{hidden.confidence} vs {seated.confidence}")

    # ---- distance from the camera -----------------------------------------
    for scale, label in ((0.42, "close to the camera"), (0.12, "far away")):
        near_far = read(feed(seated_pts, seated_vis, scale=scale))
        r.check(near_far.seated, f"seated is recognised {label}",
                f"{near_far.posture} at scale {scale}")

    # ---- body tilt ---------------------------------------------------------
    leaning = read(feed(seated_pts, seated_vis, tilt=18.0))
    r.check(leaning.seated, "seated survives leaning to one side",
            f"{leaning.posture}")

    # ---- someone drifting across the module is not seated -----------------
    drifting = read(feed(standing_pts, standing_vis, drift=0.10))
    r.check(drifting.posture in (STANDING, FLOATING),
            "a person moving across the module is not called seated",
            f"{drifting.posture}")

    # ---- the classifier reports it -----------------------------------------
    from app.classifier import ActivityClassifier
    from _harness import frozen

    window = PoseWindow()
    clf = ActivityClassifier()
    base = 2000.0
    for i in range(int(4.0 * FPS)):
        t = i / FPS
        placed = seated_pts * 0.25 + np.array([0.5 * ASPECT, 0.5])
        landmarks = [Landmark(v[0] / ASPECT, v[1], float(seated_vis[j]))
                     for j, v in enumerate(placed)]
        pose = normalise(landmarks, W, H)
        if pose is not None:
            pose.timestamp = base + t
            window.push(pose)
        with frozen(base + t):
            clf.update(window)

    r.check(clf.posture is not None and clf.posture.seated,
            "the classifier exposes the seated posture as a modifier")
    r.check(clf.label == config.ACT_SEATED,
            "a seated person doing nothing else reads as Seated at Workstation",
            f"got {clf.label}")

    # ---- posture never crashes on a thin window ----------------------------
    r.check(read(PoseWindow()).posture == "Unknown",
            "an empty window reports Unknown rather than guessing")

    return r


if __name__ == "__main__":
    run().report()
