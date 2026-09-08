"""Generate the Demo Mode clip and its matching pose track.

Demo Mode has to work with no webcam and no lighting luck, so the clip ships
with the exact 33 landmarks for every frame written alongside it. Playback
feeds those landmarks straight into the normal pipeline, which means
normalisation, the feature window, the classifier and the logger all run for
real during a demo - the activity labels are genuinely recognised, not faked.

Run once:  python tools/make_demo_clip.py
Writes:    demo/demo_feed.mp4  and  demo/demo_poses.npz
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

WIDTH, HEIGHT = config.FRAME_WIDTH, config.FRAME_HEIGHT
FPS = config.TARGET_FPS
OUT_DIR = Path(__file__).resolve().parent.parent / "demo"

# The activity rota. Each crew member runs the same rota started at a
# different point, so at any moment the module shows several different
# activities at once - which is the whole point of tracking a crew.
SEQUENCE = [
    ("Idle", 11.0),
    ("Experiment Operation", 13.0),
    ("Exercise", 12.0),
    ("In-Transit/Movement", 9.0),
    ("Eating/Rest", 12.0),
    ("Maintenance", 11.0),
    ("Anomaly (No Motion)", 20.0),
]

# Where each crew member works, as fractions of the frame, plus how large they
# appear. Two rows, so six people fit at a size the pose model can still read.
# One station per crew member the clip should contain. v1 demos a single crew
# member, matching MAX_CREW; the extra stations are kept so a multi-crew clip
# is one edit away when multi-crew is switched back on.
ALL_STATIONS = [
    (0.50, 0.78, 150.0),
    (0.18, 0.42, 62.0),
    (0.84, 0.42, 62.0),
    (0.20, 0.92, 78.0),
    (0.53, 0.95, 82.0),
    (0.86, 0.92, 78.0),
]
CREW_STATIONS = ALL_STATIONS[:config.MAX_CREW]

# ---------------------------------------------------------------- body model --
# Everything is authored in torso units: hips at the origin, one torso up to the
# shoulders, screen-up is negative Y. This is the same frame the normaliser
# produces, so the generated poses are consistent with live capture by design.

BASE_HIP_L = (-0.15, 0.0)
BASE_HIP_R = (0.15, 0.0)
BASE_SHOULDER_L = (-0.21, -1.0)
BASE_SHOULDER_R = (0.21, -1.0)
BASE_KNEE_L = (-0.17, 0.62)
BASE_KNEE_R = (0.17, 0.62)
BASE_ANKLE_L = (-0.19, 1.18)
BASE_ANKLE_R = (0.19, 1.18)


def _elbow(shoulder, wrist, side):
    """Place the elbow so the arm bends outward instead of folding flat."""
    sx, sy = shoulder
    wx, wy = wrist
    mx, my = (sx + wx) / 2.0, (sy + wy) / 2.0
    dx, dy = wx - sx, wy - sy
    length = math.hypot(dx, dy) or 1e-3
    # Perpendicular offset, scaled so a straight arm bends less than a bent one.
    bend = max(0.0, 0.62 - length * 0.34) * side
    return (mx - dy / length * bend, my + dx / length * bend)


def build_landmarks(wrist_l, wrist_r, lean=0.0):
    """Return all 33 landmarks in torso units for one frame."""
    p = np.zeros((33, 2), dtype=np.float32)

    p[23], p[24] = BASE_HIP_L, BASE_HIP_R
    p[11], p[12] = BASE_SHOULDER_L, BASE_SHOULDER_R
    p[15], p[16] = wrist_l, wrist_r
    p[13] = _elbow(BASE_SHOULDER_L, wrist_l, -1.0)
    p[14] = _elbow(BASE_SHOULDER_R, wrist_r, 1.0)
    p[25], p[26] = BASE_KNEE_L, BASE_KNEE_R
    p[27], p[28] = BASE_ANKLE_L, BASE_ANKLE_R

    # Head cluster, riding just above the shoulder line.
    neck_y = -1.06
    p[0] = (0.0, neck_y - 0.40)                       # nose
    p[1] = (-0.05, neck_y - 0.47); p[2] = (-0.07, neck_y - 0.47)
    p[3] = (-0.09, neck_y - 0.47); p[4] = (0.05, neck_y - 0.47)
    p[5] = (0.07, neck_y - 0.47);  p[6] = (0.09, neck_y - 0.47)
    p[7] = (-0.13, neck_y - 0.43); p[8] = (0.13, neck_y - 0.43)
    p[9] = (-0.05, neck_y - 0.33); p[10] = (0.05, neck_y - 0.33)

    # Hands: knuckles, fingertips and thumbs just beyond each wrist.
    for wrist, (pinky, index, thumb), side in (
            (wrist_l, (17, 19, 21), -1.0), (wrist_r, (18, 20, 22), 1.0)):
        wx, wy = wrist
        p[pinky] = (wx + 0.07 * side, wy + 0.07)
        p[index] = (wx + 0.10 * side, wy + 0.03)
        p[thumb] = (wx + 0.04 * side, wy + 0.02)

    # Feet.
    p[29], p[31] = (-0.19, 1.26), (-0.27, 1.26)
    p[30], p[32] = (0.19, 1.26), (0.27, 1.26)

    if lean:
        a = math.radians(lean)
        rot = np.array([[math.cos(a), -math.sin(a)],
                        [math.sin(a), math.cos(a)]], dtype=np.float32)
        p = p @ rot.T
    return p


# ------------------------------------------------------------ arm choreography --

def arms_for(activity, t, rng):
    """Wrist positions in torso units for this activity at time t."""
    jitter = lambda a: (rng.normal(0, a), rng.normal(0, a))

    if activity == "Idle":
        # Someone at rest is not a statue: they breathe, shift their weight,
        # and let their hands drift. Without that the clip was stiller than a
        # genuinely motionless crew member, so the anomaly watch fired on it -
        # the clip was wrong about idleness, not the classifier.
        breathe = 0.022 * math.sin(2 * math.pi * 0.24 * t)
        drift_l = 0.055 * math.sin(2 * math.pi * 0.13 * t + 0.4)
        drift_r = 0.050 * math.sin(2 * math.pi * 0.11 * t + 2.1)
        jx, jy = jitter(0.005)
        return ((-0.44 + drift_l + jx, -0.10 + breathe + jy),
                (0.44 + drift_r + jx, -0.10 + breathe + jy))

    if activity == "Experiment Operation":
        dx = 0.055 * math.sin(2 * math.pi * 0.85 * t)
        dy = 0.045 * math.sin(2 * math.pi * 1.25 * t + 0.7)
        return (-0.29 + dx, -0.56 + dy), (0.29 + dx, -0.56 - dy)

    if activity == "Exercise":
        y = -0.52 + 0.58 * math.sin(2 * math.pi * 1.45 * t)
        x = 0.56 + 0.16 * math.sin(2 * math.pi * 1.45 * t)
        return (-x, y), (x, y)

    if activity == "In-Transit/Movement":
        swing = 0.16 * math.sin(2 * math.pi * 0.9 * t)
        return (-0.42, -0.12 + swing), (0.42, -0.12 - swing)

    if activity == "Eating/Rest":
        phase = 0.5 + 0.5 * math.sin(2 * math.pi * 0.32 * t)
        return (-0.42 + 0.36 * phase, -0.14 - 1.24 * phase), (0.45, -0.12)

    if activity == "Maintenance":
        burst = 0.09 * math.sin(2 * math.pi * 0.75 * t) if int(t) % 2 == 0 else 0.0
        return (-0.52, -1.52 + burst), (0.52, -1.47 + burst)

    # Anomaly: motionless.
    return (-0.44, -0.10), (0.44, -0.10)


# ------------------------------------------------------------------ rendering --

SUIT = (188, 176, 158)
SUIT_DARK = (150, 138, 120)
TRIM = (229, 135, 57)
SKIN = (168, 186, 214)


def draw_background(canvas):
    """A plain station-module wall: panel seams, a rail, gentle vignette."""
    canvas[:] = (46, 40, 34)
    cv2.rectangle(canvas, (0, 0), (WIDTH, 150), (58, 51, 43), -1)
    cv2.rectangle(canvas, (0, HEIGHT - 120), (WIDTH, HEIGHT), (36, 31, 26), -1)

    for x in range(0, WIDTH, 120):
        cv2.line(canvas, (x, 0), (x, HEIGHT), (58, 51, 44), 1, cv2.LINE_AA)
    for y in range(150, HEIGHT - 120, 96):
        cv2.line(canvas, (0, y), (WIDTH, y), (58, 51, 44), 1, cv2.LINE_AA)

    # Equipment rail and a couple of stowage panels for depth.
    cv2.rectangle(canvas, (0, 300), (WIDTH, 312), (72, 64, 54), -1)
    for x0 in (70, 300, 640, 830):
        cv2.rectangle(canvas, (x0, 176), (x0 + 128, 286), (54, 47, 40), -1)
        cv2.rectangle(canvas, (x0, 176), (x0 + 128, 286), (76, 68, 58), 1, cv2.LINE_AA)

    vignette = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    cv2.circle(vignette, (WIDTH // 2, HEIGHT // 2), int(WIDTH * 0.62), 1.0, -1)
    vignette = cv2.GaussianBlur(vignette, (0, 0), 140)
    vignette = 0.45 + 0.55 * (vignette / (vignette.max() or 1.0))
    return (canvas * vignette[:, :, None]).astype(np.uint8)


def draw_figure(canvas, pts_px):
    """Draw the crew member as a simple solid figure from the landmark set."""
    ipt = lambda i: (int(round(pts_px[i][0])), int(round(pts_px[i][1])))

    # Legs and arms as thick rounded strokes.
    for a, b, w in ((23, 25, 26), (25, 27, 22), (24, 26, 26), (26, 28, 22)):
        cv2.line(canvas, ipt(a), ipt(b), SUIT_DARK, w, cv2.LINE_AA)
    for a, b, w in ((11, 13, 20), (13, 15, 17), (12, 14, 20), (14, 16, 17)):
        cv2.line(canvas, ipt(a), ipt(b), SUIT, w, cv2.LINE_AA)

    # Torso.
    torso = np.array([ipt(11), ipt(12), ipt(24), ipt(23)], dtype=np.int32)
    cv2.fillPoly(canvas, [torso], SUIT, cv2.LINE_AA)
    cv2.polylines(canvas, [torso], True, SUIT_DARK, 2, cv2.LINE_AA)

    # Mission trim across the chest, so the figure reads as a crew member.
    chest_l = ((pts_px[11] + pts_px[23] * 0.28) / 1.28).astype(int)
    chest_r = ((pts_px[12] + pts_px[24] * 0.28) / 1.28).astype(int)
    cv2.line(canvas, tuple(chest_l), tuple(chest_r), TRIM, 5, cv2.LINE_AA)

    # Hands and head.
    for i in (15, 16):
        cv2.circle(canvas, ipt(i), 11, SKIN, -1, cv2.LINE_AA)
    neck = ((pts_px[11] + pts_px[12]) / 2).astype(int)
    head = pts_px[0]
    radius = int(max(18, np.linalg.norm(pts_px[7] - pts_px[8]) * 1.15))
    cv2.line(canvas, tuple(neck), (int(head[0]), int(head[1])), SUIT, 16, cv2.LINE_AA)
    cv2.circle(canvas, (int(head[0]), int(head[1])), radius, SKIN, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(head[0]), int(head[1])), radius, SUIT_DARK, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------- main --

def schedule_for(crew_index):
    """The rota for one crew member, rotated so everyone is out of step."""
    rota = SEQUENCE[crew_index % len(SEQUENCE):] + SEQUENCE[:crew_index % len(SEQUENCE)]
    timeline, clock = [], 0.0
    for activity, seconds in rota:
        timeline.append((clock, clock + seconds, activity))
        clock += seconds
    return timeline, clock


def activity_at(timeline, total, t):
    """Which activity this crew member is doing at time t, and how far in."""
    t = t % total
    for start, end, activity in timeline:
        if start <= t < end:
            return activity, t - start
    return timeline[-1][2], 0.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUT_DIR / "demo_feed.mp4"
    pose_path = OUT_DIR / "demo_poses.npz"

    rng = np.random.default_rng(20260907)
    # The module wall never changes, so it is rendered once and copied.
    backdrop = draw_background(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit(f"Could not open a video writer for {video_path}")

    crew = len(CREW_STATIONS)
    schedules = [schedule_for(i) for i in range(crew)]
    loop_seconds = max(total for _, total in schedules)
    frames = int(loop_seconds * FPS)

    all_points = np.zeros((frames, crew, 33, 3), dtype=np.float32)
    all_labels = np.empty((frames, crew), dtype=object)

    for i in range(frames):
        t = i / FPS
        canvas = backdrop.copy()
        placed = []

        for c, (hip_x, hip_y, torso_px) in enumerate(CREW_STATIONS):
            timeline, total = schedules[c]
            activity, local_t = activity_at(timeline, total, t)
            wrist_l, wrist_r = arms_for(activity, local_t, rng)

            # In-Transit means actually crossing the module, not swaying on
            # the spot: the crew member traverses most of the frame, which is
            # what the classifier measures as centre-of-mass movement.
            if activity == "In-Transit/Movement":
                span = next(e - s for s, e, a in timeline if a == activity)
                # Traverse continuously, there and back, so the crew member is
                # actually in transit for the whole phase instead of arriving
                # early and then standing still.
                phase = (local_t / span) * 2.0
                travel = phase if phase <= 1.0 else (2.0 - phase)
                cx = 0.16 + 0.68 * travel
                lean = 5.0 * math.sin(2 * math.pi * 0.9 * local_t)
            else:
                cx, lean = hip_x, 0.0

            body = build_landmarks(wrist_l, wrist_r, lean)
            px = np.empty_like(body)
            px[:, 0] = body[:, 0] * torso_px + cx * WIDTH
            px[:, 1] = body[:, 1] * torso_px + hip_y * HEIGHT

            placed.append((torso_px, px))
            all_points[i, c, :, 0] = px[:, 0] / WIDTH
            all_points[i, c, :, 1] = px[:, 1] / HEIGHT
            all_points[i, c, :, 2] = 1.0          # visibility
            all_labels[i, c] = activity

        # Farthest crew first, so nearer people occlude them correctly.
        for _, px in sorted(placed, key=lambda item: item[0]):
            draw_figure(canvas, px)
        writer.write(canvas)

    writer.release()
    np.savez_compressed(pose_path,
                        points=all_points,
                        labels=all_labels.astype("U24"),
                        fps=np.array([FPS]))

    print(f"wrote {video_path.name}: {frames} frames, {frames / FPS:.1f}s, {crew} crew")
    print(f"wrote {pose_path.name}: {all_points.shape}")
    for c in range(crew):
        rota = " -> ".join(a for _, _, a in schedules[c][0][:3])
        print(f"   CM-{c + 1} starts: {rota}")


if __name__ == "__main__":
    main()
