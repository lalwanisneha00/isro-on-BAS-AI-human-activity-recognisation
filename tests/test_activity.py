"""Activity recognition: the seven classes, and the cases that used to confuse it."""

import math

import numpy as np

from _harness import DT, FPS, Results, body_points, drive, frozen
from app import config
from app.classifier import ActivityClassifier
from app.features import _periodicity
from app.normalize import PoseWindow

RNG = np.random.default_rng(7)


# --------------------------------------------------------------- motions ----
def idle(t):
    j = RNG.normal(0, 0.004, 2)
    return (-0.50 + j[0], -0.15 + j[1]), (0.50 + j[0], -0.15 + j[1])


def exercise(t):
    y = -0.55 + 0.55 * math.sin(2 * math.pi * 1.6 * t)
    x = 0.55 + 0.15 * math.sin(2 * math.pi * 1.6 * t)
    return (-x, y), (x, y)


def experiment(t):
    dx = 0.05 * math.sin(2 * math.pi * 0.9 * t)
    dy = 0.04 * math.sin(2 * math.pi * 1.3 * t)
    return (-0.30 + dx, -0.55 + dy), (0.30 + dx, -0.55 - dy)


def eating(t):
    phase = 0.5 + 0.5 * math.sin(2 * math.pi * 0.35 * t)
    return (-0.45 + 0.40 * phase, -0.20 - 1.15 * phase), (0.50, -0.15)


def maintenance(t):
    burst = 0.10 * math.sin(2 * math.pi * 0.8 * t) if int(t) % 2 == 0 else 0.0
    return (-0.55, -1.45 + burst), (0.55, -1.40 + burst)


def transit(t):
    j = RNG.normal(0, 0.02, 2)
    return (-0.50 + j[0], -0.20 + j[1]), (0.50 + j[0], -0.20 + j[1])


def motionless(t):
    return (-0.50, -0.15), (0.50, -0.15)


SCENARIOS = [
    (config.ACT_IDLE, idle, 5.0, None),
    (config.ACT_EXERCISE, exercise, 5.0, None),
    (config.ACT_EXPERIMENT, experiment, 5.0, None),
    (config.ACT_EATING, eating, 6.0, None),
    (config.ACT_MAINTENANCE, maintenance, 5.0, None),
    (config.ACT_TRANSIT, transit, 5.0, lambda t: (0.18 + 0.11 * t, 0.5)),
    (config.ACT_ANOMALY, motionless, 14.0, None),
]

# Motions that sit right on the Exercise / Experiment boundary. These are the
# cases the first classifier got wrong: it used a noisy reversal count that
# could not tell a wide repeated sweep from fast fiddling in one spot.
BORDERLINE = [
    ("typing at a console", config.ACT_EXPERIMENT,
     lambda t: ((-0.28 + 0.025 * math.sin(2 * math.pi * 3.0 * t),
                 -0.55 + 0.02 * math.sin(2 * math.pi * 3.7 * t)),
                (0.28 + 0.025 * math.sin(2 * math.pi * 3.3 * t),
                 -0.55 + 0.02 * math.sin(2 * math.pi * 2.9 * t)))),
    ("adjusting a knob quickly", config.ACT_EXPERIMENT,
     lambda t: ((-0.30 + 0.07 * math.sin(2 * math.pi * 2.4 * t), -0.52),
                (0.30 + 0.07 * math.sin(2 * math.pi * 2.4 * t), -0.52))),
    ("fast small fidget at chest", config.ACT_EXPERIMENT,
     lambda t: ((-0.26 + 0.09 * math.sin(2 * math.pi * 2.8 * t),
                 -0.60 + 0.06 * math.sin(2 * math.pi * 3.1 * t)),
                (0.26 + 0.09 * math.sin(2 * math.pi * 2.6 * t),
                 -0.60 + 0.06 * math.sin(2 * math.pi * 3.4 * t)))),
    ("handling a sample two-handed", config.ACT_EXPERIMENT,
     lambda t: ((-0.22 + 0.10 * math.sin(2 * math.pi * 1.1 * t),
                 -0.48 + 0.08 * math.cos(2 * math.pi * 1.1 * t)),
                (0.22 + 0.10 * math.sin(2 * math.pi * 1.1 * t),
                 -0.48 + 0.08 * math.cos(2 * math.pi * 1.1 * t)))),
    ("wide arm swings", config.ACT_EXERCISE,
     lambda t: ((-0.55 - 0.15 * math.sin(2 * math.pi * 1.5 * t),
                 -0.5 + 0.55 * math.sin(2 * math.pi * 1.5 * t)),
                (0.55 + 0.15 * math.sin(2 * math.pi * 1.5 * t),
                 -0.5 + 0.55 * math.sin(2 * math.pi * 1.5 * t)))),
    ("jumping-jack arms", config.ACT_EXERCISE,
     lambda t: ((-0.30 - 0.35 * abs(math.sin(2 * math.pi * 1.2 * t)),
                 -0.2 - 1.10 * abs(math.sin(2 * math.pi * 1.2 * t))),
                (0.30 + 0.35 * abs(math.sin(2 * math.pi * 1.2 * t)),
                 -0.2 - 1.10 * abs(math.sin(2 * math.pi * 1.2 * t))))),
    ("waving inside the console zone", config.ACT_EXERCISE,
     lambda t: ((-0.35, -0.55 + 0.50 * math.sin(2 * math.pi * 1.7 * t)),
                (0.35, -0.55 + 0.50 * math.sin(2 * math.pi * 1.7 * t)))),
    ("boxing punches", config.ACT_EXERCISE,
     lambda t: ((-0.20 - 0.45 * max(0, math.sin(2 * math.pi * 1.9 * t)), -0.60),
                (0.20 + 0.45 * max(0, math.sin(2 * math.pi * 1.9 * t + math.pi)),
                 -0.60))),
]


def run() -> Results:
    r = Results("activity recognition")

    # ---- the periodicity detector, in isolation ---------------------------
    t = np.linspace(0, 2, 48)
    noise = RNG.standard_normal(48)
    signals = [
        ("1.5 Hz arm swing", 0.6 * np.sin(2 * np.pi * 1.5 * t), 0.6, True),
        ("2.5 Hz fast reps", 0.5 * np.sin(2 * np.pi * 2.5 * t), 0.5, True),
        ("1.0 Hz with noise",
         0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * noise, 0.5, True),
        ("a single slow reach", 0.6 * np.linspace(0, 1, 48), 0.6, False),
        ("slow one-way drift", 0.5 * np.sin(2 * np.pi * 0.3 * t), 0.5, False),
        ("motionless jitter", 0.004 * noise, 0.01, False),
        ("a tiny fidget", 0.05 * np.sin(2 * np.pi * 1.2 * t), 0.05, False),
    ]
    for name, signal, amplitude, should_be_rhythmic in signals:
        strength, _ = _periodicity(signal, 2.0, amplitude)
        rhythmic = strength >= config.PERIODICITY_RHYTHMIC
        r.check(rhythmic == should_be_rhythmic,
                f"periodicity: {name} "
                f"{'is' if should_be_rhythmic else 'is not'} rhythmic",
                f"strength {strength:.2f}")

    # ---- each of the seven activities -------------------------------------
    for expected, motion, seconds, hip_fn in SCENARIOS:
        clf = drive(PoseWindow(), ActivityClassifier(), motion, seconds,
                    base=5000.0, hip_fn=hip_fn)
        r.check(clf.label == expected, f"recognises {expected}",
                f"got {clf.label} at {clf.confidence:.2f}")

    # ---- the borderline cases ---------------------------------------------
    for name, expected, motion in BORDERLINE:
        clf = drive(PoseWindow(), ActivityClassifier(), motion, 5.0, base=6000.0)
        r.check(clf.label == expected, f"borderline: {name} -> {expected}",
                f"got {clf.label}")

    # ---- an anomaly must not fire before its threshold ---------------------
    clf = drive(PoseWindow(), ActivityClassifier(), motionless,
                config.ANOMALY_STILL_SECONDS - 3.0, base=7000.0)
    r.check(clf.label != config.ACT_ANOMALY,
            "stillness under the threshold is Idle, not an anomaly",
            f"got {clf.label}")

    # ---- confidence stays inside its stated range -------------------------
    clf = drive(PoseWindow(), ActivityClassifier(), exercise, 5.0, base=8000.0)
    r.check(0.0 <= clf.confidence <= 1.0, "confidence stays within 0..1",
            f"got {clf.confidence}")
    r.check(all(0.0 <= v <= 1.0 for v in clf.scores.values()),
            "every activity score stays within 0..1",
            str({k: round(v, 2) for k, v in clf.scores.items()}))

    # ---- the noise floor ---------------------------------------------------
    # Landmarks jitter by a pixel or two even on a motionless person. Measured
    # by differencing every frame, that jitter accumulated into "movement":
    # a still person read as walking across the module, which drove Eating,
    # Idle and Experiment to exactly zero. These guard the fix.
    from _harness import pose_at
    from app.features import extract

    for jitter_px in (1.0, 2.0, 3.0):
        rng = np.random.default_rng(3)
        window = PoseWindow()
        sigma = jitter_px / config.FRAME_WIDTH
        for i in range(int(2.5 * FPS)):
            points = body_points((-0.50, -0.15), (0.50, -0.15)).astype(np.float64)
            points += rng.normal(0, sigma / 0.25, points.shape)
            hip = (0.5 + rng.normal(0, sigma), 0.5 + rng.normal(0, sigma))
            pose = pose_at(9000.0 + i / FPS, (0, 0), (0, 0), hip=hip)
            pose.points = points.astype(np.float32)
            window.push(pose)
        f = extract(window)

        r.check(f.body_speed < config.BODY_SPEED_STILL,
                f"a still person reads as still at {jitter_px:.0f}px jitter",
                f"body_speed {f.body_speed:.4f} vs "
                f"{config.BODY_SPEED_STILL}")
        r.check(f.wrist_speed < config.WRIST_SPEED_STILL,
                f"still hands read as still at {jitter_px:.0f}px jitter",
                f"wrist_speed {f.wrist_speed:.3f} vs {config.WRIST_SPEED_STILL}")

    # Every activity must survive realistic noise, not just clean input.
    noisy_ok = 0
    for expected, motion, seconds, hip_fn in SCENARIOS:
        rng = np.random.default_rng(11)
        window, clf = PoseWindow(), ActivityClassifier()
        sigma = 2.0 / config.FRAME_WIDTH
        for i in range(int(seconds * FPS)):
            t = i * DT
            wl, wr = motion(t)
            points = body_points(wl, wr).astype(np.float64)
            points += rng.normal(0, sigma / 0.25, points.shape)
            hip = hip_fn(t) if hip_fn else (0.5, 0.5)
            hip = (hip[0] + rng.normal(0, sigma), hip[1] + rng.normal(0, sigma))
            pose = pose_at(9500.0 + t, (0, 0), (0, 0), hip=hip)
            pose.points = points.astype(np.float32)
            window.push(pose)
            with frozen(9500.0 + t):
                clf.update(window)
        noisy_ok += r.check(clf.label == expected,
                            f"recognises {expected} through camera noise",
                            f"got {clf.label}")
    r.note(f"{noisy_ok}/{len(SCENARIOS)} activities survive 2px landmark jitter")

    # ---- an empty window must not crash or invent an activity -------------
    clf = ActivityClassifier()
    label, confidence = clf.update(PoseWindow())
    r.check(label == config.ACT_NO_CREW and confidence == 0.0,
            "an empty window reports no crew")

    return r


if __name__ == "__main__":
    run().report()
