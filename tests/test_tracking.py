"""Crew identity tracking.

v1 ships with MAX_CREW = 1, but the tracker is written for a whole crew. These
tests drive it at capacity 6 explicitly, so the multi-crew path stays honest
and can be switched back on by changing one config value.
"""

import math

from _harness import FPS, Results, frozen, landmarks_at
from app import config
from app.normalize import normalise
from app.tracking import CrewTracker

W, H = config.FRAME_WIDTH, config.FRAME_HEIGHT

IDLE = ((-0.50, -0.15), (0.50, -0.15))


def exercise(t):
    y = -0.55 + 0.55 * math.sin(2 * math.pi * 1.6 * t)
    x = 0.55 + 0.15 * math.sin(2 * math.pi * 1.6 * t)
    return (-x, y), (x, y)


def experiment(t):
    dx = 0.05 * math.sin(2 * math.pi * 0.9 * t)
    dy = 0.04 * math.sin(2 * math.pi * 1.3 * t)
    return (-0.30 + dx, -0.55 + dy), (0.30 + dx, -0.55 - dy)


def feed(tracker, crew_fn, seconds, base):
    """Push simulated frames of several people through the tracker."""
    for i in range(int(seconds * FPS)):
        now = base + i / FPS
        poses = []
        for landmarks in crew_fn(i / FPS):
            pose = normalise(landmarks, W, H)
            if pose is not None:
                pose.timestamp = now
                poses.append(pose)
        with frozen(now):
            tracker.update(poses, now=now)
    return tracker


def run() -> Results:
    r = Results("crew tracking")

    # ---- three people, three different activities, at once -----------------
    def three(t):
        return [
            landmarks_at(0.20, 0.55, 0.22, *exercise(t)),
            landmarks_at(0.50, 0.55, 0.22, *experiment(t)),
            landmarks_at(0.80, 0.55, 0.22, *IDLE),
        ]

    tracker = feed(CrewTracker(max_crew=6), three, 5.0, 1000.0)
    labels = {t.name: t.label for t in tracker.tracks}
    r.check(len(tracker.tracks) == 3, "three people produce three tracks",
            f"got {len(tracker.tracks)}")
    r.check(set(labels.values()) == {config.ACT_EXERCISE, config.ACT_EXPERIMENT,
                                     config.ACT_IDLE},
            "each member is classified independently", str(labels))

    # ---- identities survive movement ---------------------------------------
    def moving(t):
        return [
            landmarks_at(0.20 + 0.16 * math.sin(0.5 * t), 0.55, 0.22, *IDLE),
            landmarks_at(0.50, 0.55 + 0.05 * math.sin(0.7 * t), 0.22, *IDLE),
            landmarks_at(0.80 - 0.16 * math.sin(0.5 * t), 0.55, 0.22, *IDLE),
        ]

    tracker = CrewTracker(max_crew=6)
    seen = []
    base = 2000.0
    for i in range(int(8 * FPS)):
        now = base + i / FPS
        poses = []
        for landmarks in moving(i / FPS):
            pose = normalise(landmarks, W, H)
            if pose is not None:
                pose.timestamp = now
                poses.append(pose)
        with frozen(now):
            tracker.update(poses, now=now)
        if i % FPS == 0:
            seen.append(tuple(sorted(t.id for t in tracker.tracks)))
    r.check(len(set(seen)) == 1 and len(seen[0]) == 3,
            "identities never churn while people move about", str(set(seen)))

    # ---- the requested capacity --------------------------------------------
    spots = [(0.12, 0.45), (0.30, 0.62), (0.48, 0.42),
             (0.66, 0.63), (0.84, 0.45), (0.95, 0.62)]

    def six(t):
        return [landmarks_at(x, y, 0.15, *IDLE) for x, y in spots]

    tracker = feed(CrewTracker(max_crew=6), six, 4.0, 3000.0)
    r.check(len(tracker.tracks) == 6, "tracks six crew members at once",
            f"got {len(tracker.tracks)}")
    r.check([t.name for t in tracker.tracks]
            == [f"CM-{i}" for i in range(1, 7)],
            "crew names are stable and ordered",
            str([t.name for t in tracker.tracks]))

    # ---- capacity is a hard limit ------------------------------------------
    def eight(t):
        extra = spots + [(0.05, 0.80), (0.40, 0.85)]
        return [landmarks_at(x, y, 0.13, *IDLE) for x, y in extra]

    tracker = feed(CrewTracker(max_crew=6), eight, 3.0, 4000.0)
    r.check(len(tracker.tracks) <= 6, "never exceeds the configured capacity",
            f"got {len(tracker.tracks)}")

    # ---- someone leaves; everyone else keeps their identity ----------------
    tracker = CrewTracker(max_crew=6)
    base = 5000.0
    present = [(0.25, 0.55), (0.50, 0.55), (0.75, 0.55)]
    for i in range(int(4 * FPS)):
        now = base + i / FPS
        poses = []
        for x, y in present:
            pose = normalise(landmarks_at(x, y, 0.22, *IDLE), W, H)
            pose.timestamp = now
            poses.append(pose)
        with frozen(now):
            tracker.update(poses, now=now)
    before = {t.name: t.id for t in tracker.tracks}

    base = now + 0.1
    for i in range(int((config.TRACK_EXPIRY_SECONDS + 3) * FPS)):
        step = base + i / FPS
        poses = []
        for x, y in (present[0], present[2]):        # the middle one leaves
            pose = normalise(landmarks_at(x, y, 0.22, *IDLE), W, H)
            pose.timestamp = step
            poses.append(pose)
        with frozen(step):
            tracker.update(poses, now=step)
    after = {t.name: t.id for t in tracker.tracks}

    r.check(len(after) == 2, "the departed member is retired",
            f"{len(after)} still tracked")
    r.check(all(after.get(name) == before.get(name) for name in after),
            "the remaining members keep their original identities",
            f"{before} -> {after}")

    # ---- a brief occlusion must not reset anyone ---------------------------
    tracker = CrewTracker(max_crew=6)
    base = 6000.0
    for i in range(int(3 * FPS)):
        now = base + i / FPS
        pose = normalise(landmarks_at(0.5, 0.55, 0.22, *IDLE), W, H)
        pose.timestamp = now
        with frozen(now):
            tracker.update([pose], now=now)
    original = tracker.tracks[0].id

    for i in range(int(0.5 * FPS)):                  # half a second unseen
        with frozen(now + i / FPS):
            tracker.update([], now=now + i / FPS)
    r.check(len(tracker.tracks) == 1 and tracker.tracks[0].id == original,
            "a brief occlusion keeps the crew member and their history")

    # ---- the shipped default -----------------------------------------------
    r.check(CrewTracker().max_crew == config.MAX_CREW,
            f"default capacity follows config (MAX_CREW = {config.MAX_CREW})")

    return r


if __name__ == "__main__":
    run().report()
