"""Single-subject locking: choose one person, and stay with them.

The failure this guards against is subtle and was the reason multi-crew was
removed: if the followed person silently changes partway through a window, the
classifier reads a blend of two people's movements and the answer is nonsense.
"""

import math

from _harness import FPS, Results, frozen, landmarks_at
from app import config
from app.normalize import normalise
from app.subject import SubjectLock

W, H = config.FRAME_WIDTH, config.FRAME_HEIGHT

IDLE = ((-0.50, -0.15), (0.50, -0.15))


def exercise(t):
    y = -0.55 + 0.55 * math.sin(2 * math.pi * 1.6 * t)
    x = 0.55 + 0.15 * math.sin(2 * math.pi * 1.6 * t)
    return (-x, y), (x, y)


def feed(lock, people_fn, seconds, base):
    """Push simulated frames of one or more people through the lock."""
    for i in range(int(seconds * FPS)):
        now = base + i / FPS
        poses = []
        for landmarks in people_fn(i / FPS):
            pose = normalise(landmarks, W, H)
            if pose is not None:
                pose.timestamp = now
                poses.append(pose)
        with frozen(now):
            lock.update(poses, now=now)
    return now


def run() -> Results:
    r = Results("single-subject lock")

    # ---- the closest person is chosen ------------------------------------
    def near_and_far(t):
        return [
            landmarks_at(0.30, 0.55, 0.13, *IDLE),      # far away, small
            landmarks_at(0.60, 0.60, 0.26, *exercise(t)),  # close, large
        ]

    lock = SubjectLock()
    feed(lock, near_and_far, 4.0, 1000.0)
    r.check(lock.locked, "a subject is locked")
    r.check(lock.candidates == 2 and lock.ignored == 1,
            "the second person is seen but deliberately not followed",
            f"candidates {lock.candidates}, ignored {lock.ignored}")
    r.check(lock.label == config.ACT_EXERCISE,
            "the classified activity is the CLOSE person's, not the far one's",
            f"got {lock.label}")

    # ---- a passer-by must not steal the lock ------------------------------
    # Somebody crosses the background, briefly appearing as large as the
    # subject. The lock must not move, and the activity must not change.
    def passer_by(t):
        people = [landmarks_at(0.35, 0.60, 0.24, *exercise(t))]
        if 1.0 < t < 2.2:                       # crosses for ~1.2s
            people.append(landmarks_at(0.70, 0.60, 0.27, *IDLE))
        return people

    lock = SubjectLock()
    feed(lock, passer_by, 5.0, 2000.0)
    r.check(lock.switches == 0, "a passer-by does not steal the lock",
            f"{lock.switches} switches")
    r.check(lock.label == config.ACT_EXERCISE,
            "the subject's activity is unaffected by someone walking past",
            f"got {lock.label}")

    # ---- somebody standing beside the subject -----------------------------
    def bystander(t):
        return [
            landmarks_at(0.35, 0.58, 0.25, *exercise(t)),
            landmarks_at(0.68, 0.58, 0.24, *IDLE),      # similar size, static
        ]

    lock = SubjectLock()
    feed(lock, bystander, 6.0, 3000.0)
    r.check(lock.switches == 0, "a bystander of similar size does not take over",
            f"{lock.switches} switches")
    r.check(lock.label == config.ACT_EXERCISE,
            "the subject keeps being classified, not the bystander",
            f"got {lock.label}")

    # ---- a brief occlusion keeps the subject and their history ------------
    lock = SubjectLock()
    last = feed(lock, lambda t: [landmarks_at(0.5, 0.58, 0.25, *exercise(t))],
                4.0, 4000.0)
    locked_at = lock.locked_at

    for i in range(int(0.8 * FPS)):             # under the grace period
        now = last + i / FPS
        with frozen(now):
            lock.update([], now=now)
    r.check(lock.locked, "the lock survives a brief occlusion")
    r.check(lock.locked_at == locked_at,
            "the subject was not re-acquired as a new person")

    # ---- a long absence releases the lock ---------------------------------
    for i in range(int((config.SUBJECT_GRACE_SECONDS + 1.5) * FPS)):
        now = last + 1.0 + i / FPS
        with frozen(now):
            lock.update([], now=now)
    r.check(not lock.locked, "a long absence releases the lock")
    r.check(lock.window.count == 0,
            "the window is cleared, so a new person starts with a clean history")

    # ---- switching subjects clears the buffer -----------------------------
    # If the lock does move, the new person must not be classified using the
    # previous person's movement history.
    lock = SubjectLock()
    feed(lock, lambda t: [landmarks_at(0.30, 0.58, 0.16, *exercise(t))],
         4.0, 5000.0)
    before = lock.locked_at

    def takeover(t):
        return [landmarks_at(0.70, 0.58, 0.30, *IDLE)]   # a different, closer person

    feed(lock, takeover, 5.0, 5100.0)
    r.check(lock.locked_at != before, "a genuine takeover re-locks",
            "lock time unchanged")
    r.check(lock.label == config.ACT_IDLE,
            "the new subject is classified on their own movement",
            f"got {lock.label}")

    # ---- nobody in frame at all ------------------------------------------
    lock = SubjectLock()
    with frozen(6000.0):
        result = lock.update([], now=6000.0)
    r.check(result is None and not lock.locked,
            "an empty frame locks nobody and does not crash")
    r.check(lock.state()["activity"] == config.ACT_NO_CREW,
            "an empty frame reports no crew", lock.state()["activity"])

    # ---- exactly one person is ever followed ------------------------------
    def crowd(t):
        return [landmarks_at(0.15 + 0.17 * i, 0.58, 0.20, *IDLE)
                for i in range(5)]

    lock = SubjectLock()
    feed(lock, crowd, 4.0, 7000.0)
    r.check(lock.ignored == lock.candidates - 1,
            "with five people present, four are ignored",
            f"candidates {lock.candidates}, ignored {lock.ignored}")

    r.check(config.ENABLE_MULTI_CREW is False,
            "multi-crew is off on the active path")

    return r


if __name__ == "__main__":
    run().report()
