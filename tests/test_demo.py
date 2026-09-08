"""Demo Mode: the shipped clip must recognise every activity it performs.

This is the closest thing to an end-to-end test. The clip's pose track goes
through the real normaliser, the real window and the real classifier, so a
regression anywhere in that chain shows up here as a scripted activity the
system can no longer recognise.
"""

from pathlib import Path

import numpy as np

from _harness import Landmark, Results, frozen
from app import config
from app.classifier import ActivityClassifier
from app.demo import DemoSource
from app.normalize import PoseWindow, normalise

DEMO_DIR = Path(__file__).resolve().parent.parent / config.DEMO_DIR

# Anomaly deliberately waits ANOMALY_STILL_SECONDS before firing, so it can
# never match for the whole of its phase. The others should match most of it.
THRESHOLDS = {config.ACT_ANOMALY: 0.25}
DEFAULT_THRESHOLD = 0.55


def run() -> Results:
    r = Results("demo mode")

    poses_path = DEMO_DIR / config.DEMO_POSE_NAME
    video_path = DEMO_DIR / config.DEMO_VIDEO_NAME

    if not poses_path.exists() or not video_path.exists():
        r.check(False, "demo clip is present",
                "run: python tools/make_demo_clip.py")
        return r
    r.check(True, "demo clip is present")

    # ---- the clip loads without eating memory -----------------------------
    source = DemoSource(DEMO_DIR)
    r.check(source.available, "demo clip loads", source.detail)
    r.check(source.duration > 30, "clip is long enough to show every activity",
            f"{source.duration:.0f}s")

    frame, crew = source.next()
    r.check(frame is not None and frame.shape[:2]
            == (config.FRAME_HEIGHT, config.FRAME_WIDTH),
            "playback yields correctly sized frames")
    r.check(len(crew) == source.crew_count and len(crew[0]) == 33,
            "playback yields 33 landmarks per crew member")

    # Looping back to the start must not fail or stall.
    source.rewind()
    for _ in range(5):
        frame, _ = source.next()
    r.check(frame is not None, "clip loops cleanly")
    source.close()

    # ---- every scripted activity is genuinely recognised ------------------
    data = np.load(poses_path, allow_pickle=False)
    points, labels = data["points"], data["labels"]
    fps = float(data["fps"][0])
    if points.ndim == 3:                       # single-crew clip
        points = points[:, None, :, :]
        labels = labels[:, None] if labels.ndim == 1 else labels

    window, classifier = PoseWindow(), ActivityClassifier()
    base = 50_000.0
    matched, total = {}, {}

    for i in range(len(points)):
        now = base + i / fps
        landmarks = [Landmark(*row) for row in points[i][0]]
        pose = normalise(landmarks, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if pose is not None:
            pose.timestamp = now
        window.push(pose)
        with frozen(now):
            classifier.update(window)

        truth = str(labels[i][0])
        total[truth] = total.get(truth, 0) + 1
        if classifier.label == truth:
            matched[truth] = matched.get(truth, 0) + 1

    for activity in config.ACTIVITY_ORDER:
        if activity not in total:
            continue
        share = matched.get(activity, 0) / total[activity]
        threshold = THRESHOLDS.get(activity, DEFAULT_THRESHOLD)
        r.check(share >= threshold, f"recognises {activity} from the clip",
                f"{share:.0%} of its frames, needed {threshold:.0%}")

    # The clip is regenerated whenever the task list grows, so name what it
    # is still missing rather than just failing with a count.
    missing = [a for a in config.ACTIVITY_ORDER if a not in total]
    r.check(len(total) >= 7,
            f"the clip performs a full activity rota ({len(total)} activities)",
            f"only {len(total)}")
    if missing:
        r.note("clip does not yet cover: " + ", ".join(missing))

    return r


if __name__ == "__main__":
    run().report()
