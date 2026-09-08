"""Seed the evaluation dataset from the generated demo clip.

The demo clip already performs every activity with known labels, so it can be
split into one labelled clip per activity. That gives the evaluation harness
something to measure from the moment it is built, and it catches regressions
in the pipeline.

It is not evidence the system works on people. These clips are tagged
`source: demo-clip`, and the evaluator always reports them apart from footage
recorded from life.

    python tools/seed_dataset.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                              # noqa: E402
from app.dataset import DATASET_ROOT, save_clip, slug  # noqa: E402

MIN_SECONDS = 4.0


def main() -> int:
    poses_path = (Path(__file__).resolve().parent.parent
                  / config.DEMO_DIR / config.DEMO_POSE_NAME)
    if not poses_path.exists():
        print(f"  Demo pose track not found at {poses_path}")
        print("  Build it first:  python tools/make_demo_clip.py")
        return 1

    data = np.load(poses_path, allow_pickle=False)
    points, labels = data["points"], data["labels"]
    fps = float(data["fps"][0])

    if points.ndim == 3:                       # single-crew clip
        points = points[:, None, :, :]
    if labels.ndim == 1:
        labels = labels[:, None]

    # Remove anything seeded from a previous run, so re-seeding cannot pile up
    # duplicate copies of the same footage.
    removed = 0
    for existing in DATASET_ROOT.rglob("demo_*.npz"):
        existing.unlink()
        removed += 1
    if removed:
        print(f"  Replaced {removed} previously seeded clips")

    # Walk the label track and cut it into runs of one activity.
    written = 0
    start = 0
    track = [str(labels[i][0]) for i in range(len(labels))]

    for index in range(1, len(track) + 1):
        if index < len(track) and track[index] == track[start]:
            continue

        label, frames = track[start], index - start
        seconds = frames / fps
        if label in config.ACTIVITY_ORDER and seconds >= MIN_SECONDS:
            path = save_clip(
                points[start:index], label, fps,
                meta={"source": "demo-clip", "subject": "demo",
                      "notes": "cut from the generated Demo Mode clip"},
            )
            # A stable name, so re-seeding replaces rather than accumulates.
            path.rename(path.with_name(f"demo_{slug(label)}.npz"))
            written += 1
            print(f"  {label:24} {seconds:5.1f}s  ->  demo_{slug(label)}.npz")
        start = index

    print()
    print(f"  Seeded {written} clips into {DATASET_ROOT}")
    print("  These are generated, not filmed. Record real clips with:")
    print("      python tools/record_clip.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
