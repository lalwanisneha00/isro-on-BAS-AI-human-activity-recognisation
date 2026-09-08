"""Labelled clip storage for measuring recognition accuracy.

A clip is one recording of one crew member performing one activity, stored as
the extracted pose track plus - when available - the original video.

Keeping the video matters. The pose track is whatever the current detector
produced; if the detector is ever replaced, every saved clip can be
re-extracted from its footage instead of asking someone to record the whole
dataset again.

Layout:

    dataset/
      exercise/
        exercise_sneha_20260908_141203.npz     pose track + label + metadata
        exercise_sneha_20260908_141203.mp4     the footage it came from
      idle/
        ...
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"

# Clips whose poses were generated rather than filmed. They are useful for
# catching regressions, but they are not evidence the system works on people,
# so the evaluator always reports them separately.
SYNTHETIC_SOURCES = {"synthetic", "demo-clip"}


def slug(activity: str) -> str:
    """Folder-safe name for an activity label."""
    return re.sub(r"[^a-z0-9]+", "_", activity.lower()).strip("_")


ACTIVITY_BY_SLUG = {slug(name): name for name in config.ACTIVITY_ORDER}


class Clip:
    """One labelled recording."""

    def __init__(self, path, points, label, fps, meta):
        self.path = Path(path)
        self.points = points              # (frames, crew, 33, 3) x, y, visibility
        self.label = label
        self.fps = float(fps)
        self.meta = meta

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def crew_count(self) -> int:
        return int(self.points.shape[1])

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def source(self) -> str:
        return str(self.meta.get("source", "recorded"))

    @property
    def is_synthetic(self) -> bool:
        return self.source in SYNTHETIC_SOURCES

    @property
    def subject(self) -> str:
        return str(self.meta.get("subject", "unknown"))

    @property
    def video_path(self):
        candidate = self.path.with_suffix(".mp4")
        return candidate if candidate.exists() else None

    def __repr__(self):
        return (f"<Clip {self.name} {self.label!r} {self.duration:.1f}s "
                f"{self.crew_count} crew>")


def save_clip(points, label, fps, meta=None, root=None) -> Path:
    """Write one labelled clip. `points` is (frames, crew, 33, 3)."""
    root = Path(root) if root else DATASET_ROOT
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3:                       # single crew member
        points = points[:, None, :, :]
    if points.ndim != 4 or points.shape[2:] != (33, 3):
        raise ValueError(f"expected (frames, crew, 33, 3), got {points.shape}")

    meta = dict(meta or {})
    meta.setdefault("source", "recorded")
    meta.setdefault("recorded_at", datetime.now().isoformat(timespec="seconds"))

    folder = root / slug(label)
    folder.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    subject = slug(str(meta.get("subject", "crew")))
    path = folder / f"{slug(label)}_{subject}_{stamp}.npz"

    np.savez_compressed(
        path,
        points=points,
        label=np.array([label]),
        fps=np.array([float(fps)]),
        meta=np.array([json.dumps(meta)]),
    )
    return path


def load_clip(path) -> Clip:
    """Read one clip back, tolerating older single-crew files."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    points = data["points"]
    if points.ndim == 3:
        points = points[:, None, :, :]

    label = str(data["label"][0])
    fps = float(data["fps"][0]) if "fps" in data else float(config.TARGET_FPS)

    meta = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"][0]))
        except (ValueError, TypeError):
            meta = {}

    return Clip(path, points, label, fps, meta)


def load_dataset(root=None, include_synthetic=True) -> list:
    """Every clip on disk, sorted for a stable report order."""
    root = Path(root) if root else DATASET_ROOT
    if not root.exists():
        return []

    clips = []
    for path in sorted(root.rglob("*.npz")):
        try:
            clip = load_clip(path)
        except Exception:
            continue                       # a damaged clip must not stop a run
        if clip.is_synthetic and not include_synthetic:
            continue
        clips.append(clip)
    return clips


def summarise(clips: list) -> dict:
    """Counts and durations per activity, for the dataset report."""
    summary = {}
    for clip in clips:
        entry = summary.setdefault(clip.label, {
            "clips": 0, "seconds": 0.0, "recorded": 0, "synthetic": 0,
            "subjects": set(), "multi_crew": 0,
        })
        entry["clips"] += 1
        entry["seconds"] += clip.duration
        entry["synthetic" if clip.is_synthetic else "recorded"] += 1
        entry["subjects"].add(clip.subject)
        if clip.crew_count > 1:
            entry["multi_crew"] += 1
    return summary
