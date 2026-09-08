"""The measuring instrument itself has to be trustworthy.

If the evaluation harness is wrong, every accuracy number after this point is
wrong too, so it gets the same scrutiny as the classifier.
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from _harness import FPS, Results, body_points
from app import config
from app.dataset import load_clip, load_dataset, save_clip, slug, summarise

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


def _clip_points(motion, seconds, fps=FPS):
    """A pose track in MediaPipe's normalised-image form."""
    rows = []
    for i in range(int(seconds * fps)):
        pts = body_points(*motion(i / fps))
        row = np.zeros((33, 3), dtype=np.float32)
        row[:, 0] = pts[:, 0] * 0.25 + 0.5
        row[:, 1] = pts[:, 1] * 0.25 + 0.5
        row[:, 2] = 1.0
        rows.append(row)
    return np.stack(rows)


def steady(t):
    return (-0.50, -0.15), (0.50, -0.15)


def swinging(t):
    y = -0.55 + 0.55 * math.sin(2 * math.pi * 1.6 * t)
    x = 0.55 + 0.15 * math.sin(2 * math.pi * 1.6 * t)
    return (-x, y), (x, y)


def run() -> Results:
    r = Results("evaluation harness")
    scratch = Path(tempfile.mkdtemp(prefix="bas_test_ds_"))

    try:
        # ---- clips survive a round trip ---------------------------------
        points = _clip_points(swinging, 5.0)
        path = save_clip(points, config.ACT_EXERCISE, FPS,
                         meta={"subject": "tester"}, root=scratch)
        clip = load_clip(path)

        r.check(clip.label == config.ACT_EXERCISE, "label survives a round trip")
        r.check(clip.frames == len(points), "frame count survives",
                f"{clip.frames} vs {len(points)}")
        r.check(clip.crew_count == 1, "single-crew clips get a crew axis")
        r.check(abs(clip.duration - 5.0) < 0.1, "duration is right",
                f"{clip.duration:.2f}s")
        r.check(clip.subject == "tester", "metadata survives")
        r.check(not clip.is_synthetic, "recorded clips are not marked generated")
        r.check(np.allclose(clip.points[:, 0, :, :], points, atol=1e-5),
                "the pose track itself is unchanged")

        # ---- generated clips are kept separate ---------------------------
        save_clip(_clip_points(steady, 4.0), config.ACT_IDLE, FPS,
                  meta={"source": "demo-clip", "subject": "demo"}, root=scratch)

        everything = load_dataset(scratch)
        recorded_only = load_dataset(scratch, include_synthetic=False)
        r.check(len(everything) == 2 and len(recorded_only) == 1,
                "generated clips can be excluded from a run",
                f"{len(everything)} all, {len(recorded_only)} recorded")

        summary = summarise(everything)
        r.check(summary[config.ACT_EXERCISE]["recorded"] == 1
                and summary[config.ACT_IDLE]["synthetic"] == 1,
                "the dataset summary counts each kind separately")

        # ---- a damaged clip must not stop a run --------------------------
        (scratch / slug(config.ACT_IDLE) / "broken.npz").write_bytes(b"not an npz")
        r.check(len(load_dataset(scratch)) == 2,
                "a corrupt clip is skipped rather than crashing the harness")

        # ---- the evaluator agrees with the classifier --------------------
        import evaluate as harness

        result = harness.score(load_dataset(scratch))
        r.check(result["frames"] > 0, "the evaluator scores frames")
        r.check(0.0 <= result["accuracy"] <= 1.0, "accuracy is a proportion",
                str(result["accuracy"]))

        matrix_total = int(result["matrix"].sum())
        r.check(matrix_total == result["frames"],
                "every scored frame lands in the confusion matrix",
                f"{matrix_total} vs {result['frames']}")

        rows = harness.per_class(result)
        for row in rows:
            if not row["support"]:
                continue
            for metric in ("precision", "recall", "f1"):
                value = row[metric]
                if value is not None and not (0.0 <= value <= 1.0):
                    r.check(False, f"{row['label']} {metric} within 0..1",
                            str(value))
                    break
            else:
                continue
            break
        else:
            r.check(True, "precision, recall and F1 all stay within 0..1")

        # ---- warm-up frames are excluded, not scored as wrong ------------
        exercise = [c for c in load_dataset(scratch) if c.label == config.ACT_EXERCISE]
        scored = harness.score(exercise)
        r.check(scored["frames"] < exercise[0].frames,
                "warm-up frames are excluded from scoring",
                f"{scored['frames']} of {exercise[0].frames}")
        r.check(scored["settle_median"] is not None
                and scored["settle_median"] < config.WINDOW_SECONDS + 0.5,
                "settle time is reported and is about one window",
                str(scored["settle_median"]))

        # ---- an activity defined by duration gets its lead-in excluded ----
        r.check(config.ACT_ANOMALY in harness.CLASS_LEAD_IN,
                "the anomaly lead-in is declared, not silently scored as error")

        # ---- a clip of pure no-detection must not crash ------------------
        blank = np.zeros((int(4 * FPS), 33, 3), dtype=np.float32)
        save_clip(blank, config.ACT_IDLE, FPS,
                  meta={"subject": "empty"}, root=scratch)
        empty_result = harness.score(load_dataset(scratch))
        r.check(empty_result["frames"] >= 0,
                "a clip where nobody was detected is handled without crashing")

    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return r


if __name__ == "__main__":
    run().report()
