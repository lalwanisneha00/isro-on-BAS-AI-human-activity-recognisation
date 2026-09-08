"""Measure recognition accuracy over the labelled clip dataset.

    python tools/evaluate.py                 evaluate everything
    python tools/evaluate.py --recorded-only ignore generated clips
    python tools/evaluate.py --save baseline name this run for comparison
    python tools/evaluate.py --compare baseline  show the change since then

Every clip is replayed through the real pipeline - the same normaliser, window
and classifier the console uses - so a number here means the same thing it
means on screen.

Two honesty rules are built in:

* The classifier cannot know anything until its rolling window fills, so it is
  guaranteed wrong at the start of every clip. Scores are reported after that
  warm-up, and the warm-up cost is reported separately as "settle time".
* Generated clips prove the code still runs; only clips recorded from life are
  evidence it works on people. They are counted separately and never merged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                   # noqa: E402
from app.classifier import ActivityClassifier            # noqa: E402
from app.dataset import DATASET_ROOT, load_dataset, summarise  # noqa: E402
from app.normalize import PoseWindow, normalise          # noqa: E402
from app.tracking import CrewTracker                     # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "dataset" / "_results"

# States that are the system saying "not yet" rather than naming an activity.
NON_ANSWERS = {config.ACT_ACQUIRING, config.ACT_NO_CREW}

# Some activities are defined by how long they have gone on, so they cannot be
# detected from the first frame however good the classifier is. An anomaly is
# sustained stillness: for its first ANOMALY_STILL_SECONDS the correct answer
# really is Idle. Scoring those frames as mistakes would punish the system for
# obeying its own specification - and would also wreck Idle's precision. They
# are excluded, and the wait is reported as detection latency instead.
CLASS_LEAD_IN = {config.ACT_ANOMALY: config.ANOMALY_STILL_SECONDS}


class _Landmark:
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = float(x), float(y), float(visibility)


class _Clock:
    """Replays a clip on its own timeline instead of in real time."""

    def __init__(self):
        self.now = 100_000.0
        self._real = time.time

    def __enter__(self):
        time.time = lambda: self.now
        return self

    def __exit__(self, *exc):
        time.time = self._real


def evaluate_clip(clip, clock) -> dict:
    """Replay one clip, returning what the system said frame by frame."""
    window = PoseWindow()
    classifier = ActivityClassifier()

    predictions, ready_from = [], None
    base = clock.now

    for index in range(clip.frames):
        clock.now = base + index / clip.fps

        landmarks = [_Landmark(*row) for row in clip.points[index][0]]
        # A frame the recorder marked as "nobody found" has zero visibility.
        pose = None
        if max(row[2] for row in clip.points[index][0]) > 0:
            pose = normalise(landmarks, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if pose is not None:
            pose.timestamp = clock.now

        window.push(pose)
        label, confidence = classifier.update(window)

        if ready_from is None and label not in NON_ANSWERS:
            ready_from = index
        predictions.append((label, confidence))

    clock.now = base + clip.frames / clip.fps + 1.0
    return {"predictions": predictions, "ready_from": ready_from}


def score(clips, crew_capacity=1) -> dict:
    """Run the whole dataset and gather the numbers."""
    labels = list(config.ACTIVITY_ORDER)
    index_of = {name: i for i, name in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels) + 1), dtype=int)  # +1 = no answer
    per_clip, settle_times, flips_per_minute = [], [], []
    detection_delays = []
    scored_frames = correct_frames = 0

    with _Clock() as clock:
        for clip in clips:
            outcome = evaluate_clip(clip, clock)
            predictions = outcome["predictions"]
            ready_from = outcome["ready_from"]

            truth = clip.label
            if truth not in index_of:
                continue

            # Only frames after the window filled are scored, and after any
            # lead-in this activity is defined to need.
            start = ready_from if ready_from is not None else len(predictions)
            lead_in = CLASS_LEAD_IN.get(truth, 0.0)
            start = max(start, int(lead_in * clip.fps))
            usable = predictions[start:]

            # How long after the activity became detectable it was named.
            detected_at = None
            for offset, (label, _) in enumerate(usable):
                if label == truth:
                    detected_at = offset / clip.fps
                    break

            clip_correct = 0
            for label, _ in usable:
                column = index_of.get(label, len(labels))
                matrix[index_of[truth], column] += 1
                if label == truth:
                    clip_correct += 1

            scored_frames += len(usable)
            correct_frames += clip_correct

            # Instability: how often the answer changed during a clip that
            # should have been one steady activity.
            changes = sum(1 for a, b in zip(usable, usable[1:]) if a[0] != b[0])
            minutes = max(1e-6, len(usable) / clip.fps / 60.0)
            flips_per_minute.append(changes / minutes)

            settle = (ready_from / clip.fps) if ready_from is not None else None
            if settle is not None:
                settle_times.append(settle)
            if detected_at is not None:
                detection_delays.append(detected_at)

            per_clip.append({
                "name": clip.name,
                "label": truth,
                "accuracy": (clip_correct / len(usable)) if usable else 0.0,
                "flips": changes,
                "settle": settle,
                "synthetic": clip.is_synthetic,
                "seconds": clip.duration,
                "top_confusion": _top_confusion(usable, truth),
            })

    return {
        "labels": labels,
        "matrix": matrix,
        "frames": scored_frames,
        "correct": correct_frames,
        "accuracy": (correct_frames / scored_frames) if scored_frames else 0.0,
        "per_clip": per_clip,
        "settle_median": float(np.median(settle_times)) if settle_times else None,
        "detect_median": (float(np.median(detection_delays))
                          if detection_delays else None),
        "flips_median": float(np.median(flips_per_minute)) if flips_per_minute else 0.0,
        "clip_count": len(per_clip),
        "crew_capacity": crew_capacity,
    }


def _top_confusion(predictions, truth):
    """The wrong answer this clip gave most often."""
    tally = {}
    for label, _ in predictions:
        if label != truth:
            tally[label] = tally.get(label, 0) + 1
    if not tally:
        return None
    label, count = max(tally.items(), key=lambda kv: kv[1])
    return {"label": label, "share": count / max(1, len(predictions))}


def per_class(result) -> list:
    """Precision, recall and F1 for each activity."""
    labels, matrix = result["labels"], result["matrix"]
    rows = []
    for i, name in enumerate(labels):
        support = int(matrix[i].sum())
        true_positive = int(matrix[i, i])
        predicted = int(matrix[:, i].sum())

        recall = true_positive / support if support else None
        precision = true_positive / predicted if predicted else None
        if precision and recall and (precision + recall):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        no_answer = int(matrix[i, len(labels)])
        rows.append({"label": name, "support": support, "precision": precision,
                     "recall": recall, "f1": f1, "no_answer": no_answer})
    return rows


# ------------------------------------------------------------------ output --
def _pct(value):
    return "   -  " if value is None else f"{value:6.1%}"


def report(result, title) -> None:
    labels, matrix = result["labels"], result["matrix"]

    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

    if not result["frames"]:
        print("  No frames to score.")
        return

    print(f"  Clips {result['clip_count']}   "
          f"scored frames {result['frames']}   "
          f"overall accuracy {result['accuracy']:.1%}")
    if result["settle_median"] is not None:
        print(f"  Median settle time {result['settle_median']:.2f}s   "
              f"median label changes per minute {result['flips_median']:.1f}")
    if result.get("detect_median") is not None:
        print(f"  Median time to name the activity once detectable "
              f"{result['detect_median']:.2f}s")
    if CLASS_LEAD_IN:
        excluded = ", ".join(f"{name} (first {seconds:.0f}s)"
                             for name, seconds in CLASS_LEAD_IN.items())
        print(f"  Excluded by definition: {excluded}")
    print()

    print(f"  {'activity':24} {'precis':>7} {'recall':>7} {'F1':>7} "
          f"{'frames':>7} {'no answer':>10}")
    print("  " + "-" * 68)
    for row in per_class(result):
        if not row["support"]:
            continue
        print(f"  {row['label']:24} {_pct(row['precision'])} {_pct(row['recall'])} "
              f"{_pct(row['f1'])} {row['support']:>7} {row['no_answer']:>10}")

    # ---- confusion matrix ----------------------------------------------
    print()
    print("  Confusion matrix   rows = what it really was, "
          "columns = what the system said")
    header = "".join(f"{name[:6]:>8}" for name in labels) + f"{'(none)':>8}"
    print(f"  {'':24}{header}")
    for i, name in enumerate(labels):
        if not matrix[i].sum():
            continue
        cells = ""
        for j in range(len(labels) + 1):
            value = matrix[i, j]
            cells += f"{'.':>8}" if value == 0 else f"{value:>8}"
        print(f"  {name:24}{cells}")

    # ---- worst clips ----------------------------------------------------
    worst = sorted(result["per_clip"], key=lambda c: c["accuracy"])[:6]
    if worst and worst[0]["accuracy"] < 0.95:
        print()
        print("  Weakest clips")
        for clip in worst:
            confusion = clip["top_confusion"]
            note = (f"mostly called {confusion['label']} "
                    f"({confusion['share']:.0%})") if confusion else "correct"
            tag = " [generated]" if clip["synthetic"] else ""
            print(f"    {clip['accuracy']:6.1%}  {clip['label']:22} "
                  f"{clip['flips']:>3} flips  {note}{tag}")


def compact(result) -> dict:
    """The numbers worth storing for a before/after comparison."""
    return {
        "accuracy": result["accuracy"],
        "frames": result["frames"],
        "clips": result["clip_count"],
        "settle_median": result["settle_median"],
        "flips_median": result["flips_median"],
        "detect_median": result.get("detect_median"),
        "per_class": {row["label"]: {"precision": row["precision"],
                                     "recall": row["recall"],
                                     "f1": row["f1"],
                                     "support": row["support"]}
                      for row in per_class(result) if row["support"]},
    }


def show_comparison(previous, current) -> None:
    print()
    print("=" * 78)
    print(f"  Change since '{previous['name']}' ({previous.get('saved_at', '?')})")
    print("=" * 78)

    was, now = previous["overall"]["accuracy"], current["accuracy"]
    arrow = "improved" if now > was else ("worse" if now < was else "unchanged")
    print(f"  Overall accuracy  {was:.1%}  ->  {now:.1%}   ({arrow}, "
          f"{now - was:+.1%})")

    old_flips = previous["overall"].get("flips_median")
    if old_flips is not None:
        print(f"  Label changes/min {old_flips:.1f}  ->  "
              f"{current['flips_median']:.1f}")
    print()

    print(f"  {'activity':24} {'F1 before':>10} {'F1 after':>10} {'change':>9}")
    print("  " + "-" * 58)
    old_classes = previous["overall"].get("per_class", {})
    for row in per_class(current):
        if not row["support"]:
            continue
        before = (old_classes.get(row["label"]) or {}).get("f1")
        after = row["f1"]
        if before is None and after is None:
            continue
        change = (f"{after - before:+.1%}"
                  if (before is not None and after is not None) else "new")
        print(f"  {row['label']:24} {_pct(before)}     {_pct(after)}    {change:>9}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure recognition accuracy")
    parser.add_argument("--recorded-only", action="store_true",
                        help="ignore generated clips")
    parser.add_argument("--save", metavar="NAME",
                        help="store these results under a name")
    parser.add_argument("--compare", metavar="NAME",
                        help="show the change since a stored result")
    parser.add_argument("--dataset", default=str(DATASET_ROOT))
    args = parser.parse_args()

    clips = load_dataset(args.dataset, include_synthetic=not args.recorded_only)
    if not clips:
        print()
        print(f"  No clips found in {args.dataset}")
        print("  Record some with:  python tools/record_clip.py")
        print()
        return 1

    # ---- what the dataset actually contains ----------------------------
    summary = summarise(clips)
    print()
    print(f"  Dataset: {len(clips)} clips, "
          f"{sum(c.duration for c in clips):.0f}s total")
    print(f"  {'activity':24} {'clips':>6} {'recorded':>9} "
          f"{'generated':>10} {'seconds':>8}")
    print("  " + "-" * 62)
    for name in config.ACTIVITY_ORDER:
        entry = summary.get(name)
        if not entry:
            print(f"  {name:24} {'-':>6} {'-':>9} {'-':>10} {'-':>8}")
            continue
        print(f"  {name:24} {entry['clips']:>6} {entry['recorded']:>9} "
              f"{entry['synthetic']:>10} {entry['seconds']:>8.0f}")

    recorded = [c for c in clips if not c.is_synthetic]
    generated = [c for c in clips if c.is_synthetic]

    overall = score(clips)
    report(overall, "ALL CLIPS")

    if recorded and generated:
        report(score(recorded), "RECORDED FROM LIFE ONLY  (the number that counts)")

    if args.compare:
        path = RESULTS_DIR / f"{args.compare}.json"
        if path.exists():
            show_comparison(json.loads(path.read_text()), overall)
        else:
            print(f"\n  No stored result named '{args.compare}'.")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": args.save,
            "saved_at": time.strftime("%Y-%m-%d %H:%M"),
            "crew_capacity": config.MAX_CREW,
            "overall": compact(overall),
            "recorded_only": compact(score(recorded)) if recorded else None,
        }
        path = RESULTS_DIR / f"{args.save}.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"\n  Saved as '{args.save}'. Compare later with: "
              f"python tools/evaluate.py --compare {args.save}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
