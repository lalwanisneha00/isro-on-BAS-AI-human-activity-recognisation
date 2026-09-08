"""Record labelled clips for measuring recognition accuracy.

    python tools/record_clip.py                 pick an activity from a menu
    python tools/record_clip.py --activity Exercise --seconds 12
    python tools/record_clip.py --list          show what has been recorded

A window opens showing you with your skeleton drawn on. You get a countdown,
then it records while you perform the activity, then it offers to keep or
discard the take. Both the pose track and the footage are saved: if the pose
detector is ever replaced, the clips can be re-extracted rather than re-filmed.

Keys while recording:  Q or ESC to abort the take.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                    # noqa: E402
from app.camera import CameraSource                       # noqa: E402
from app.dataset import (DATASET_ROOT, load_dataset, save_clip,  # noqa: E402
                         summarise)
from app.pose import PoseTracker, draw_skeleton           # noqa: E402

WINDOW = "BAS - record clip"
COUNTDOWN_SECONDS = 3

ACCENT = (255, 217, 0)
ALERT = (25, 178, 250)
WHITE = (255, 255, 255)
DIM = (170, 150, 120)


def _put(frame, text, xy, scale=0.6, colour=WHITE, weight=1):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                weight, cv2.LINE_AA)


def _banner(frame, lines, colour=ACCENT):
    """A readable panel at the top of the preview."""
    height = 26 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), (12, 9, 7), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    for i, line in enumerate(lines):
        _put(frame, line, (16, 30 + i * 26), 0.62,
             colour if i == 0 else DIM, 2 if i == 0 else 1)
    return frame


def choose_activity() -> str:
    """Menu of the activities the system knows about."""
    print()
    print("  Which activity are you about to perform?")
    print()
    for i, name in enumerate(config.ACTIVITY_ORDER, 1):
        print(f"    {i}. {name}")
    print()
    while True:
        answer = input("  Number (or blank to quit): ").strip()
        if not answer:
            return ""
        if answer.isdigit() and 1 <= int(answer) <= len(config.ACTIVITY_ORDER):
            return config.ACTIVITY_ORDER[int(answer) - 1]
        print("  Not one of the options.")


def show_dataset() -> None:
    clips = load_dataset()
    summary = summarise(clips)
    print()
    print(f"  Dataset: {DATASET_ROOT}")
    print()
    print(f"  {'activity':24} {'clips':>6} {'recorded':>9} {'synthetic':>10} "
          f"{'seconds':>8}  subjects")
    print("  " + "-" * 74)
    for name in config.ACTIVITY_ORDER:
        entry = summary.get(name)
        if not entry:
            print(f"  {name:24} {0:>6} {0:>9} {0:>10} {0:>8.0f}  -")
            continue
        subjects = ", ".join(sorted(entry["subjects"]))
        print(f"  {name:24} {entry['clips']:>6} {entry['recorded']:>9} "
              f"{entry['synthetic']:>10} {entry['seconds']:>8.0f}  {subjects}")
    print("  " + "-" * 74)
    recorded = sum(1 for c in clips if not c.is_synthetic)
    print(f"  {len(clips)} clips total, {recorded} of them recorded from life")
    print()


def record(activity, seconds, subject, camera_index, save_video, notes):
    """Run one take. Returns the saved path, or None if it was discarded."""
    camera = CameraSource(index=camera_index).start()
    tracker = PoseTracker()

    if not tracker.available:
        print(f"  ! Pose model unavailable: {tracker.error}")
        camera.stop()
        return None

    print(f"  Opening camera... (activity: {activity}, {seconds}s)")
    for _ in range(40):
        if camera.status == "live":
            break
        time.sleep(0.25)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    points, frames_kept = [], []
    aborted = False

    try:
        # ---- framing: wait until the crew member is actually detected ------
        ready_since = None
        while True:
            frame = camera.read()
            poses = tracker.detect(frame)
            detected = bool(poses)
            if detected:
                frame = draw_skeleton(frame, poses[0], ACCENT)

            if detected:
                ready_since = ready_since or time.time()
            else:
                ready_since = None

            held = (time.time() - ready_since) if ready_since else 0.0
            view = _banner(frame, [
                f"ACTIVITY: {activity}",
                "Stand where your whole upper body is visible."
                if not detected else
                f"Detected - hold still {max(0, 1.5 - held):.1f}s to begin",
                "SPACE to start now  |  Q to cancel",
            ], ACCENT if detected else ALERT)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                aborted = True
                break
            if key == 32 and detected:
                break
            if held >= 1.5:
                break

        # ---- countdown ------------------------------------------------------
        if not aborted:
            start = time.time()
            while True:
                remaining = COUNTDOWN_SECONDS - (time.time() - start)
                if remaining <= 0:
                    break
                frame = camera.read()
                poses = tracker.detect(frame)
                if poses:
                    frame = draw_skeleton(frame, poses[0], ACCENT)
                view = _banner(frame, [
                    f"STARTING IN {int(remaining) + 1}",
                    f"Get ready to perform: {activity}",
                ])
                cv2.imshow(WINDOW, view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    aborted = True
                    break

        # ---- the take -------------------------------------------------------
        if not aborted:
            started = time.time()
            missed = 0
            while True:
                elapsed = time.time() - started
                if elapsed >= seconds:
                    break

                frame = camera.read()
                poses = tracker.detect(frame)

                if poses:
                    row = np.array([[lm.x, lm.y, getattr(lm, "visibility", 1.0)]
                                    for lm in poses[0]], dtype=np.float32)
                    frame_out = draw_skeleton(frame.copy(), poses[0], ACCENT)
                else:
                    # Keep the timeline honest: a frame where nobody was found
                    # is stored as zero-visibility, not silently dropped.
                    row = np.zeros((33, 3), dtype=np.float32)
                    missed += 1
                    frame_out = frame.copy()

                points.append(row)
                if save_video:
                    frames_kept.append(frame)

                progress = elapsed / seconds
                bar = int(progress * (frame_out.shape[1] - 32))
                cv2.rectangle(frame_out, (16, frame_out.shape[0] - 26),
                              (16 + bar, frame_out.shape[0] - 16), ACCENT, -1)
                view = _banner(frame_out, [
                    f"RECORDING  {seconds - elapsed:4.1f}s left",
                    f"{activity}   |   {len(points)} frames"
                    + ("   (no pose in some frames)" if missed else ""),
                ], ALERT)
                cv2.imshow(WINDOW, view)

                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    aborted = True
                    break

        # ---- keep or discard -------------------------------------------------
        if aborted or len(points) < config.TARGET_FPS:
            print("  Take discarded.")
            return None

        found = int(sum(1 for row in points if row[:, 2].max() > 0))
        coverage = found / len(points)
        while True:
            frame = camera.read()
            view = _banner(frame, [
                "KEEP THIS TAKE?",
                f"{len(points)} frames, pose found in {coverage:.0%} of them",
                "K to keep   |   R to redo   |   Q to discard",
            ], ACCENT if coverage > 0.8 else ALERT)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("k"):
                break
            if key == ord("r"):
                print("  Redoing.")
                return "redo"
            if key in (ord("q"), 27):
                print("  Take discarded.")
                return None

    finally:
        cv2.destroyWindow(WINDOW)
        tracker.close()
        camera.stop()

    meta = {
        "source": "recorded",
        "subject": subject,
        "pose_coverage": round(coverage, 3),
        "camera_index": camera_index,
    }
    if notes:
        meta["notes"] = notes

    path = save_clip(np.stack(points), activity, config.TARGET_FPS, meta)

    if save_video and frames_kept:
        writer = cv2.VideoWriter(str(path.with_suffix(".mp4")),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 config.TARGET_FPS,
                                 (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        if writer.isOpened():
            for frame in frames_kept:
                writer.write(frame)
            writer.release()

    print(f"  Saved: {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    print(f"         {len(points)} frames, pose found in {coverage:.0%}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labelled activity clips")
    parser.add_argument("--activity", help="activity label; omit to pick from a menu")
    parser.add_argument("--seconds", type=float, default=12.0,
                        help="length of each take (default 12)")
    parser.add_argument("--subject", default="crew",
                        help="who is performing, so clips can be grouped")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--no-video", action="store_true",
                        help="save only the pose track, not the footage")
    parser.add_argument("--notes", default="", help="anything worth recording")
    parser.add_argument("--list", action="store_true",
                        help="show what has been recorded so far")
    args = parser.parse_args()

    if args.list:
        show_dataset()
        return

    while True:
        activity = args.activity or choose_activity()
        if not activity:
            break
        if activity not in config.ACTIVITY_ORDER:
            print(f"  Unknown activity: {activity}")
            print("  Expected one of: " + ", ".join(config.ACTIVITY_ORDER))
            return

        result = record(activity, args.seconds, args.subject, args.camera,
                        not args.no_video, args.notes)
        if result == "redo":
            continue

        if args.activity:              # one-shot mode
            break
        again = input("  Record another? [Y/n]: ").strip().lower()
        if again == "n":
            break

    show_dataset()


if __name__ == "__main__":
    main()
