"""Download the YOLO weights once, into models/.

Runtime is fully offline; this is a setup step, like fetching the pose model.

    python tools/fetch_yolo.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def main() -> int:
    target = ROOT / config.YOLO_WEIGHTS_PATH
    if target.exists():
        print(f"  Already present: {config.YOLO_WEIGHTS_PATH} "
              f"({target.stat().st_size / 1e6:.1f} MB)")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Fetching {config.YOLO_MODEL_NAME} weights...")
    try:
        from ultralytics import YOLO

        # Ultralytics downloads to its own cache on first use; copy the file
        # into the project so the repository is self-contained and the app
        # never reaches for the network at runtime.
        model = YOLO("yolov8n.pt")
        source = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) \
            else Path("yolov8n.pt")
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        elif not target.exists():
            raise RuntimeError(f"could not locate the downloaded weights at {source}")
    except Exception as exc:
        print(f"  Failed: {exc}")
        print("  The console still runs without object detection.")
        return 1

    print(f"  Saved {config.YOLO_WEIGHTS_PATH} "
          f"({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
