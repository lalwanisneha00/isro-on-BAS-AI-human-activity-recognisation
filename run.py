"""Start the BAS Crew Activity Recognition console.

    python run.py              live webcam
    python run.py --demo       start in Demo Mode, no webcam needed
    python run.py --port 8100  use a different port
    python run.py --no-browser don't open a browser window

Everything runs locally. Nothing is uploaded, and no network is needed once
the dependencies are installed.
"""

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Redirected output is block-buffered, which would hold the start-up banner
# back until the process ended. The banner is the only instruction the user
# gets, so it has to appear immediately.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

from app import config  # noqa: E402

BANNER = r"""
   ___  ___   ___    ___                    __  __         _ _
  | _ )/ _ \ / __|  / __|_ _ _____ __ ___  |  \/  |___ _ _ (_) |_ ___ _ _
  | _ \ (_) |\__ \ | (__| '_/ -_) V  V /   | |\/| / _ \ ' \| |  _/ _ \ '_|
  |___/\___/ |___/  \___|_| \___|\_/\_/    |_|  |_\___/_||_|_|\__\___/_|
"""


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def _console_already_running(host: str, port: int) -> bool:
    """True if the thing holding this port is another copy of this console."""
    try:
        import json
        import urllib.request
        with urllib.request.urlopen(
                f"http://{host}:{port}/api/status", timeout=1.0) as response:
            return "camera_status" in json.loads(response.read().decode())
    except Exception:
        return False


def _resolve_port(host: str, preferred: int, explicit: bool) -> int:
    """Pick a port, refusing to start a second console fighting for the webcam.

    Quietly moving to the next free port turned out to be a trap: running the
    command again without stopping the first copy started another console, and
    every copy opened the same webcam. Four of them ended up competing for it
    and the video died. A second console is now something you have to ask for.
    """
    if _port_is_free(host, preferred):
        return preferred

    if _console_already_running(host, preferred):
        if not explicit:
            raise SystemExit(f"""
  The console is already running at http://{host}:{preferred}

  Open that address in your browser.

  To restart it, stop the other one first: press CTRL+C in its terminal
  window, or close that window, then run this again.

  Starting a second copy would make both of them fight over the webcam,
  and neither would get a picture. If you genuinely want two consoles,
  pass a different --port.
""")
        print(f"  Note: another console is already on port {preferred}.")

    for offset in range(1, 10):
        candidate = preferred + offset
        if _port_is_free(host, candidate):
            print(f"  Note: port {preferred} is in use by another program, "
                  f"using {candidate} instead.")
            return candidate

    raise SystemExit(f"  Ports {preferred}-{preferred + 9} are all in use. "
                     f"Close the other program, or pass --port.")


def _preflight() -> list:
    """Warn about anything missing, without refusing to start."""
    warnings = []

    model = ROOT / config.POSE_MODEL_PATH
    if not model.exists():
        warnings.append(
            f"Pose model not found at {config.POSE_MODEL_PATH}.\n"
            "     Live tracking will be unavailable. See the README to restore it.")

    clip = ROOT / config.DEMO_DIR / config.DEMO_VIDEO_NAME
    if not clip.exists():
        warnings.append(
            "Demo clip not found. Demo Mode will be unavailable.\n"
            "     Create it with:  python tools/make_demo_clip.py")

    return warnings


def _open_browser_later(url: str) -> None:
    """Open the console once the server is actually accepting connections."""
    def wait_and_open():
        for _ in range(60):
            time.sleep(0.25)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.25)
                if probe.connect_ex((config.HOST, config.PORT)) == 0:
                    webbrowser.open(url)
                    return

    threading.Thread(target=wait_and_open, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BAS Crew Activity Recognition console")
    parser.add_argument("--demo", action="store_true",
                        help="start in Demo Mode, without using the webcam")
    parser.add_argument("--port", type=int, default=None,
                        help=f"port to serve on (default {config.PORT})")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="webcam index if the default is the wrong camera")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window automatically")
    args = parser.parse_args()

    config.CAMERA_INDEX = args.camera
    config.START_IN_DEMO_MODE = args.demo
    config.PORT = _resolve_port(config.HOST,
                                args.port if args.port else config.PORT,
                                explicit=args.port is not None)

    url = f"http://{config.HOST}:{config.PORT}"

    print(BANNER)
    print(f"  {config.MISSION_NAME} - {config.MODULE_NAME}")
    print()
    for warning in _preflight():
        print(f"  !  {warning}")
    print(f"  Mode      : {'Demo Mode (no webcam)' if args.demo else 'Live webcam'}")
    print(f"  Console   : {url}")
    print(f"  Logs      : {config.LOG_DIR}/{config.LOG_CSV_NAME}")
    print()
    print("  Press CTRL+C to stop.")
    print()
    # MediaPipe's C++ core logs before Python logging exists, so these cannot
    # be silenced from here. Naming them stops them reading as failures.
    print("  (MediaPipe prints a few XNNPACK / feedback-manager notices below.")
    print("   They are normal and can be ignored.)")
    print()

    if not args.no_browser:
        _open_browser_later(url)

    import uvicorn
    try:
        uvicorn.run("app.server:app", host=config.HOST, port=config.PORT,
                    log_level="warning")
    except KeyboardInterrupt:
        pass
    print("\n  Console stopped. Activity log saved.")


if __name__ == "__main__":
    main()
