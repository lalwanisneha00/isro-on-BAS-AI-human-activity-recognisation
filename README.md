# BAS Crew Activity Recognition

**SIH26174 — AI Human Activity Recognition for Onboard BAS Experiments**

A mission-control console that watches a crew member through an ordinary
webcam and works out what they are doing — exercising, operating an experiment,
eating, doing maintenance, idle, moving through the module, or motionless long
enough to be a concern. It replaces the manual activity log a crew member would
otherwise have to fill in by hand.

Everything runs on the laptop it is started on. Once installed, it needs no
internet, no account, and no cloud service.

---

## Running it

```
python run.py
```

That is the whole thing. A browser opens at <http://127.0.0.1:8000> showing the
console.

If the webcam is busy, the room is dark, or you would rather not rely on either
during a presentation:

```
python run.py --demo
```

This replays a recorded crew feed that ships with the project and never touches
the camera.

Other options:

| Command | What it does |
|---|---|
| `python run.py --port 8100` | Serve on a different port |
| `python run.py --camera 1` | Use a second webcam instead of the built-in one |
| `python run.py --no-browser` | Don't open a browser automatically |
| `python tests/run_all.py` | Run the test suite |

Press **Ctrl+C** to stop. Stopping this way writes the activity that was in
progress to the log; force-closing the window loses only that last entry.

> On start-up MediaPipe prints a few notices about "XNNPACK delegate" and
> "feedback manager". These are normal and are not errors.

---

## First-time setup

You need **Python 3.10 or newer**. Then, from this folder:

```
pip install -r requirements.txt
```

That installs OpenCV, MediaPipe, FastAPI and Uvicorn.

Two files are needed that are not Python packages:

- **`models/pose_landmarker_lite.task`** — the body-tracking model (5.8 MB).
  If it is missing, download it once from
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`
  and save it into the `models/` folder.
- **`demo/demo_feed.mp4`** — the Demo Mode clip. If missing, rebuild it with
  `python tools/make_demo_clip.py`.

Both are already in place. This is the only step that ever needs the internet.

---

## What the console shows

**Crew Cabin Camera** — the live feed with a skeleton drawn over the crew
member, coloured by whatever they are currently doing.

**Crew Roster** — who is being tracked, their current activity, and how
confident the system is. Underneath, the score every activity received, so you
can see *why* a label was chosen rather than having to trust it.

**Cumulative Time** — total time spent on each activity this session.

**Activity Timeline** — the last five minutes, with one lane per activity.
Hover any block for its start time and duration.

**Activity Log** — every completed activity segment, newest first.

**Normalisation Telemetry** — the diagnostic panel. The left column of numbers
moves as you move; the right column must stay pinned. See below.

---

## The seven activities

| Activity | What triggers it |
|---|---|
| **Exercise** | Hands sweeping a wide arc, briskly and repeatedly |
| **Experiment Operation** | Hands working in a small area in front of the body |
| **Eating/Rest** | A hand returning to the face while the body stays put |
| **Maintenance** | Arms raised or reaching out, working in bursts |
| **Idle** | Present and calm |
| **In-Transit/Movement** | The whole body crossing the module |
| **Anomaly (No Motion)** | No movement at all for more than 8 seconds |

An anomaly turns the console amber and raises a banner across the top. Amber is
used for nothing else, so it can only ever mean one thing.

---

## How it works

Five stages, each in its own file:

1. **Capture** ([`app/camera.py`](app/camera.py)) — reads the webcam on its own
   thread. If the camera is unplugged, busy, or starts trickling frames instead
   of failing outright, it reconnects on its own.

2. **Pose** ([`app/pose.py`](app/pose.py)) — MediaPipe finds 33 body landmarks
   per frame.

3. **Normalisation** ([`app/normalize.py`](app/normalize.py)) — the important
   idea. Raw coordinates are useless aboard a station where a crew member can
   float at any distance, any angle, even upside down. Every pose is rewritten
   into a body-local frame: hips at the origin, distances divided by torso
   length, spine turned to point "up". After that the shoulder midpoint always
   sits at (0, −1), whatever the person is doing.

   That is what the Normalisation Telemetry panel demonstrates: walk towards the
   camera and *Torso length* changes a lot, while *Shoulder Y* stays at −1.0000.

4. **Features and classification**
   ([`app/features.py`](app/features.py), [`app/classifier.py`](app/classifier.py))
   — roughly two seconds of normalised poses are measured for hand speed, how
   far the hands travel, whether the movement genuinely repeats, where the hands
   are relative to the face and the body, and whether the whole body is moving.
   Each activity is scored from those measurements and the strongest wins.

   Scoring every activity, rather than running a chain of if/else tests, is what
   makes the confidence figure and the score breakdown meaningful.

5. **Logging** ([`app/logbook.py`](app/logbook.py)) — when the activity changes,
   the previous one is written as a single row to `logs/activity_log.csv` and
   `logs/activity_log.db`. Segments shorter than 1.5 seconds are discarded as
   tracking noise.

Two details worth knowing, because both were bugs before they were features:

- The analysis window is bounded by **time, not frame count**. A fixed frame
  count silently becomes a longer window when the frame rate drops, which
  shifts every threshold tuned against it.
- Repetition is measured by **autocorrelation**, not by counting direction
  changes. Counting reversals scored motionless hands at dozens of "reversals"
  from landmark jitter alone.

---

## Tuning it

Every threshold lives in [`app/config.py`](app/config.py), grouped and
commented. The ones most worth adjusting:

| Setting | Meaning |
|---|---|
| `ANOMALY_STILL_SECONDS` | How long stillness lasts before an alert (default 8) |
| `EXERCISE_MIN_RANGE` | How far hands must travel to count as exercise |
| `FOCUSED_MAX_RANGE` | How far they may travel and still be fine work |
| `WINDOW_SECONDS` | Length of the analysis window |
| `MIN_LOG_SECONDS` | Shortest segment worth writing to the log |
| `MAX_CREW` | Crew members tracked at once (see below) |

After changing anything, run `python tests/run_all.py` — the suite covers the
cases these values control, including the ones that are easy to break.

---

## Multi-crew

The tracker assigns stable identities to several people at once, keeps them
through brief occlusions, and gives each their own window, classifier and log
entries. It is tested at six crew members.

**It ships switched off** (`MAX_CREW = 1`). Asking MediaPipe for several poses
costs frame rate and admits weak, spurious detections, and both hurt accuracy
for the single-person case. Raising `MAX_CREW` turns it back on; regenerate the
demo clip afterwards with `python tools/make_demo_clip.py` so the clip contains
that many people.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| Amber **NO SIGNAL**, no video | Another app has the camera. Close Zoom/Teams/Camera and refresh. |
| **Pose Track: Searching** forever | Get your head, shoulders and hips all in frame; add light. |
| **Pose Track: Offline** | `models/pose_landmarker_lite.task` is missing — see setup. |
| Activity flickers or reads wrong | Improve lighting first; then tune the thresholds above. |
| Frame rate under 10 fps | Usually a dark room: the webcam lengthens its exposure. |
| Port already in use | It moves to the next free port and says so. |
| Demo Mode says clip not found | `python tools/make_demo_clip.py` |

---

## Layout

```
run.py                  start here
requirements.txt
app/
  config.py             every setting and threshold
  camera.py             webcam capture, reconnection, stall watchdog
  pose.py               MediaPipe landmarks + skeleton drawing
  normalize.py          body-local reference frame, rolling window
  features.py           motion measurements over the window
  classifier.py         scores the seven activities
  tracking.py           crew identities across frames
  logbook.py            CSV + SQLite activity log
  demo.py               Demo Mode playback
  pipeline.py           ties the stages together on one thread
  server.py             local web server and JSON endpoints
  web/                  the console itself
models/                 pose model
demo/                   Demo Mode clip and its pose track
tools/make_demo_clip.py rebuilds the demo clip
tests/run_all.py        the test suite
logs/                   written at runtime
```

---

## Scope

This is version 1, built for the internal hackathon round. The classifier is
rule-based: it needs no training data, and every decision it makes can be
traced to a measurement you can read on screen. A later version could learn
these boundaries from labelled footage instead, which is where the accuracy
ceiling of the rule-based approach eventually sits.
