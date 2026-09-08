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
| `python tools/evaluate.py` | Measure accuracy against the labelled clips |
| `python tools/record_clip.py` | Record labelled clips for evaluation |

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

That installs OpenCV, MediaPipe, FastAPI, Uvicorn and Ultralytics (which
brings PyTorch with it, so expect a large download the first time).

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

## What it recognises

Twenty-one activities, in three tiers. Tier 1 is the core list and is the one
that has to be reliable; the later tiers are built on the same machinery but
lean harder on seeing the right object.

**Tier 1 - core**

| Activity | What triggers it |
|---|---|
| **Exercise** | Hands sweeping a wide arc, briskly and repeatedly |
| **Drinking** | Hand to the mouth **with a drink container in it** |
| **Eating** | Hand travelling to the mouth and back, or a utensil in hand |
| **Laptop Work** | A laptop in view, hands working in front of it, body still |
| **Reading Procedure** | Head angled down over still hands holding something |
| **Experiment Operation** | Hands working a fixed forward zone, finely, no object naming it better |
| **Maintenance/Repair** | A tool in hand, or arms raised and reaching, in bursts |
| **Seated at Workstation** | Settled at a station with nothing more specific happening |
| **Idle/Floating** | Present and calm |
| **In-Transit/Movement** | The whole body crossing the module |
| **Anomaly (No Motion)** | No movement at all for more than 8 seconds |
| **Uncertain** | Nothing matched well enough to name honestly |

**Tier 2** - Holding Object, Tablet/Device Use, Sample Handling,
Stowage/Retrieval, Communication, Photography/Observation, Writing/Logging.

**Tier 3 (stretch)** - Health Check/Medical, Hygiene, Equipment
Donning/Doffing. These work in principle and are scored the same way, but they
have had the least testing.

An anomaly turns the console amber and raises a banner across the top. Amber is
used for nothing else, so it can only ever mean one thing.

### Sitting, in a system built for microgravity

A fair question from a judge: *there is no sitting in space, so why detect it?*

Crew do not sit, but they **restrain themselves at a workstation** - feet in
loops, thighs against a brace - and the geometry is close to a seated one:
torso upright and stable, hips fixed, legs folded and still. Recognising that
posture is recognising "settled at a station to work", which is exactly the
distinction an activity log needs. It is also the only way the system can be
tested on Earth, where every development hour happens with someone in a chair.

Sitting is treated as a **posture, not a rival activity**: somebody can be
seated *and* operating an experiment, and the console reports both.

It works facing away from the camera (no face landmarks are used anywhere) and
with the legs hidden behind a desk, where it falls back to torso cues and says
so, with lower confidence.

## Objects, and an honest limitation

Pose alone cannot separate some activities that matter. A hand raised to the
mouth is the same skeleton whether the crew member is drinking or eating. What
settles it is what they are holding, so a **YOLOv8n** model runs alongside the
pose tracker.

**The caveat, stated plainly.** That model is trained on COCO, which knows 80
everyday object classes and none of the things actually aboard a station - no
sample vials, no torque tools, no cargo transfer bags. Training a detector that
does know them needs a labelled dataset of flight hardware, which v1 does not
have. So v1 does two things deliberately:

- **Only whitelisted classes are used.** Everything else the model reports -
  chairs, plants, people, whatever is behind the crew member - is discarded
  before it reaches the console or the log.
- **Whitelisted classes are shown under the name of the item they stand in
  for.** A COCO "bottle" is displayed as a *Water Pouch*, because in this
  system that is the role it plays.

| COCO class | Shown as | Supports |
|---|---|---|
| bottle | Water Pouch | Drinking, Holding |
| cup | Drink Container | Drinking |
| bowl | Food Container | Eating |
| spoon / fork | Utensil | Eating |
| knife | Utensil / Tool | Eating, Maintenance |
| laptop | Crew Laptop | Laptop Work |
| keyboard / mouse | Workstation Input | Laptop Work |
| cell phone | Handheld Device | Tablet Use, Photography |
| book | Procedure Manual | Reading Procedure |
| scissors | Hand Tool | Maintenance |
| toothbrush | Hygiene Item | Hygiene, Sample Handling |
| remote | Handheld Controller | Experiment Operation |
| backpack | Stowage Bag | Stowage/Retrieval |
| clock | Timer | Experiment Operation |

**Custom training on real flight hardware is the obvious next version.** Being
upfront about this is better than a judge discovering it.

Objects **assist** classification and never decide it alone: every activity
still scores from pose, so the system works with nothing detected - which is
most of the time. Detection runs on a background thread every 4th frame, so
the video never waits for it.

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

## The console controls

Three buttons on the camera panel, sized to be hit during a live demo:

- **Skeleton** - overlay on or off.
- **Objects** - detection boxes on or off.
- **Privacy** - hides the video and keeps the skeleton and the analysis. Crew
  privacy is a genuine documented concern in spaceflight, and the monitoring
  works just as well without anyone being recognisable.

None of them touch classification. The same frames are analysed and the same
rows logged whatever is toggled; only the drawing changes.

## Downloading the log

Three buttons on the Activity Log panel, with a scope selector for
**this session** or **all history**:

- **CSV** - opens straight into Excel, one row per activity segment.
- **JSON** - the same data for anything reading it programmatically.
- **Report** - a plain-text mission-day summary: time per activity, how many
  times the crew member changed task, and every anomaly flagged.

Filenames carry a timestamp, so repeated downloads never overwrite each other.
Everything is generated locally; nothing leaves the machine.

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
| `SUBJECT_SWITCH_RATIO` | How much closer someone must be to take the lock |
| `SEATED_THRESHOLD` | How readily a posture reads as seated |
| `YOLO_EVERY_N_FRAMES` | How often object detection runs |
| `UNCERTAIN_BELOW` | Confidence below which it says Uncertain |

After changing anything, run `python tests/run_all.py` — the suite covers the
cases these values control, including the ones that are easy to break.

---

## One crew member, deliberately

v1 monitors **exactly one person**, and says so on screen with a
`SUBJECT LOCKED` indicator.

The detector is allowed to *see* several people so the system can choose
between them, then it picks one and discards the rest before anything
downstream - window, classifier, log - ever sees them. The subject is whoever
is **closest to the camera**, and the lock is sticky:

- Somebody walking past takes the lock only by appearing 1.45x closer for 1.5
  continuous seconds.
- The lock survives 2 seconds of the subject being hidden, so a hand across
  the lens does not wipe their history.
- If the lock does legitimately move, the window is cleared, so a new person
  is never classified using the previous one's movement.

Multi-crew tracking is still in the repository (`app/tracking.py`, tested at
six people) behind `ENABLE_MULTI_CREW = False`. It was switched off because it
cost real accuracy: a window that silently swaps between people classifies a
blend of both.

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
