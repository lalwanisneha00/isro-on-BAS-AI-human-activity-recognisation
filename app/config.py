"""Central configuration for the BAS Crew Activity Recognition system."""

# ---------------------------------------------------------------- camera ----
CAMERA_INDEX = 0          # 0 = default built-in webcam
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
TARGET_FPS = 24
JPEG_QUALITY = 80

# ------------------------------------------------------------------ server --
HOST = "127.0.0.1"
PORT = 8000

# ------------------------------------------------------- mission identity ---
MISSION_NAME = "BHARATIYA ANTARIKSH STATION"
MODULE_NAME = "CREW ACTIVITY RECOGNITION"

# -------------------------------------------------------------- pose ------
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
LANDMARK_VISIBILITY_THRESHOLD = 0.4   # below this a joint is drawn faded

# ------------------------------------------------------ overlay palette ----
# OpenCV works in BGR, so these are reversed from the CSS hex values.
COLOR_ACCENT = (255, 217, 0)      # #00D9FF electric cyan
COLOR_JOINT = (255, 255, 255)
COLOR_ALERT = (32, 176, 255)      # #FFB020 amber

# ----------------------------------------------------- normalisation ------
WINDOW_SECONDS = 2.0                  # rolling analysis window, in seconds
WINDOW_SIZE = int(WINDOW_SECONDS * TARGET_FPS)   # nominal frames, display only
# Classification needs both a minimum span of time and a minimum number of
# samples within it, so a stuttering frame rate cannot produce a "full" window
# built from three frames.
MIN_WINDOW_SECONDS = 1.2
MIN_WINDOW_FRAMES = 10
MIN_TORSO_LENGTH = 0.02               # below this the pose is too small to trust

# Landmark indices used by the normaliser (MediaPipe Pose topology).
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24

# ------------------------------------------------------- activity classes --
ACT_EXERCISE = "Exercise"
ACT_EXPERIMENT = "Experiment Operation"
ACT_EATING = "Eating/Rest"
ACT_MAINTENANCE = "Maintenance"
ACT_IDLE = "Idle"
ACT_TRANSIT = "In-Transit/Movement"
ACT_ANOMALY = "Anomaly (No Motion)"
ACT_ACQUIRING = "Acquiring"
ACT_NO_CREW = "No Crew Detected"

ACTIVITY_ORDER = [ACT_EXERCISE, ACT_EXPERIMENT, ACT_EATING, ACT_MAINTENANCE,
                  ACT_IDLE, ACT_TRANSIT, ACT_ANOMALY]

# Activity colours. The six real activities take the first six slots of a
# categorical palette validated for this dark surface (adjacent-pair CVD
# delta-E 8.4, normal-vision 19.3, all >= 3:1 contrast). The order is fixed and
# never sorted by value, because ordering is what keeps neighbouring colours
# distinguishable. Anomaly is deliberately NOT a categorical slot - it is a
# reserved status colour (amber, 9.49:1 on this surface) so an alert can never
# be mistaken for one more activity.
ACTIVITY_HEX = {
    ACT_EXERCISE:     "#3987e5",   # blue
    ACT_EXPERIMENT:   "#d95926",   # orange
    ACT_EATING:       "#199e70",   # aqua
    ACT_MAINTENANCE:  "#c98500",   # yellow
    ACT_IDLE:         "#d55181",   # magenta
    ACT_TRANSIT:      "#008300",   # green
    ACT_ANOMALY:      "#fab219",   # status: warning
    ACT_ACQUIRING:    "#5A6478",
    ACT_NO_CREW:      "#6E7686",
}


def _bgr(hex_colour: str) -> tuple:
    """OpenCV draws in BGR, the web page in RGB - one source of truth."""
    h = hex_colour.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


ACTIVITY_COLORS = {name: _bgr(h) for name, h in ACTIVITY_HEX.items()}

# ---------------------------------------------- classifier thresholds ------
# Speeds are in torso-lengths per second, so they hold at any camera distance.
# Recalibrated once trajectory smoothing removed the noise these were hiding.
# Measured, at realistic camera noise: a frozen person reads 0.044, someone
# idle but alive 0.075, someone typing 0.148, and real work 0.20 upward. The
# old values were set against noise-inflated numbers three times as large,
# which is why small genuine movements were being read as stillness.
WRIST_SPEED_STILL = 0.09
WRIST_SPEED_FOCUSED = 0.18
WRIST_SPEED_FAST = 1.10
WRIST_SPEED_VIGOROUS = 2.20

# Body drift is measured in raw frame widths per second.
BODY_SPEED_STILL = 0.012
BODY_SPEED_MOVING = 0.055
BODY_SPEED_FAST = 0.190

# Autocorrelation strength that counts as genuinely repetitive movement.
# Validated against synthetic signals: real waves score 0.75-0.98, noise 0.51,
# and non-repeating movement (a single reach, drift, jitter) scores 0.00.
PERIODICITY_RHYTHMIC = 0.62
PERIODICITY_STRONG = 0.88

# How far the hands roam over the window, in torso lengths. This is the
# measurement that separates exercise from working at a rack: both can be
# brisk, but only exercise sweeps a wide arc.
# The two activities meet at a single boundary: hands that stay within about
# a third of a torso length are doing fine work, hands that sweep further are
# exercising. Speed cannot draw this line - vigorous fine work is fast within
# a small area, and a slow stretch is wide but unhurried.
EXERCISE_MIN_RANGE = 0.30
EXERCISE_FULL_RANGE = 0.60
FOCUSED_MAX_RANGE = 0.30          # above this, hands are not doing fine work
FOCUSED_RANGE_TAPER = 0.08        # narrow, so the boundary stays decisive
FOCUSED_MIN_RANGE = 0.025         # below this, the hands are simply still
HEAD_ZONE_RADIUS = 0.52           # wrist-to-mouth distance counted as "at face"
WORKSTATION_X = 0.85              # hands kept in front of the torso
WORKSTATION_Y_TOP = -1.05
WORKSTATION_Y_BOTTOM = 0.25
ELEVATED_MARGIN = 0.10            # wrist above shoulder by this much
EXTENDED_REACH = 1.15             # wrist this far from shoulder counts as extended

# Stillness for an alert is judged on how far the hands travelled, not on
# their speed. Speed barely separates a frozen person (0.044) from one who is
# simply idle (0.075); travel separates them cleanly (0.032 against 0.112).
ANOMALY_STILL_RANGE = 0.07
ANOMALY_STILL_SECONDS = 8.0       # sustained stillness before an alert
LABEL_HOLD_FRAMES = 8             # hysteresis, stops the label flickering


# Eating is intermittent - a hand goes to the mouth, comes down, goes back up.
# A 2 second window cannot span a whole cycle, so recent face contact is
# remembered for this long and keeps the meal label stable between bites.
EATING_MEMORY_SECONDS = 6.0
FACE_CONTACT_TRIGGER = 0.35

# MediaPipe occasionally drops a frame or two when a limb crosses the torso.
# Discarding the whole window for a momentary miss would reset the label, so
# short gaps are tolerated and only a real absence clears the buffer.
POSE_GAP_TOLERANCE_SECONDS = 0.5

# ----------------------------------------------------------- activity log --
LOG_DIR = "logs"
LOG_CSV_NAME = "activity_log.csv"
LOG_DB_NAME = "activity_log.db"

# A segment shorter than this is tracking noise, not a real activity, so it is
# folded away rather than written as its own row.
MIN_LOG_SECONDS = 1.5
HISTORY_LIMIT = 80

# States that belong in a mission log. "Acquiring" is pure system warm-up and
# is deliberately excluded; an empty module is worth recording.
LOGGED_STATES = set(ACTIVITY_ORDER) | {ACT_NO_CREW}

# How much history the timeline strip shows.
TIMELINE_WINDOW_SECONDS = 300.0

# A webcam that loses a fight over the device can end up opened but delivering
# roughly one frame a second, for ever, while still reporting itself healthy.
# The capture loop watches its own frame rate and reacquires the device if it
# stays below this floor.
CAMERA_MIN_HEALTHY_FPS = 6.0
CAMERA_STALL_SECONDS = 4.0

# ------------------------------------------------------------- demo mode --
DEMO_DIR = "demo"
# Set by run.py --demo, so a demo can start without touching the webcam.
START_IN_DEMO_MODE = False
DEMO_VIDEO_NAME = "demo_feed.mp4"
DEMO_POSE_NAME = "demo_poses.npz"

# ------------------------------------------------------------ crew tracking --
# Crew members tracked at once. Multi-crew works (the tracker and roster are
# tested to six), but asking MediaPipe for several poses costs frame rate and
# lets weak, spurious detections through, which hurt single-person accuracy.
# v1 ships single-crew for reliability; raise this to bring the roster back.
MAX_CREW = 1

# Detections are matched to existing crew tracks by how far the hip centre
# moved, measured in torso lengths so the gate holds at any camera distance.
TRACK_MATCH_RADIUS = 1.8
# A crew member who leaves frame or is occluded keeps their identity, and their
# activity history, for this long before the track is retired.
TRACK_EXPIRY_SECONDS = 2.0
CREW_NAME_PREFIX = "CM"

# ------------------------------------------------------- pose sanity gate --
# A weak detection - a coat on a chair, a reflection, half a body at the frame
# edge - still arrives as 33 landmarks and would be tracked as a crew member,
# drawing a nonsense skeleton and logging nonsense activity. A pose has to look
# like a body before it is accepted.
MIN_MEAN_VISIBILITY = 0.35        # averaged over the whole skeleton
MIN_SHOULDER_RATIO = 0.20         # shoulder width, relative to torso length
MAX_SHOULDER_RATIO = 2.20
MAX_OUT_OF_FRAME = 0.45           # fraction of landmarks allowed outside view
