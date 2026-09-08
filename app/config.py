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

ACT_SEATED = "Seated at Workstation"

# --- Tier 1: the core list, and the one that has to be reliable ------------
ACT_DRINKING = "Drinking"
ACT_LAPTOP = "Laptop Work"
ACT_READING = "Reading Procedure"
ACT_UNCERTAIN = "Uncertain"

# --- Tier 2 ---------------------------------------------------------------
ACT_HOLDING = "Holding Object"
ACT_TABLET = "Tablet/Device Use"
ACT_SAMPLE = "Sample Handling"
ACT_STOWAGE = "Stowage/Retrieval"
ACT_COMMS = "Communication"
ACT_PHOTO = "Photography/Observation"
ACT_WRITING = "Writing/Logging"

# --- Tier 3: stretch ------------------------------------------------------
ACT_MEDICAL = "Health Check/Medical"
ACT_HYGIENE = "Hygiene"
ACT_DONNING = "Equipment Donning/Doffing"

# Display order. Related work sits together, so the timeline reads in bands
# rather than jumping between unrelated kinds of task.
ACTIVITY_ORDER = [
    # operations
    ACT_EXPERIMENT, ACT_LAPTOP, ACT_SAMPLE, ACT_WRITING, ACT_TABLET,
    # maintenance
    ACT_MAINTENANCE, ACT_STOWAGE, ACT_DONNING,
    # sustenance
    ACT_EATING, ACT_DRINKING,
    # information
    ACT_READING, ACT_COMMS, ACT_PHOTO,
    # rest and self-care
    ACT_SEATED, ACT_IDLE, ACT_MEDICAL, ACT_HYGIENE,
    # movement
    ACT_EXERCISE, ACT_TRANSIT, ACT_HOLDING,
    # alert
    ACT_ANOMALY,
]

# Which tier each activity belongs to, so the console can be honest about
# what is core and what is a stretch.
ACTIVITY_TIER = {
    ACT_EXERCISE: 1, ACT_DRINKING: 1, ACT_EATING: 1, ACT_LAPTOP: 1,
    ACT_READING: 1, ACT_EXPERIMENT: 1, ACT_MAINTENANCE: 1, ACT_SEATED: 1,
    ACT_IDLE: 1, ACT_TRANSIT: 1, ACT_ANOMALY: 1, ACT_UNCERTAIN: 1,
    ACT_HOLDING: 2, ACT_TABLET: 2, ACT_SAMPLE: 2, ACT_STOWAGE: 2,
    ACT_COMMS: 2, ACT_PHOTO: 2, ACT_WRITING: 2,
    ACT_MEDICAL: 3, ACT_HYGIENE: 3, ACT_DONNING: 3,
}

# Activity colours, assigned by FAMILY rather than one hue per activity.
#
# The task list runs to twenty-odd activities, and twenty-odd colours nobody
# can tell apart is worse than none - the timeline becomes confetti. Related
# activities therefore share a colour, and the label text carries the identity
# (every timeline lane, chart bar and log row is named). Colour says what kind
# of work it is; the words say which.
#
# The families use the first six slots of a categorical palette validated for
# this dark surface: adjacent-pair CVD delta-E 8.4, normal-vision 19.3, all at
# least 3:1 contrast. Anomaly sits outside as a reserved status amber, so an
# alert can never be mistaken for one more kind of work.
FAMILY_HEX = {
    "operations":  "#3987e5",   # blue    - working an experiment or a console
    "maintenance": "#d95926",   # orange  - tools, repair, stowage
    "sustenance":  "#199e70",   # aqua    - eating and drinking
    "information": "#c98500",   # yellow  - reading, talking, observing
    "rest":        "#d55181",   # magenta - idle, seated, self-care
    "movement":    "#008300",   # green   - exercise and transit
    "alert":       "#fab219",   # status  - reserved for anomalies only
    "neutral":     "#6E7686",   # grey    - unknown, uncertain, nobody there
}

ACTIVITY_FAMILY = {
    # operations - working an experiment, a console or a sample
    ACT_EXPERIMENT: "operations", ACT_LAPTOP: "operations",
    ACT_SAMPLE: "operations", ACT_WRITING: "operations",
    ACT_TABLET: "operations",
    # maintenance - tools, repair, moving hardware about
    ACT_MAINTENANCE: "maintenance", ACT_STOWAGE: "maintenance",
    ACT_DONNING: "maintenance",
    # sustenance
    ACT_EATING: "sustenance", ACT_DRINKING: "sustenance",
    # information - reading, talking, observing
    ACT_READING: "information", ACT_COMMS: "information",
    ACT_PHOTO: "information",
    # rest and self-care
    ACT_SEATED: "rest", ACT_IDLE: "rest",
    ACT_MEDICAL: "rest", ACT_HYGIENE: "rest",
    # movement
    ACT_EXERCISE: "movement", ACT_TRANSIT: "movement",
    ACT_HOLDING: "movement",
    # alert, and the honest non-answers
    ACT_ANOMALY: "alert",
    ACT_UNCERTAIN: "neutral", ACT_ACQUIRING: "neutral", ACT_NO_CREW: "neutral",
}


def family_of(activity: str) -> str:
    return ACTIVITY_FAMILY.get(activity, "neutral")


def colour_of(activity: str) -> str:
    return FAMILY_HEX[family_of(activity)]


ACTIVITY_HEX = {name: colour_of(name) for name in ACTIVITY_FAMILY}


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
LOGGED_STATES = set(ACTIVITY_ORDER) | {ACT_NO_CREW, ACT_UNCERTAIN}

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
# ---------------------------------------------------- subject selection ----
# v1 monitors ONE crew member. The detector is allowed to see a few candidates
# so the system can choose deliberately - the person closest to the camera -
# rather than accepting whichever pose the detector happened to return first.
# Measured cost of raising the budget from 1 to 3: 14.5ms -> 13.3ms per frame,
# i.e. none. MediaPipe runs one detection pass either way.
POSE_DETECTION_BUDGET = 3

# How far, in torso lengths, a candidate may be from where the subject was and
# still be recognised as the same person.
SUBJECT_MATCH_RADIUS = 2.0
# The lock survives this long with nobody visible, so stepping behind a rack
# does not wipe the subject's activity history.
SUBJECT_GRACE_SECONDS = 2.0
# A different person takes over only by appearing this much closer...
SUBJECT_SWITCH_RATIO = 1.45
# ...and staying that way for this long. One frame of somebody looming past
# the lens must not move the lock.
SUBJECT_SWITCH_SECONDS = 1.5

# Multi-crew tracking is kept in the repository (app/tracking.py, and its
# tests) but is not on the active path. It cost real accuracy: a detector
# asked for several poses returns weak ones too, and a window that swaps
# between people classifies a blend of both.
ENABLE_MULTI_CREW = False
MAX_CREW = 1

# Detections are matched to existing crew tracks by how far the hip centre
# moved, measured in torso lengths so the gate holds at any camera distance.
TRACK_MATCH_RADIUS = 1.8
# A crew member who leaves frame or is occluded keeps their identity, and their
# activity history, for this long before the track is retired.
TRACK_EXPIRY_SECONDS = 2.0
CREW_NAME_PREFIX = "CM"
# The single monitored subject, named in the log and on screen.
SUBJECT_NAME = "CM-1"
# Skeleton overlay on the video. Toggled live from the console.
SHOW_SKELETON = True

# ------------------------------------------------------- pose sanity gate --
# A weak detection - a coat on a chair, a reflection, half a body at the frame
# edge - still arrives as 33 landmarks and would be tracked as a crew member,
# drawing a nonsense skeleton and logging nonsense activity. A pose has to look
# like a body before it is accepted.
MIN_MEAN_VISIBILITY = 0.35        # averaged over the whole skeleton
MIN_SHOULDER_RATIO = 0.20         # shoulder width, relative to torso length
MAX_SHOULDER_RATIO = 2.20
MAX_OUT_OF_FRAME = 0.45           # fraction of landmarks allowed outside view

# ---------------------------------------------------------------- posture --
# Seated detection, all in torso lengths or degrees so it holds at any camera
# distance. See app/posture.py for why sitting matters aboard a station.
SEATED_KNEE_ANGLE = 110.0         # a folded leg; below this reads as seated
STANDING_KNEE_ANGLE = 165.0       # a straight leg
SEATED_HIP_HEIGHT = 0.75          # hips this far above the ankles: seated
STANDING_HIP_HEIGHT = 1.30        # hips a full torso up: standing
SEATED_HIP_TRAVEL = 0.10          # hips barely move when settled
UPRIGHT_TOLERANCE = 18.0          # degrees of spine tilt still counted upright
SEATED_VISIBLE_EXTENT = 2.10      # body height in torso lengths when legs hidden
SEATED_THRESHOLD = 0.55           # score above which the subject reads as seated

# Mean visibility across knees and ankles below which the lower body is
# treated as out of shot - behind a desk, or cropped by the frame.
LEG_VISIBILITY_MIN = 0.55

# ------------------------------------------------------ object detection --
# A COCO-pretrained YOLO, hard-restricted to the objects our task list needs.
# See app/objects.py for what that restriction means and why it is a stated
# v1 simplification rather than a hidden one.
ENABLE_OBJECT_DETECTION = True
YOLO_MODEL_NAME = "YOLOv8n"
YOLO_WEIGHTS_PATH = "models/yolov8n.pt"
YOLO_IMAGE_SIZE = 480             # smaller than the 640 default; CPU budget
YOLO_CONFIDENCE = 0.35
# Objects move far more slowly than limbs, so detection runs on a fraction of
# the frames and the newest result is reused in between. Pose tracking never
# waits for it.
YOLO_EVERY_N_FRAMES = 4
# How near a wrist an object must be, in torso lengths, to count as held.
OBJECT_HAND_RADIUS = 0.85

# ------------------------------------------------- body-relative zones -----
# All in torso lengths, measured in the body-local frame: hips at the origin,
# one torso up to the shoulders, screen-up negative.
HEAD_DOWN_RISE = 0.36             # nose this close to the shoulders: head down
ROTATED_SHOULDER_RATIO = 0.30     # shoulders this narrow: torso turned away
CHEST_ZONE_TOP = -0.95            # the band a console or sample is worked in
CHEST_ZONE_BOTTOM = -0.20
PERIPHERAL_REACH = 0.80           # a hand thrown out to the side
EYE_LEVEL_TOP = -1.15             # held up to look at, but not overhead
EYE_LEVEL_BOTTOM = -1.60
TORSO_CONTACT_RADIUS = 0.20       # a hand actually against the torso
TORSO_CONTACT_HALF_WIDTH = 0.24   # and near the body centre line
CROSS_BODY_MARGIN = 0.12          # a hand past the body's centre line

# How much a supporting object lifts an activity's score. Objects sharpen a
# decision the pose has already narrowed; they never make it on their own.
OBJECT_BOOST = 1.0
# An object alone can carry an activity this far when the posture is neutral.
OBJECT_ONLY_FLOOR = 0.55
# Below this the system says Uncertain instead of naming an activity.
UNCERTAIN_BELOW = 0.42

# A winning score at or below this means nothing really matched.
UNCERTAIN_SCORE_FLOOR = 0.10

# ---------------------------------------------------------- video display --
# "normal" shows the camera; "privacy" replaces it with a plain field so the
# skeleton and the analysis stay visible while the crew member does not.
VIDEO_MODE = "normal"
SHOW_OBJECTS = True

# How far a hand must travel, in torso lengths, for a hand-near-the-face
# reading to count as bringing something TO the mouth rather than the head
# simply being bowed over still hands.
HAND_TO_MOUTH_TRAVEL = 0.28

# Cores the object model may use. The video pipeline needs the rest.
YOLO_THREADS = 2
