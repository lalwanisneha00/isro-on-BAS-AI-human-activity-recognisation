"""Microgravity-invariant pose normalisation and the rolling analysis window.

Aboard a station a crew member can float at any distance, any tilt, even upside
down, so raw pixel coordinates are meaningless for recognising an activity.
Every frame is therefore rewritten into a body-local frame of reference:

  1. Translate  - the hip midpoint becomes the origin (0, 0).
  2. Scale      - distances are divided by torso length, so 1.0 unit == one
                  torso. Moving closer to or further from the camera no longer
                  changes any number.
  3. Rotate     - the spine is turned to point straight "up", so body tilt or
                  a sideways float no longer changes any number.

After this the shoulder midpoint always lands at almost exactly (0, -1),
whatever the crew member is doing. That invariant is what the dashboard shows
as proof the stage is working.
"""

import math
import time
from collections import deque

import numpy as np

from . import config


def _midpoint(points: np.ndarray, a: int, b: int) -> np.ndarray:
    return (points[a] + points[b]) * 0.5


class NormalisedPose:
    """One frame of pose, expressed in the body-local frame of reference."""

    __slots__ = ("timestamp", "points", "hip_center", "torso_length",
                 "spine_angle", "confidence", "visibility",
                 "hips_estimated", "framing", "shoulder_center")

    def __init__(self, timestamp, points, hip_center, torso_length,
                 spine_angle, confidence, visibility=None,
                 hips_estimated=False, framing="full",
                 shoulder_center=(0.0, 0.0)):
        self.timestamp = timestamp
        self.points = points              # (33, 2) torso-relative coordinates
        self.hip_center = hip_center      # raw image position, for reference
        self.torso_length = torso_length  # raw image scale, for reference
        self.spine_angle = spine_angle    # degrees from vertical
        self.confidence = confidence
        # Per-landmark visibility, kept because "the model cannot see the
        # legs" is a different statement from "the legs are near the hips",
        # and posture detection has to tell those two apart.
        self.visibility = (visibility if visibility is not None
                           else np.ones(len(points), dtype=np.float32))
        # True when the hips were inferred from the shoulders rather than
        # seen. Everything downstream is measured in torso lengths, so it
        # matters whether that scale was measured or reconstructed.
        self.hips_estimated = hips_estimated
        # "full"  - legs in shot, posture can be measured outright
        # "upper" - torso and both arms, the ordinary webcam view
        # "torso" - head, shoulders and part of the arms only
        self.framing = framing
        # Raw shoulder midpoint in the frame. This is the anchor for judging
        # whether somebody has left their station: a reconstructed hip sits
        # more than a shoulder width below them, so it swings whenever they
        # lean and exaggerates movement they did not make.
        self.shoulder_center = shoulder_center


def normalise(landmarks, frame_width: int, frame_height: int):
    """Convert MediaPipe landmarks into a NormalisedPose, or None if unusable."""
    if not landmarks or len(landmarks) < 33:
        return None

    # MediaPipe normalises x by width and y by height independently, which
    # squashes the picture on a non-square frame. Rescaling x by the aspect
    # ratio restores true proportions, so a distance means the same in any
    # direction.
    aspect = frame_width / float(frame_height)
    raw = np.array([[lm.x * aspect, lm.y] for lm in landmarks], dtype=np.float32)
    visibility = np.array([getattr(lm, "visibility", 1.0) for lm in landmarks],
                          dtype=np.float32)

    # The four torso anchors must be trustworthy or the whole frame is junk.
    anchors = (config.LM_LEFT_SHOULDER, config.LM_RIGHT_SHOULDER,
               config.LM_LEFT_HIP, config.LM_RIGHT_HIP)
    anchor_confidence = float(np.mean([visibility[i] for i in anchors]))
    if anchor_confidence < config.LANDMARK_VISIBILITY_THRESHOLD:
        return None

    # A detection has to look like a body before it is trusted. The checks
    # below deliberately look at the UPPER body only. Judging a pose on all
    # 33 landmarks fails the ordinary webcam view, where the legs are out of
    # shot: the model still emits them, guessed, with near-zero visibility,
    # and they drag the whole-body averages down until a perfectly good
    # upper-body pose is thrown away.
    upper = list(range(0, 25))
    if float(np.mean(visibility[upper])) < config.MIN_MEAN_VISIBILITY:
        return None

    outside = np.mean((raw[upper, 0] < -0.15) | (raw[upper, 0] > aspect + 0.15)
                      | (raw[upper, 1] < -0.15) | (raw[upper, 1] > 1.15))
    if float(outside) > config.MAX_OUT_OF_FRAME:
        return None

    # Shoulders are the anchor, not the hips. They are the most reliably
    # located landmarks on a seated or half-framed person, and requiring the
    # hips is what made detection flicker: their visibility hovers around the
    # threshold and tips under it on any small change.
    shoulder_visibility = float(np.mean([visibility[config.LM_LEFT_SHOULDER],
                                         visibility[config.LM_RIGHT_SHOULDER]]))
    if shoulder_visibility < config.SHOULDER_VISIBILITY_MIN:
        return None

    shoulder_center = _midpoint(raw, config.LM_LEFT_SHOULDER,
                                config.LM_RIGHT_SHOULDER)
    shoulder_width = float(np.linalg.norm(raw[config.LM_LEFT_SHOULDER]
                                          - raw[config.LM_RIGHT_SHOULDER]))
    if shoulder_width < config.MIN_SHOULDER_WIDTH:
        return None                      # too small or too far to work with

    hip_visibility = float(np.mean([visibility[config.LM_LEFT_HIP],
                                    visibility[config.LM_RIGHT_HIP]]))
    leg_visibility = float(np.mean([visibility[i] for i in (25, 26, 27, 28)]))

    hips_estimated = hip_visibility < config.HIP_VISIBILITY_MIN

    if hips_estimated:
        # The hips were not seen. MediaPipe still reports a position for them,
        # but it is a guess that jumps between frames, and since torso length
        # is measured from it the whole unit of scale jumped with it.
        #
        # A reconstructed hip is steadier than a guessed one: human torsos are
        # a fairly fixed multiple of shoulder width, so the shoulders - which
        # ARE clearly visible - can place the hips. The direction is the
        # perpendicular to the shoulder line, pointed away from the head.
        along = raw[config.LM_LEFT_SHOULDER] - raw[config.LM_RIGHT_SHOULDER]
        length = float(np.linalg.norm(along))
        if length < 1e-6:
            return None
        down = np.array([-along[1], along[0]], dtype=np.float32) / length
        if float(np.dot(down, shoulder_center - raw[0])) < 0:
            down = -down                 # point away from the nose, not at it

        torso_length = shoulder_width * config.TORSO_PER_SHOULDER
        hip_center = shoulder_center + down * torso_length

        # How much of the crew member is actually in shot decides how much can
        # honestly be said about them. Both arms in view is a good deal more
        # to work with than a head and shoulders.
        wrist_visibility = float(np.mean([visibility[15], visibility[16]]))
        framing = ("upper" if wrist_visibility >= config.WRIST_VISIBILITY_MIN
                   else "torso")
    else:
        hip_center = _midpoint(raw, config.LM_LEFT_HIP, config.LM_RIGHT_HIP)
        spine = shoulder_center - hip_center
        torso_length = float(np.linalg.norm(spine))
        if torso_length < config.MIN_TORSO_LENGTH:
            return None
        framing = ("full" if leg_visibility >= config.LEG_VISIBILITY_MIN
                   else "upper")

        # Human proportions: shoulders are neither a point nor several torsos
        # wide. Only meaningful when the hips were actually measured.
        ratio = shoulder_width / torso_length
        if not (config.MIN_SHOULDER_RATIO <= ratio <= config.MAX_SHOULDER_RATIO):
            return None

    spine = shoulder_center - hip_center
    torso_length = float(np.linalg.norm(spine))
    if torso_length < config.MIN_TORSO_LENGTH:
        return None

    anchor_confidence = max(shoulder_visibility,
                            (shoulder_visibility + hip_visibility) / 2.0)

    # 1. translate to the hip origin, 2. divide out the torso scale
    centred = (raw - hip_center) / torso_length

    # 3. rotate so the spine points to screen-up, which is -Y in image space.
    unit = spine / torso_length
    ux, uy = float(unit[0]), float(unit[1])
    rotation = np.array([[-uy, ux],
                         [-ux, -uy]], dtype=np.float32)
    oriented = centred @ rotation.T

    spine_angle = math.degrees(math.atan2(ux, -uy))

    return NormalisedPose(
        timestamp=time.time(),
        points=oriented.astype(np.float32),
        hip_center=(float(hip_center[0] / aspect), float(hip_center[1])),
        torso_length=torso_length,
        spine_angle=spine_angle,
        confidence=anchor_confidence,
        visibility=visibility,
        hips_estimated=hips_estimated,
        framing=framing,
        shoulder_center=(float(shoulder_center[0] / aspect),
                         float(shoulder_center[1])),
    )


class PoseWindow:
    """Rolling buffer holding the last WINDOW_SECONDS of normalised poses.

    The window is bounded by *time*, not by a frame count. A fixed frame count
    silently changes length whenever the frame rate moves - at 15 fps a
    48-frame buffer covers 3.2 seconds, not 2 - which shifts every feature and
    threshold tuned against it. Evicting by timestamp keeps the analysis window
    the same two seconds whatever the machine manages.
    """

    def __init__(self, seconds: float = config.WINDOW_SECONDS):
        self.seconds = seconds
        self._frames = deque()
        self._misses = 0
        self._miss_deadline = None

    def push(self, pose) -> None:
        """Add a frame.

        A `None` means nobody was detected this frame. A brief run of those is
        treated as tracking noise and ignored; only a sustained absence clears
        the buffer, so the activity label survives a momentary dropout.
        """
        if pose is None:
            # Gap tolerance is measured in seconds for the same reason the
            # window is: a frame count would mean different things at
            # different frame rates.
            now = time.time()
            if self._miss_deadline is None:
                self._miss_deadline = now + config.POSE_GAP_TOLERANCE_SECONDS
            elif now >= self._miss_deadline:
                self._frames.clear()
            return

        self._miss_deadline = None
        self._frames.append(pose)

        cutoff = pose.timestamp - self.seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def clear(self) -> None:
        self._frames.clear()
        self._miss_deadline = None

    @property
    def frames(self) -> list:
        return list(self._frames)

    @property
    def count(self) -> int:
        return len(self._frames)

    @property
    def size(self) -> int:
        """Nominal frame capacity, for display only."""
        return max(1, int(round(self.seconds * config.TARGET_FPS)))

    @property
    def is_full(self) -> bool:
        """True once the window spans nearly its full duration."""
        return (len(self._frames) >= config.MIN_WINDOW_FRAMES
                and self.span_seconds >= self.seconds * 0.85)

    @property
    def is_ready(self) -> bool:
        """Enough history to classify, even if the window is not yet full."""
        return (len(self._frames) >= config.MIN_WINDOW_FRAMES
                and self.span_seconds >= config.MIN_WINDOW_SECONDS)

    @property
    def span_seconds(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1].timestamp - self._frames[0].timestamp

    @property
    def latest(self):
        return self._frames[-1] if self._frames else None

    def stack(self) -> np.ndarray:
        """All buffered poses as one (frames, 33, 2) array for the classifier."""
        if not self._frames:
            return np.empty((0, 33, 2), dtype=np.float32)
        return np.stack([f.points for f in self._frames])
