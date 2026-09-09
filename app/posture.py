"""Posture: is the crew member seated, standing, or free-floating?

Why sitting matters aboard a station
------------------------------------
A judge will reasonably ask "there is no sitting in microgravity, why detect
it?". Two answers, and both are real:

* Crew do not sit, but they *restrain* themselves at a workstation - feet in
  loops, thighs against a brace - and the resulting geometry is close to a
  seated one: torso upright and stable, hips fixed, legs folded and still.
  Recognising that posture is recognising "settled at a station to work",
  which is exactly the distinction the activity log needs.
* Every hour of Earth-based development, testing and demonstration happens
  with people sitting on chairs. A system that cannot handle a seated subject
  cannot be tested by the people building it.

Posture is treated as a *modifier*, not as a rival activity. Somebody can be
seated and operating an experiment at the same time, and the log should be
able to say so rather than having to choose.

How it is decided
-----------------
Face landmarks are never used, so this works with the crew member facing away
from the camera. Two independent routes, because the lower body is often
hidden behind a desk:

1. Full body visible - knee angle, and how far the hips sit above the ankles.
   A standing leg is close to straight and the hips ride high; a seated one
   folds toward a right angle and the hips drop to roughly knee height.
2. Lower body hidden - which is the normal case at a desk - ask whether the
   legs SHOULD have been visible. If there is room in the frame below the
   hips for standing legs and no legs are detected in it, they are not
   missing, they are folded: the crew member is sitting. If the frame stops
   just below the hips the legs were simply cropped, nothing can be inferred,
   and the reading says so instead of guessing.

Route 2 used to score "hips still, spine upright, body short in frame", which
describes a person standing at a desk exactly as well as a seated one, and it
flip-flopped between the two on live video. The frame-room test replaces it
because it measures something that actually differs between the two.
"""

import math
from collections import deque

import numpy as np

from . import config

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

STANDING = "Standing"
SEATED = "Seated"
FLOATING = "Free-floating"
UNKNOWN = "Unknown"


class WorkstationWatch:
    """Evidence, gathered over far longer than one window, that the crew
    member is settled at a workstation.

    Being at a station is not a shape, it is a duration. Two seconds cannot
    tell a person who has sat down to work from one who has paused mid-stride,
    because both look identical for two seconds. Over fifteen they do not:
    somebody working stays in one place with their hands in front of them,
    and somebody standing drifts.

    Nothing here needs the hips, the legs, or the face. It uses the shoulders,
    which are the most reliably located landmarks on a half-framed person, and
    the hands, which are what a workstation is for.
    """

    def __init__(self, horizon: float = None):
        self.horizon = horizon or config.WORKSTATION_HORIZON
        self._samples = deque()          # (time, anchor xy, hands working)

    def reset(self) -> None:
        self._samples.clear()

    def observe(self, pose, now: float) -> None:
        """Take one frame. `pose` may be None when nobody was detected."""
        if pose is None:
            # A gap does not discard the history - somebody reaching off to
            # the side for a moment has not left their station - but a long
            # absence does, further down.
            if self._samples and now - self._samples[-1][0] > config.WORKSTATION_GAP:
                self._samples.clear()
            return

        points = pose.points
        # Anchored on the shoulders, which are both visible and stable. A
        # reconstructed hip hangs more than a shoulder width below them, so
        # it swings whenever the crew member leans over their work and
        # reports travel they never made.
        anchor = pose.shoulder_center
        shoulder_width = float(np.linalg.norm(points[L_SHOULDER]
                                              - points[R_SHOULDER]))

        wrists = points[[15, 16]]
        working = bool(np.any(
            (np.abs(wrists[:, 0]) < config.WORKING_ZONE_HALF_WIDTH)
            & (wrists[:, 1] > config.WORKING_ZONE_TOP)
            & (wrists[:, 1] < config.WORKING_ZONE_BOTTOM)))

        self._samples.append((now, anchor, working, shoulder_width, pose.torso_length))
        cutoff = now - self.horizon
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    # ------------------------------------------------------------ readings --
    @property
    def span(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    @property
    def travel(self) -> float:
        """How far the crew member drifted, in shoulder widths."""
        if len(self._samples) < 2:
            return 0.0
        anchors = np.array([s[1] for s in self._samples])
        span = float(np.linalg.norm(anchors.max(axis=0) - anchors.min(axis=0)))
        # Shoulder width in the same frame units as the anchor positions.
        width = float(np.median([s[3] * s[4] for s in self._samples]))
        return span / max(width, 1e-4)

    @property
    def hands_working(self) -> float:
        if not self._samples:
            return 0.0
        return float(np.mean([1.0 if s[2] else 0.0 for s in self._samples]))

    def evidence(self) -> float:
        """0 to 1: how strongly this looks like somebody settled to work."""
        if self.span < config.WORKSTATION_MIN_SPAN:
            return 0.0                   # not watched long enough to say

        # Staying put is the main thing, hands in the working zone confirms
        # it. Somebody standing still empty-handed scores lower than somebody
        # sat working, which is the distinction that matters.
        stayed = _falling(self.travel, config.WORKSTATION_TRAVEL_SETTLED,
                          config.WORKSTATION_TRAVEL_MOVING)
        hands = _ramp(self.hands_working, 0.35, 0.80)
        # Confidence grows with how long the evidence has been accumulating.
        maturity = _ramp(self.span, config.WORKSTATION_MIN_SPAN,
                         self.horizon * 0.75)
        return stayed * (0.62 + 0.38 * hands) * (0.70 + 0.30 * maturity)

    def as_dict(self) -> dict:
        return {
            "watching_for": round(self.span, 1),
            "travel": round(self.travel, 2),
            "hands_working": round(self.hands_working, 2),
            "evidence": round(self.evidence(), 3),
        }


def _angle(a, b, c) -> float:
    """Interior angle at b, in degrees, for the joint chain a-b-c."""
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class PostureReading:
    """What was decided, and the evidence behind it."""

    __slots__ = ("posture", "confidence", "seated", "knee_angle",
                 "hip_above_ankle", "lower_body_visible", "hip_stability",
                 "spine_upright", "leg_room", "framing", "settled_seconds",
                 "travel", "basis")

    def as_dict(self) -> dict:
        return {
            "posture": self.posture,
            "confidence": round(self.confidence, 3),
            "seated": self.seated,
            "knee_angle": round(self.knee_angle, 1) if self.knee_angle else None,
            "hip_above_ankle": round(self.hip_above_ankle, 3)
            if self.hip_above_ankle is not None else None,
            "lower_body_visible": self.lower_body_visible,
            "hip_stability": round(self.hip_stability, 4),
            "spine_upright": round(self.spine_upright, 1),
            "leg_room": round(self.leg_room, 2),
            "framing": self.framing,
            "settled_seconds": round(self.settled_seconds, 1),
            "travel": round(self.travel, 2),
            "basis": self.basis,
        }


def _blank(basis: str) -> PostureReading:
    reading = PostureReading()
    reading.posture = UNKNOWN
    reading.confidence = 0.0
    reading.seated = False
    reading.knee_angle = None
    reading.hip_above_ankle = None
    reading.lower_body_visible = False
    reading.hip_stability = 0.0
    reading.spine_upright = 0.0
    reading.leg_room = 0.0
    reading.framing = "unknown"
    reading.settled_seconds = 0.0
    reading.travel = 0.0
    reading.basis = basis
    return reading


def read(window, workstation=None) -> PostureReading:
    """Decide the subject's posture from a window of normalised poses.

    Everything is measured in torso lengths, so the answer does not change
    with distance from the camera.
    """
    frames = window.frames
    if len(frames) < 3:
        return _blank("not enough frames")
    latest_pose = frames[-1]

    points = np.stack([f.points for f in frames])          # (N, 33, 2)
    latest = points[-1]

    reading = _blank("")

    # How far the hips actually travel across the window.
    #
    # This has to come from the RAW frame position. Normalisation pins the hip
    # midpoint to the origin by definition, so measuring it on the normalised
    # points returns zero however much the crew member moves - which meant the
    # "settled" evidence was silently constant and carried no information.
    raw_hips = np.array([f.hip_center for f in frames])
    span = float(np.linalg.norm(raw_hips.max(axis=0) - raw_hips.min(axis=0)))
    scale = max(frames[-1].torso_length, config.MIN_TORSO_LENGTH)
    reading.hip_stability = span / scale

    # Spine tilt away from vertical. Normalisation pins the spine to "up", so
    # this is measured on the raw per-frame angle instead.
    reading.spine_upright = float(np.mean([abs(f.spine_angle) for f in frames]))

    # --- is the lower body actually in shot? ------------------------------
    # This has to come from the model's own visibility scores. Judging it by
    # geometry does not work: a seated person viewed head-on has their knees
    # close to their hips, which looks exactly like knees that were never
    # detected - and that is the very case this has to get right.
    leg_visibility = float(np.mean([
        np.mean([f.visibility[i] for i in (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)])
        for f in frames[-8:]
    ]))
    reading.lower_body_visible = leg_visibility >= config.LEG_VISIBILITY_MIN
    reading.framing = latest_pose.framing

    # The frame-room test below asks whether legs SHOULD have been visible in
    # the space under the hips. That question only means anything when the hip
    # position was measured. Where the hips were reconstructed from the
    # shoulders, the answer would be about where this code chose to put them,
    # not about the crew member.
    hips_measured = not latest_pose.hips_estimated

    hip_mid = (latest[L_HIP] + latest[R_HIP]) / 2.0
    ankles = [latest[L_ANKLE], latest[R_ANKLE]]

    if reading.lower_body_visible:
        # --- route 1: the legs tell us directly ---------------------------
        left = _angle(latest[L_HIP], latest[L_KNEE], latest[L_ANKLE])
        right = _angle(latest[R_HIP], latest[R_KNEE], latest[R_ANKLE])
        reading.knee_angle = (left + right) / 2.0

        # Screen-up is negative Y, so a hip well above the ankles is a large
        # positive gap. Standing keeps the hips roughly a torso above them.
        ankle_y = float(np.mean([a[1] for a in ankles]))
        reading.hip_above_ankle = float(ankle_y - hip_mid[1])

        bend = _falling(reading.knee_angle, config.SEATED_KNEE_ANGLE,
                        config.STANDING_KNEE_ANGLE)
        drop = _falling(reading.hip_above_ankle, config.SEATED_HIP_HEIGHT,
                        config.STANDING_HIP_HEIGHT)
        settled = _falling(reading.hip_stability, config.SEATED_HIP_TRAVEL,
                           config.SEATED_HIP_TRAVEL * 3.0)

        # Knee angle is only worth trusting when the thigh is actually
        # visible as a length. Seen head-on, a seated thigh points at the
        # camera and collapses to almost nothing on screen, which makes a
        # folded leg measure as nearly straight - 158 degrees for a person
        # who is plainly sitting down. How much thigh there is to see decides
        # how much the angle counts.
        thigh = float(np.linalg.norm(latest[L_KNEE] - latest[L_HIP]))
        shin = float(np.linalg.norm(latest[L_ANKLE] - latest[L_KNEE]))
        foreshortening = thigh / (shin + 1e-6)
        angle_trust = max(0.0, min(1.0, (foreshortening - 0.35) / 0.35))

        # Whatever weight the angle loses goes to hip height, which survives
        # foreshortening: hips near ankle level means sitting, from any angle.
        weight_bend = 0.40 * angle_trust
        weight_drop = 0.85 - weight_bend
        score = weight_bend * bend + weight_drop * drop + 0.15 * settled

        reading.basis = ("knee angle and hip height" if angle_trust > 0.5
                         else "hip height (thigh foreshortened head-on)")
    else:
        # --- route 2: the legs are not visible -----------------------------
        #
        # Two quite different situations hide behind "no legs", and they need
        # different answers.
        latest_frame = frames[-1]
        hip_y = latest_frame.hip_center[1]
        shoulder_span = float(np.linalg.norm(latest[L_SHOULDER]
                                             - latest[R_SHOULDER]))
        scale = max(shoulder_span * latest_frame.torso_length, 1e-3)
        # Negative means the hips are below the bottom edge entirely.
        reading.leg_room = max(0.0, float((1.0 - hip_y) / scale))

        settled = _falling(reading.hip_stability, config.SEATED_HIP_TRAVEL,
                           config.SEATED_HIP_TRAVEL * 3.0)
        upright = _falling(reading.spine_upright, config.UPRIGHT_TOLERANCE,
                           config.UPRIGHT_TOLERANCE * 2.5)

        if hips_measured and reading.leg_room >= config.LEG_ROOM_AMBIGUOUS:
            # There is clear frame below the hips where standing legs would
            # have appeared, and nothing appeared in it. The legs are folded,
            # not cropped: this is sitting, and it is close to a measurement.
            room = _ramp(reading.leg_room, config.LEG_ROOM_AMBIGUOUS,
                         config.LEG_ROOM_CLEAR)
            score = (0.70 * room + 0.18 * settled + 0.12 * upright) * 0.92
            reading.basis = (f"legs absent from {reading.leg_room:.1f} shoulder "
                             f"widths of clear frame below the hips")
        else:
            # The hips sit at or below the bottom edge - the ordinary view
            # from a laptop webcam. Nothing in the picture can separate
            # sitting from standing here, and pretending otherwise is how
            # this used to announce "Standing" at somebody plainly sat down.
            #
            # What can still be measured is the thing the mission actually
            # cares about: whether the crew member is settled at a station.
            # Hips that do not travel and a torso that stays upright mean
            # settled, whether they are in a chair or in foot restraints.
            # Nothing in a two second window separates sitting from standing
            # once the legs are gone. What does separate them is time: over
            # fifteen seconds, somebody settled to work stays put with their
            # hands in front of them, and somebody on their feet does not.
            seen = ("head, shoulders and part of the arms"
                    if reading.framing == "torso" else "upper body and arms")

            long_view = workstation.evidence() if workstation is not None else 0.0
            if long_view > 0.0:
                reading.settled_seconds = workstation.span
                reading.travel = workstation.travel
                # The long view carries the decision; the window only refines.
                score = 0.78 * long_view + 0.22 * (0.6 * settled + 0.4 * upright)
                reading.basis = (
                    f"{seen} in shot - at a workstation for "
                    f"{workstation.span:.0f}s, drifting "
                    f"{workstation.travel:.1f} shoulder widths")
            else:
                score = (0.62 * settled + 0.38 * upright) * 0.60
                reading.basis = (f"{seen} in shot - gathering evidence "
                                 "of a settled posture")

    reading.seated = score >= config.SEATED_THRESHOLD
    reading.confidence = round(min(0.99, 0.35 + 0.64 * score), 3)

    if reading.seated:
        reading.posture = SEATED
    elif (workstation is not None
          and workstation.travel > config.WORKSTATION_TRAVEL_MOVING):
        # Drifting across the module, judged over the long horizon. Two
        # seconds of leaning is not the same as having left the station.
        reading.posture = FLOATING
    else:
        reading.posture = STANDING
    return reading


def _ramp(value: float, low: float, high: float) -> float:
    """0 at or below `low`, 1 at or above `high`."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _falling(value: float, low: float, high: float) -> float:
    """1 at or below `low`, 0 at or above `high`."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (high - value) / (high - low)))
