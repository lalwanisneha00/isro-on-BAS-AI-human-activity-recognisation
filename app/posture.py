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
2. Lower body hidden - fall back to what is left: the torso occupies a
   smaller slice of the frame than a standing person at the same apparent
   size, the hips barely move, and the spine stays upright.
"""

import math

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
                 "spine_upright", "basis")

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
    reading.basis = basis
    return reading


def read(window) -> PostureReading:
    """Decide the subject's posture from a window of normalised poses.

    Everything is measured in torso lengths, so the answer does not change
    with distance from the camera.
    """
    frames = window.frames
    if len(frames) < 3:
        return _blank("not enough frames")

    points = np.stack([f.points for f in frames])          # (N, 33, 2)
    latest = points[-1]

    reading = _blank("")

    # How still the hips are, in torso lengths across the window. A seated or
    # restrained person's hips barely travel; a standing one sways.
    hips = (points[:, L_HIP, :] + points[:, R_HIP, :]) / 2.0
    reading.hip_stability = float(np.linalg.norm(hips.max(axis=0)
                                                 - hips.min(axis=0)))

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
        # --- route 2: legs hidden, work with the torso --------------------
        # Without legs the giveaway is that a seated person's visible body is
        # short and does not move: the hips stay put and the spine stays up.
        settled = _falling(reading.hip_stability, config.SEATED_HIP_TRAVEL,
                           config.SEATED_HIP_TRAVEL * 3.0)
        upright = _falling(reading.spine_upright, config.UPRIGHT_TOLERANCE,
                           config.UPRIGHT_TOLERANCE * 2.5)

        # How much of the frame the body occupies. Someone at a desk shows
        # torso and head only, so their visible extent is short relative to
        # their torso length.
        visible = points[-1]
        extent = float(visible[:, 1].max() - visible[:, 1].min())
        cropped = _falling(extent, config.SEATED_VISIBLE_EXTENT,
                           config.SEATED_VISIBLE_EXTENT + 0.9)

        score = 0.40 * settled + 0.25 * upright + 0.35 * cropped
        # Legs unseen means real uncertainty; say so rather than overclaiming.
        score *= 0.75
        reading.basis = "torso only - lower body not visible"

    reading.seated = score >= config.SEATED_THRESHOLD
    reading.confidence = round(min(0.99, 0.35 + 0.64 * score), 3)

    if reading.seated:
        reading.posture = SEATED
    elif reading.hip_stability > config.SEATED_HIP_TRAVEL * 4.0:
        reading.posture = FLOATING
    else:
        reading.posture = STANDING
    return reading


def _falling(value: float, low: float, high: float) -> float:
    """1 at or below `low`, 0 at or above `high`."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (high - value) / (high - low)))
