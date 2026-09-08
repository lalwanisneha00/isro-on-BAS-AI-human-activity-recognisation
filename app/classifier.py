"""Rule-based activity classifier for BAS crew monitoring, v1.

Each of the seven activities gets a 0-1 score built from the motion features,
and the strongest score wins. Scoring every class rather than running a chain
of if/else tests gives two things a cascade cannot: a meaningful confidence
value, and a readable breakdown of why a label was chosen.

The rules are deliberately simple and physically motivated - v1 needs no
training data, and every threshold sits in config.py where it can be tuned
against real footage.
"""

import time

from . import config
from .features import extract


def _ramp(value: float, low: float, high: float) -> float:
    """0 below `low`, 1 above `high`, linear in between."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _falling(value: float, low: float, high: float) -> float:
    """1 below `low`, 0 above `high` - the inverse ramp."""
    return 1.0 - _ramp(value, low, high)


def _band(value: float, low: float, high: float, edge: float) -> float:
    """1 inside [low, high], tapering to 0 within `edge` of either side."""
    if value < low:
        return _ramp(value, low - edge, low)
    if value > high:
        return _falling(value, high, high + edge)
    return 1.0


def score_activities(f, face_recency: float = 0.0) -> dict:
    """Score all seven activities from one window of motion features.

    `face_recency` is 1.0 just after a hand was at the mouth and decays to 0
    over the eating memory window, so a meal survives the gaps between bites.
    """
    scores = {}

    # Hands at the face count for a while after the fact, not just right now.
    effective_head = max(f.head_zone_fraction, 0.55 * face_recency)

    # Exercise: the hands sweep a wide arc, briskly, over and over. Reach is
    # the gate - a fast small movement is fine work, not a rep - and genuine
    # repetition is what lifts the score the rest of the way.
    range_term = _ramp(f.wrist_range, config.EXERCISE_MIN_RANGE,
                       config.EXERCISE_FULL_RANGE)
    speed_term = _ramp(f.peak_wrist_speed, config.WRIST_SPEED_FOCUSED,
                       config.WRIST_SPEED_FAST)
    rhythm_term = _ramp(f.periodicity, config.PERIODICITY_RHYTHMIC,
                        config.PERIODICITY_STRONG)
    scores[config.ACT_EXERCISE] = (range_term * speed_term
                                   * (0.45 + 0.55 * rhythm_term))

    # In-Transit: the whole body drifts across the frame.
    scores[config.ACT_TRANSIT] = _ramp(f.body_speed, config.BODY_SPEED_MOVING,
                                       config.BODY_SPEED_FAST)

    # Eating/Rest: a hand returns to the face while the body stays put.
    scores[config.ACT_EATING] = (
        _ramp(effective_head, 0.20, 0.65)
        * _falling(f.body_speed, config.BODY_SPEED_STILL, config.BODY_SPEED_MOVING)
        * _falling(f.peak_wrist_speed, config.WRIST_SPEED_FAST,
                   config.WRIST_SPEED_VIGOROUS)
    )

    # Maintenance: arms raised or reaching out, working in bursts.
    posture_term = max(_ramp(f.elevated_fraction, 0.25, 0.75),
                       _ramp(f.extended_fraction, 0.25, 0.75))
    scores[config.ACT_MAINTENANCE] = (
        posture_term
        * _band(f.wrist_speed, config.WRIST_SPEED_STILL * 1.5,
                config.WRIST_SPEED_FAST, 0.35)
        # Working with a tool is intermittent, not metronomic; steady
        # repetition at a wide reach is exercise.
        * _falling(f.periodicity, config.PERIODICITY_RHYTHMIC,
                   config.PERIODICITY_STRONG)
    )

    # Experiment Operation: hands held in a forward workstation zone, moving
    # in small focused increments rather than sweeping around. The hands must
    # actually be doing something - motionless hands in the same zone are Idle,
    # and hands up at the face are eating, not operating a rack.
    scores[config.ACT_EXPERIMENT] = (
        _ramp(f.workstation_fraction, 0.45, 0.90)
        # The hands must be doing something, but within a small area: that is
        # the whole difference between operating a rack and exercising.
        * _ramp(f.wrist_speed, config.WRIST_SPEED_STILL, config.WRIST_SPEED_FOCUSED)
        * _ramp(f.wrist_range, config.FOCUSED_MIN_RANGE,
                config.FOCUSED_MIN_RANGE * 2.5)
        * _falling(f.wrist_range, config.FOCUSED_MAX_RANGE,
                   config.FOCUSED_MAX_RANGE + config.FOCUSED_RANGE_TAPER)
        * _falling(effective_head, 0.15, 0.45)
        * _falling(f.elevated_fraction, 0.25, 0.60)
        * _falling(f.body_speed, config.BODY_SPEED_STILL, config.BODY_SPEED_MOVING)
    )

    # Idle: present and calm, nothing much happening anywhere.
    scores[config.ACT_IDLE] = (
        _falling(f.wrist_speed, config.WRIST_SPEED_STILL,
                 config.WRIST_SPEED_FOCUSED)
        * _falling(f.body_speed, config.BODY_SPEED_STILL,
                   config.BODY_SPEED_MOVING)
    )

    # Anomaly is not scored here - it is a sustained-stillness state that the
    # classifier tracks over time, well beyond a single 2 second window.
    scores[config.ACT_ANOMALY] = 0.0
    return scores


class ActivityClassifier:
    """Turns each window into a stable, confidence-scored activity label."""

    def __init__(self):
        self.label = config.ACT_NO_CREW
        self.confidence = 0.0
        self.scores = {}
        self.features = None
        self.since = time.time()

        self._candidate = None
        self._candidate_count = 0
        self._still_since = None
        self._last_face_contact = None

    # ------------------------------------------------------------- helpers --
    @property
    def duration(self) -> float:
        return time.time() - self.since

    @property
    def still_seconds(self) -> float:
        if self._still_since is None:
            return 0.0
        return max(0.0, time.time() - self._still_since)

    def _commit(self, label: str, confidence: float) -> None:
        """Adopt a label, resetting the duration clock only on a real change."""
        if label != self.label:
            self.label = label
            self.since = time.time()
        self.confidence = confidence

    # ---------------------------------------------------------------- main --
    def update(self, window):
        """Classify the current window. Safe to call every frame."""
        if window.count == 0:
            self.scores, self.features = {}, None
            self._candidate, self._candidate_count = None, 0
            self._still_since = None
            self._last_face_contact = None
            self._commit(config.ACT_NO_CREW, 0.0)
            return self.label, self.confidence

        if not window.is_ready:
            self.scores, self.features = {}, None
            self._commit(config.ACT_ACQUIRING, 0.0)
            return self.label, self.confidence

        features = extract(window)
        self.features = features

        # Remember the last time a hand was up at the mouth, so the meal label
        # holds through the hand-down phase of each bite.
        now = time.time()
        if features.head_zone_fraction >= config.FACE_CONTACT_TRIGGER:
            self._last_face_contact = now
        face_recency = 0.0
        if self._last_face_contact is not None:
            elapsed = max(0.0, now - self._last_face_contact)
            face_recency = _falling(elapsed, 1.0, config.EATING_MEMORY_SECONDS)

        scores = score_activities(features, face_recency)

        # -------------------------------------------------- stillness watch --
        # Tracked across windows, since the anomaly threshold is many seconds.
        is_still = (features.wrist_range < config.ANOMALY_STILL_RANGE
                    and features.body_speed < config.BODY_SPEED_STILL)
        if is_still:
            if self._still_since is None:
                self._still_since = time.time()
        else:
            self._still_since = None

        if self.still_seconds >= config.ANOMALY_STILL_SECONDS:
            # Idle and Anomaly are the same posture; only elapsed time tells
            # them apart, so once the threshold is crossed Idle must give way.
            over = self.still_seconds - config.ANOMALY_STILL_SECONDS
            scores[config.ACT_ANOMALY] = min(1.0, 0.90 + 0.02 * over)
            scores[config.ACT_IDLE] = 0.0

        self.scores = scores

        # --------------------------------------------------------- winner ----
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score <= 0.02:
            # Nothing matched cleanly; a calm default beats a wrong guess.
            best_label, best_score, runner_score = config.ACT_IDLE, 0.35, 0.0

        total = sum(scores.values()) or 1.0
        share = best_score / total
        confidence = 0.30 + (0.42 * share) + (0.28 * best_score)
        confidence = round(max(0.30, min(0.99, confidence)), 3)

        # ------------------------------------------------------ hysteresis ---
        # A new label must win several frames running, so the readout does not
        # flicker between neighbouring activities on borderline motion.
        if best_label == self.label:
            self._candidate, self._candidate_count = None, 0
            self._commit(best_label, confidence)
            return self.label, self.confidence

        if best_label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = best_label, 1

        # An anomaly is safety-critical, so it is allowed to switch immediately.
        instant = best_label == config.ACT_ANOMALY
        if instant or self._candidate_count >= config.LABEL_HOLD_FRAMES:
            self._candidate, self._candidate_count = None, 0
            self._commit(best_label, confidence)
        else:
            self.confidence = confidence

        return self.label, self.confidence
