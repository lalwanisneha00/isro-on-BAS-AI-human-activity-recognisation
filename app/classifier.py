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
from .posture import WorkstationWatch, read as read_posture


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


def score_activities(f, face_recency: float = 0.0, posture=None,
                     roles=None, held_roles=None) -> dict:
    """Score every activity from one window of motion, posture and objects.

    `roles` is what the whitelisted objects in frame can support; `held_roles`
    is the subset actually in a hand. Objects sharpen the answer - a hand at
    the mouth is the same skeleton whether it is drinking or eating - but they
    never decide it alone. Every activity below still scores from pose, so the
    system keeps working when nothing is detected, which is most of the time.
    """
    scores = {name: 0.0 for name in config.ACTIVITY_ORDER}
    roles = roles or set()
    held = held_roles or set()

    # Hands at the face count for a while after the fact, not just right now.
    effective_head = max(f.head_zone_fraction, 0.55 * face_recency)

    still_body = _falling(f.body_speed, config.BODY_SPEED_STILL,
                          config.BODY_SPEED_MOVING)
    settled = _falling(f.wrist_speed, config.WRIST_SPEED_FOCUSED,
                       config.WRIST_SPEED_FAST)
    doing_something = _ramp(f.wrist_speed, config.WRIST_SPEED_STILL,
                            config.WRIST_SPEED_FOCUSED)
    fine_work = _falling(f.wrist_range, config.FOCUSED_MAX_RANGE,
                         config.FOCUSED_MAX_RANGE + config.FOCUSED_RANGE_TAPER)
    head_down = _ramp(f.head_down_fraction, 0.35, 0.75)
    at_chest = _ramp(f.workstation_fraction, 0.45, 0.90)

    def boost(role, strength=config.OBJECT_BOOST):
        """How much a supporting object lifts a score, if one is present."""
        if role in held:
            return 1.0 + strength
        if role in roles:
            return 1.0 + strength * 0.45
        return 1.0

    # ================================================== Tier 1: the core ====
    # Exercise: hands sweep a wide arc, briskly, over and over.
    scores[config.ACT_EXERCISE] = (
        _ramp(f.wrist_range, config.EXERCISE_MIN_RANGE, config.EXERCISE_FULL_RANGE)
        * _ramp(f.peak_wrist_speed, config.WRIST_SPEED_FOCUSED,
                config.WRIST_SPEED_FAST)
        * (0.45 + 0.55 * _ramp(f.periodicity, config.PERIODICITY_RHYTHMIC,
                               config.PERIODICITY_STRONG))
    )

    # In-Transit: the whole body crosses the module.
    scores[config.ACT_TRANSIT] = _ramp(f.body_speed, config.BODY_SPEED_MOVING,
                                       config.BODY_SPEED_FAST)

    # Drinking and Eating share a skeleton - a hand at the mouth - and are
    # told apart by what is in that hand. With nothing detected, the pose-only
    # reading is Eating, because a repeated hand-to-mouth cycle is the more
    # common of the two and the log stays honest either way.
    # A hand near the face is not enough on its own: bowing the head over a
    # manual brings the face down to the hands and measures exactly the same.
    # What separates them is travel - a bite carries the hand up from the lap
    # and back, while reading holds it still - so hand-to-mouth requires the
    # hand to have actually made that journey.
    reached_up = _ramp(f.wrist_range, config.HAND_TO_MOUTH_TRAVEL,
                       config.HAND_TO_MOUTH_TRAVEL * 2.4)
    hand_to_mouth = _ramp(effective_head, 0.20, 0.65) * still_body * reached_up
    scores[config.ACT_DRINKING] = (
        hand_to_mouth * settled
        * (config.OBJECT_BOOST if "drinking" in held else
           0.35 if "drinking" in roles else 0.0)
    )
    scores[config.ACT_EATING] = (
        hand_to_mouth
        * _falling(f.peak_wrist_speed, config.WRIST_SPEED_FAST,
                   config.WRIST_SPEED_VIGOROUS)
        * boost("eating")
        # A drink container in hand means this is drinking, not a meal.
        * (0.35 if "drinking" in held else 1.0)
    )

    # Laptop Work: a machine in view, hands worked in front of it, body still.
    scores[config.ACT_LAPTOP] = (
        at_chest * doing_something * fine_work * still_body
        * _falling(effective_head, 0.15, 0.45)
        * (config.OBJECT_BOOST if "laptop" in roles else 0.0)
    )

    # Reading Procedure: something held up, head angled down onto it, still.
    scores[config.ACT_READING] = (
        head_down * still_body * settled
        * _ramp(f.workstation_fraction, 0.35, 0.80)
        # Hands held still is the other half of reading: something held and
        # looked at, not something being worked on.
        * _falling(f.wrist_range, config.FOCUSED_MAX_RANGE * 0.7,
                   config.FOCUSED_MAX_RANGE)
        * boost("reading")
        # A manual in view makes this certain. Without one, head down over
        # still hands is still the best available reading, just less sure.
        * (0.60 if not roles & {"reading", "tablet"} else 1.0)
    )

    # Experiment Operation: hands working a fixed forward zone, finely, with
    # no object that would name the task more precisely.
    scores[config.ACT_EXPERIMENT] = (
        at_chest * doing_something * fine_work * still_body
        * _ramp(f.wrist_range, config.FOCUSED_MIN_RANGE,
                config.FOCUSED_MIN_RANGE * 2.5)
        * _falling(effective_head, 0.15, 0.45)
        * _falling(f.elevated_fraction, 0.25, 0.60)
        * boost("experiment")
        # A laptop or manual in view names the task better than this does.
        * (0.45 if roles & {"laptop", "reading"} else 1.0)
    )

    # Maintenance: a tool in hand, or arms raised and reaching, in bursts.
    posture_term = max(_ramp(f.elevated_fraction, 0.25, 0.75),
                       _ramp(f.extended_fraction, 0.25, 0.75))
    scores[config.ACT_MAINTENANCE] = (
        max(posture_term,
            config.OBJECT_ONLY_FLOOR if "maintenance" in held else 0.0)
        * _band(f.wrist_speed, config.WRIST_SPEED_STILL * 1.5,
                config.WRIST_SPEED_FAST, 0.35)
        * _falling(f.periodicity, config.PERIODICITY_RHYTHMIC,
                   config.PERIODICITY_STRONG)
        * boost("maintenance")
    )

    # Seated at Workstation: settled at a station with nothing more specific
    # happening. Being seated is a posture, not a task, so it competes with
    # Idle and is reported alongside whatever else is going on.
    if posture is not None and posture.seated:
        scores[config.ACT_SEATED] = (
            posture.confidence
            * _falling(f.wrist_speed, config.WRIST_SPEED_STILL,
                       config.WRIST_SPEED_FOCUSED)
            * still_body
            # Seated is the "settled, nothing more specific" reading. A head
            # bowed over the hands means there IS something more specific -
            # reading, writing, a tablet - so this gives way to it.
            * _falling(f.head_down_fraction, 0.35, 0.70)
        )

    # Idle: present and calm.
    scores[config.ACT_IDLE] = (
        _falling(f.wrist_speed, config.WRIST_SPEED_STILL,
                 config.WRIST_SPEED_FOCUSED)
        * still_body
        * (0.55 if posture is not None and posture.seated else 1.0)
    )

    # ================================================== Tier 2 ==============
    # Holding Object: something is in hand but nothing above describes it.
    scores[config.ACT_HOLDING] = (0.42 if held else 0.0) * still_body

    # Tablet/Device Use: a handheld device, head down over it, small motion.
    scores[config.ACT_TABLET] = (
        head_down * still_body * fine_work
        * (config.OBJECT_BOOST if "tablet" in held else 0.0)
    )

    # Sample Handling: both hands working together, precisely, at chest level.
    scores[config.ACT_SAMPLE] = (
        _ramp(f.both_hands_fraction, 0.45, 0.85)
        * _falling(f.wrist_gap, 0.85, 1.40)
        * doing_something * fine_work * still_body
        * boost("sample")
        * (0.40 if roles & {"laptop", "reading"} else 1.0)
        # Sample handling means manipulating something. With nothing detected
        # in the hands, two-handed fine work at a console is better described
        # as operating the experiment than as handling a sample.
        * (1.0 if "sample" in roles else 0.45)
    )

    # Stowage/Retrieval: reaching out to the side, torso turning with it.
    scores[config.ACT_STOWAGE] = (
        _ramp(f.peripheral_fraction, 0.30, 0.70)
        * _ramp(max(f.rotation_fraction, f.extended_fraction), 0.20, 0.60)
        * _falling(f.periodicity, config.PERIODICITY_RHYTHMIC,
                   config.PERIODICITY_STRONG)
        * boost("stowage")
    )

    # Communication: squared up to the camera, gesturing, hands empty.
    scores[config.ACT_COMMS] = (
        _ramp(f.shoulder_ratio, 0.34, 0.42)
        * _band(f.wrist_speed, config.WRIST_SPEED_FOCUSED,
                config.WRIST_SPEED_FAST, 0.30)
        * _falling(f.head_down_fraction, 0.25, 0.60)
        * still_body
        * (0.30 if held else 1.0)
        * _band(f.wrist_range, 0.20, 0.60, 0.20)
        # Gesturing while speaking is irregular. Steady repetition at this
        # range is an exercise rep, not a conversation.
        * _falling(f.periodicity, config.PERIODICITY_RHYTHMIC * 0.8,
                   config.PERIODICITY_STRONG)
    )

    # Photography/Observation: a device held up at eye level, body still.
    scores[config.ACT_PHOTO] = (
        _ramp(f.eye_level_fraction, 0.40, 0.80)
        * still_body * settled
        * (config.OBJECT_BOOST if "photography" in held else 0.0)
    )

    # Writing/Logging: a manual or surface, and small repeated one-hand motion.
    scores[config.ACT_WRITING] = (
        head_down * still_body * fine_work
        * _ramp(f.wrist_gap, 0.55, 1.10)          # one hand works, one rests
        * doing_something
        * (config.OBJECT_BOOST if roles & {"reading"} else 0.0)
    )

    # ================================================== Tier 3 ==============
    # Health Check/Medical: a hand held against the crew member's own torso.
    scores[config.ACT_MEDICAL] = (
        _ramp(f.torso_contact_fraction, 0.55, 0.90)
        * still_body * settled
        * _falling(f.wrist_range, 0.22, 0.45)
    )

    # Hygiene: a hygiene item worked near the head, small and repetitive.
    scores[config.ACT_HYGIENE] = (
        _ramp(effective_head, 0.35, 0.75)
        * still_body * fine_work
        * (config.OBJECT_BOOST if "hygiene" in held else 0.0)
    )

    # Donning/Doffing: big, slow arm movements crossing the body.
    scores[config.ACT_DONNING] = (
        _ramp(f.cross_body_fraction, 0.35, 0.75)
        * _ramp(f.wrist_range, 0.45, 0.95)
        * _falling(f.peak_wrist_speed, config.WRIST_SPEED_FAST,
                   config.WRIST_SPEED_VIGOROUS)
        * _falling(f.periodicity, config.PERIODICITY_RHYTHMIC,
                   config.PERIODICITY_STRONG)
        * still_body
    )

    # Anomaly is not scored here - it is a sustained-stillness state tracked
    # over time, well beyond a single two second window.
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
        self.posture = None
        # Watches over a far longer horizon than the classification window,
        # because being settled at a station is a matter of duration.
        self.workstation = WorkstationWatch()

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
    def update(self, window, roles=None, held_roles=None):
        """Classify the current window. Safe to call every frame.

        `roles` / `held_roles` describe the whitelisted objects in frame and
        in hand. Both are optional: with nothing detected the classifier
        works from pose alone, which is the normal case.
        """
        if window.count == 0:
            self.workstation.observe(None, time.time())
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

        self.workstation.observe(window.latest, time.time())
        self.posture = read_posture(window, self.workstation)

        # Remember the last time a hand was up at the mouth, so the meal label
        # holds through the hand-down phase of each bite.
        now = time.time()
        if features.head_zone_fraction >= config.FACE_CONTACT_TRIGGER:
            self._last_face_contact = now
        face_recency = 0.0
        if self._last_face_contact is not None:
            elapsed = max(0.0, now - self._last_face_contact)
            face_recency = _falling(elapsed, 1.0, config.EATING_MEMORY_SECONDS)

        scores = score_activities(features, face_recency, self.posture,
                                  roles, held_roles)

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

        total = sum(scores.values()) or 1.0
        share = best_score / total
        confidence = 0.30 + (0.42 * share) + (0.28 * best_score)
        confidence = round(max(0.30, min(0.99, confidence)), 3)

        # Saying "I am not sure" is more useful than a confident wrong answer,
        # and it is what stops the log filling with guesses during the moments
        # between one activity and the next.
        if best_score <= config.UNCERTAIN_SCORE_FLOOR or \
                confidence < config.UNCERTAIN_BELOW:
            best_label = config.ACT_UNCERTAIN
            confidence = round(max(0.20, min(config.UNCERTAIN_BELOW,
                                             confidence)), 3)

        # ------------------------------------------------------ hysteresis ---
        # A new label must win several frames running, so the readout does not
        # flicker between neighbouring activities on borderline motion.
        if best_label == self.label:
            self._candidate, self._candidate_count = None, 0
            self._commit(best_label, confidence)
            return self.label, self.confidence

        # Hysteresis protects a real answer from flickering. It must never
        # hold onto a non-answer: with a long task list the winner can change
        # from frame to frame while the window is warming up, so no single
        # candidate ever accumulates enough frames and the console would sit
        # on "Acquiring" indefinitely.
        if self.label in (config.ACT_ACQUIRING, config.ACT_NO_CREW):
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
