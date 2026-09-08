"""Choosing exactly one person to monitor, and staying with them.

v1 monitors a single crew member. That is a deliberate design decision, not a
missing feature: tracking several people at once cost real accuracy, because a
detector asked for several poses returns weak ones too, and a window that
silently swaps from one person to another has the classifier reading a blend
of two people's movements.

So the detector is allowed to see several candidates, and this module picks
one and holds onto them:

* The subject is whoever is closest to the camera - the person the console is
  pointed at, rather than someone crossing the module behind them.
* Once chosen, the lock is sticky. Somebody walking past, or standing nearby,
  does not steal it. The lock survives a brief occlusion, so a hand passing in
  front of the lens does not restart the activity history.
* A different person only takes over if they are clearly and persistently
  closer, or if the locked subject has genuinely gone.

Everything downstream - normalisation, the rolling window, the classifier, the
log - sees exactly one person, always.
"""

import time

from . import config
from .classifier import ActivityClassifier
from .normalize import PoseWindow


def _size_of(pose) -> float:
    """How large the person appears, as a stand-in for how close they are.

    Torso length is used rather than the full bounding box. A bounding box
    grows and shrinks whenever someone raises an arm, which would make the
    "closest person" flicker between candidates; the torso does not change
    size when a limb moves.
    """
    return float(pose.torso_length)


class SubjectLock:
    """Follows one crew member, with their own window and classifier."""

    def __init__(self):
        self.window = PoseWindow()
        self.classifier = ActivityClassifier()

        self.locked = False
        self.pose = None                # this frame's pose for the subject
        self.hip_center = None
        self.torso_length = 0.0

        self.locked_at = None
        self.last_seen = None
        self.candidates = 0             # people visible this frame
        self.ignored = 0                # candidates deliberately not followed
        self.switches = 0               # times the lock moved to someone else

        self._challenger_since = None
        self._challenger_size = 0.0

    # ----------------------------------------------------------- reporting --
    @property
    def label(self) -> str:
        return self.classifier.label

    @property
    def confidence(self) -> float:
        return self.classifier.confidence

    @property
    def is_anomaly(self) -> bool:
        return self.classifier.label == config.ACT_ANOMALY

    @property
    def lock_seconds(self) -> float:
        if not self.locked or self.locked_at is None:
            return 0.0
        return max(0.0, time.time() - self.locked_at)

    @property
    def visible(self) -> bool:
        return self.pose is not None

    def state(self) -> dict:
        clf = self.classifier
        return {
            "locked": self.locked,
            "visible": self.visible,
            "lock_seconds": round(self.lock_seconds, 1),
            "candidates": self.candidates,
            "ignored": self.ignored,
            "switches": self.switches,
            "activity": clf.label,
            "confidence": round(clf.confidence, 3),
            "duration": round(clf.duration, 1),
            "still_seconds": round(clf.still_seconds, 1),
            "is_anomaly": self.is_anomaly,
            "buffer": self.window.count,
            "posture": (self.classifier.posture.as_dict()
                        if self.classifier.posture else None),
            "scores": {name: round(clf.scores.get(name, 0.0), 3)
                       for name in config.ACTIVITY_ORDER},
        }

    # ------------------------------------------------------------- matching --
    def _distance_to(self, pose) -> float:
        """How far this candidate is from where the subject last was."""
        if self.hip_center is None:
            return float("inf")
        scale = max(self.torso_length, config.MIN_TORSO_LENGTH)
        dx = pose.hip_center[0] - self.hip_center[0]
        dy = pose.hip_center[1] - self.hip_center[1]
        return ((dx * dx + dy * dy) ** 0.5) / scale

    def _adopt(self, pose, now: float, switched: bool) -> None:
        if switched or not self.locked:
            # A new person means the buffered movement belongs to someone
            # else. Carrying it forward would classify one person using
            # another's history.
            self.window.clear()
            self.classifier = ActivityClassifier()
            self.locked_at = now
            if switched:
                self.switches += 1
        self.locked = True
        self._challenger_since = None

    def reset(self) -> None:
        self.__init__()

    # --------------------------------------------------------------- update --
    def update(self, poses: list, now: float = None,
               roles=None, held_roles=None):
        """Pick this frame's subject from the candidates and classify them.

        `roles` / `held_roles` are the whitelisted objects in frame and in
        hand; they refine the activity but are never required.
        """
        now = time.time() if now is None else now
        poses = [p for p in poses if p is not None]

        self.candidates = len(poses)
        self.ignored = 0
        self.pose = None

        if not poses:
            # Nobody in frame. Hold the lock briefly, so a person who steps
            # behind a rack keeps their identity and their activity history.
            if self.locked and self.last_seen is not None:
                if now - self.last_seen > config.SUBJECT_GRACE_SECONDS:
                    self.locked = False
                    self.window.clear()
                    self.classifier = ActivityClassifier()
            self.window.push(None)
            self.classifier.update(self.window, roles, held_roles)
            return None

        ranked = sorted(poses, key=_size_of, reverse=True)
        biggest = ranked[0]

        chosen, switched = None, False

        if self.locked:
            # Prefer whoever is where the subject was, however large they look.
            matches = [(self._distance_to(p), p) for p in poses]
            matches.sort(key=lambda pair: pair[0])
            distance, nearest = matches[0]

            if distance <= config.SUBJECT_MATCH_RADIUS:
                chosen = nearest

                # Somebody else may take over, but only by being clearly
                # closer, and only if they stay that way. A single frame of a
                # passer-by looming large must not move the lock.
                challenger_wins = (
                    biggest is not chosen
                    and _size_of(biggest) > _size_of(chosen)
                    * config.SUBJECT_SWITCH_RATIO
                )
                if challenger_wins:
                    if self._challenger_since is None:
                        self._challenger_since = now
                    elif now - self._challenger_since >= config.SUBJECT_SWITCH_SECONDS:
                        chosen, switched = biggest, True
                else:
                    self._challenger_since = None
            else:
                # Nobody where the subject was. Wait out the grace period
                # before handing the lock to whoever else is present.
                if self.last_seen is not None and \
                        now - self.last_seen <= config.SUBJECT_GRACE_SECONDS:
                    self.window.push(None)
                    self.classifier.update(self.window, roles, held_roles)
                    self.ignored = len(poses)
                    return None
                chosen, switched = biggest, True

        if chosen is None:
            chosen = biggest

        self._adopt(chosen, now, switched)

        self.pose = chosen
        self.hip_center = chosen.hip_center
        self.torso_length = chosen.torso_length
        self.last_seen = now
        self.ignored = len(poses) - 1

        self.window.push(chosen)
        self.classifier.update(self.window, roles, held_roles)
        return chosen
