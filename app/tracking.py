"""Crew identity tracking across frames.

MediaPipe reports the poses it can see in a frame, but says nothing about which
pose belongs to which person - the order can change from one frame to the next.
Without identity, six crew members would share one scrambled activity history.

This module keeps a track per crew member. Each frame, detections are matched
to existing tracks by how far the hip centre moved, measured in torso lengths
so the gate means the same thing whether someone is near the lens or across the
module. Each track owns its own rolling window and classifier, so every crew
member is recognised independently.

A track survives a short disappearance - someone turning away, or passing
behind a rack - and only retires once they have been gone long enough that
reusing their identity would be a guess.
"""

import time

from . import config
from .classifier import ActivityClassifier
from .normalize import PoseWindow


class CrewTrack:
    """One crew member: their identity, their pose window, their classifier."""

    def __init__(self, track_id: int, slot: int, pose, now: float):
        self.id = track_id
        self.slot = slot                     # 1..MAX_CREW, drives the display name
        self.name = f"{config.CREW_NAME_PREFIX}-{slot}"
        self.window = PoseWindow()
        self.classifier = ActivityClassifier()

        self.first_seen = now
        self.last_seen = now
        self.hip_center = pose.hip_center
        self.torso_length = pose.torso_length
        self.visible = True

    # ---------------------------------------------------------------- state --
    @property
    def label(self) -> str:
        return self.classifier.label

    @property
    def confidence(self) -> float:
        return self.classifier.confidence

    @property
    def is_anomaly(self) -> bool:
        return self.classifier.label == config.ACT_ANOMALY

    def observe(self, pose, now: float) -> None:
        """Take this frame's pose for this crew member."""
        self.last_seen = now
        self.visible = True
        self.hip_center = pose.hip_center
        self.torso_length = pose.torso_length
        self.window.push(pose)
        self.classifier.update(self.window)

    def miss(self, now: float) -> None:
        """This crew member was not seen this frame."""
        self.visible = False
        self.window.push(None)
        self.classifier.update(self.window)

    def distance_to(self, pose) -> float:
        """Hip-centre movement since last seen, in torso lengths."""
        scale = max(self.torso_length, config.MIN_TORSO_LENGTH)
        dx = pose.hip_center[0] - self.hip_center[0]
        dy = pose.hip_center[1] - self.hip_center[1]
        return ((dx * dx + dy * dy) ** 0.5) / scale

    def as_dict(self) -> dict:
        clf = self.classifier
        return {
            "id": self.id,
            "slot": self.slot,
            "name": self.name,
            "activity": clf.label,
            "confidence": round(clf.confidence, 3),
            "duration": round(clf.duration, 1),
            "still_seconds": round(clf.still_seconds, 1),
            "is_anomaly": self.is_anomaly,
            "visible": self.visible,
            "tracked_for": round(time.time() - self.first_seen, 1),
            "buffer": self.window.count,
            "scores": {name: round(clf.scores.get(name, 0.0), 3)
                       for name in config.ACTIVITY_ORDER},
        }


class CrewTracker:
    """Maintains up to MAX_CREW identities frame by frame."""

    def __init__(self, max_crew: int = config.MAX_CREW):
        self.max_crew = max_crew
        self._tracks = {}
        self._next_id = 1

    # ------------------------------------------------------------- querying --
    @property
    def tracks(self) -> list:
        """Active tracks, ordered by display slot so the roster never jumps."""
        return sorted(self._tracks.values(), key=lambda t: t.slot)

    @property
    def visible_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.visible)

    def _free_slot(self) -> int:
        taken = {t.slot for t in self._tracks.values()}
        for slot in range(1, self.max_crew + 1):
            if slot not in taken:
                return slot
        return self.max_crew

    def reset(self) -> None:
        self._tracks.clear()

    # -------------------------------------------------------------- updating --
    def update(self, poses: list, now: float = None) -> list:
        """Match this frame's poses to crew identities.

        Returns the list of (track, pose) pairs that were matched, so the
        caller can draw each skeleton in that member's own activity colour.
        """
        now = time.time() if now is None else now
        poses = [p for p in poses if p is not None][:self.max_crew]

        # Score every track/detection pair, then take the best matches first.
        # Greedy nearest-neighbour is enough here: crew move slowly relative to
        # the frame rate, so the closest match is the right one.
        candidates = []
        for track in self._tracks.values():
            for index, pose in enumerate(poses):
                distance = track.distance_to(pose)
                if distance <= config.TRACK_MATCH_RADIUS:
                    candidates.append((distance, track.id, index))
        candidates.sort()

        claimed_tracks, claimed_poses = set(), set()
        matched = []
        for _, track_id, index in candidates:
            if track_id in claimed_tracks or index in claimed_poses:
                continue
            claimed_tracks.add(track_id)
            claimed_poses.add(index)
            track = self._tracks[track_id]
            track.observe(poses[index], now)
            matched.append((track, poses[index]))

        # Unmatched detections are crew members we have not seen before.
        for index, pose in enumerate(poses):
            if index in claimed_poses:
                continue
            if len(self._tracks) >= self.max_crew:
                break
            track = CrewTrack(self._next_id, self._free_slot(), pose, now)
            self._next_id += 1
            self._tracks[track.id] = track
            track.observe(pose, now)
            matched.append((track, pose))

        # Tracks with no detection this frame: age them, and retire the ones
        # that have been gone long enough to be genuinely absent.
        for track in list(self._tracks.values()):
            if track.id in claimed_tracks:
                continue
            track.miss(now)
            if now - track.last_seen > config.TRACK_EXPIRY_SECONDS:
                del self._tracks[track.id]

        return matched

    def as_list(self) -> list:
        return [t.as_dict() for t in self.tracks]
