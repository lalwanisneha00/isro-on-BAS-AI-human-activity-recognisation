"""Activity logging: one row per completed activity segment.

The classifier reports a label every frame, but a mission log wants episodes,
not frames. This module watches the label stream, and the moment the activity
changes it closes off the previous segment and writes a single row recording
what happened, when, for how long, and how confident the system was.

Rows go to both a CSV (easy to open in Excel for the demo) and a SQLite file
(queryable, and what the dashboard reads), plus a short in-memory history so
the UI never has to touch the disk to redraw.

Disk problems are contained: if a write fails the run continues on the
in-memory history, because losing the log must never take down the monitor.
"""

import csv
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from . import config

LOG_ROOT = Path(__file__).resolve().parent.parent / config.LOG_DIR

CSV_COLUMNS = ["timestamp", "crew", "activity", "duration_seconds",
               "confidence", "ended_at", "samples"]


class ActivityEvent:
    """One completed activity segment."""

    __slots__ = ("started_at", "ended_at", "crew", "activity", "duration",
                 "confidence", "samples")

    def __init__(self, started_at, ended_at, crew, activity, confidence, samples):
        self.started_at = started_at
        self.ended_at = ended_at
        self.crew = crew
        self.activity = activity
        self.duration = max(0.0, ended_at - started_at)
        self.confidence = confidence
        self.samples = samples

    def as_dict(self) -> dict:
        return {
            "timestamp": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
            "ended_at": datetime.fromtimestamp(self.ended_at).isoformat(timespec="seconds"),
            "clock": datetime.fromtimestamp(self.started_at).strftime("%H:%M:%S"),
            "crew": self.crew,
            "start_epoch": round(self.started_at, 2),
            "end_epoch": round(self.ended_at, 2),
            "activity": self.activity,
            "duration": round(self.duration, 1),
            "confidence": round(self.confidence, 3),
            "samples": self.samples,
        }

    def as_row(self) -> list:
        d = self.as_dict()
        return [d["timestamp"], d["crew"], d["activity"], d["duration"],
                d["confidence"], d["ended_at"], d["samples"]]


class ActivityLogger:
    """Turns the per-frame label stream into logged activity segments."""

    def __init__(self, log_dir: Path = LOG_ROOT):
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / config.LOG_CSV_NAME
        self.db_path = self.log_dir / config.LOG_DB_NAME

        self._lock = threading.Lock()
        self._history = deque(maxlen=config.HISTORY_LIMIT)
        self._totals = {}
        self._connection = None
        self.storage_ok = True
        self.storage_detail = "Log ready"
        self.session_started = time.time()

        # One open segment per crew member, keyed by crew name. Each member
        # accumulates their own activity independently.
        self._open = {}

        self._prepare_storage()

    # -------------------------------------------------------------- storage --
    def _prepare_storage(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            if self.csv_path.exists():
                with open(self.csv_path, "r", newline="", encoding="utf-8") as fh:
                    header = next(csv.reader(fh), [])
                if header != CSV_COLUMNS:
                    # Written by an older build: set it aside rather than
                    # appending rows that would not line up with its columns.
                    self.csv_path.replace(self.csv_path.with_suffix(".csv.previous"))

            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(CSV_COLUMNS)

            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS activity_events (
                       id            INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp     TEXT    NOT NULL,
                       crew          TEXT    NOT NULL,
                       activity      TEXT    NOT NULL,
                       duration_seconds REAL NOT NULL,
                       confidence    REAL    NOT NULL,
                       ended_at      TEXT    NOT NULL,
                       samples       INTEGER NOT NULL
                   )"""
            )
            self._connection.commit()
        except Exception as exc:
            self.storage_ok = False
            self.storage_detail = f"Log storage unavailable: {exc}"

    def _write(self, event: ActivityEvent) -> None:
        """Persist one event. Failure degrades to memory-only logging."""
        if not self.storage_ok:
            return
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(event.as_row())

            if self._connection is not None:
                self._connection.execute(
                    "INSERT INTO activity_events (timestamp, crew, activity, "
                    "duration_seconds, confidence, ended_at, samples) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    event.as_row(),
                )
                self._connection.commit()
        except Exception as exc:
            self.storage_ok = False
            self.storage_detail = f"Log write failed, keeping history in memory: {exc}"

    # ------------------------------------------------------------ observing --
    def observe(self, crew: str, label: str, confidence: float,
                now: float = None) -> None:
        """Feed one frame of one crew member's classifier output."""
        now = time.time() if now is None else now

        with self._lock:
            if label not in config.LOGGED_STATES:
                # Transient system state (window still filling): let the open
                # segment stand rather than chopping it in two.
                return

            current = self._open.get(crew)
            if current is not None and current["label"] == label:
                current["confidence_sum"] += confidence
                current["samples"] += 1
                return

            self._close_segment(crew, now)
            self._open[crew] = {
                "label": label,
                "started": now,
                "confidence_sum": confidence,
                "samples": 1,
            }

    def _close_segment(self, crew: str, now: float) -> None:
        """Finish one crew member's open segment, if it lasted long enough."""
        current = self._open.pop(crew, None)
        if current is None:
            return

        duration = now - current["started"]
        if duration < config.MIN_LOG_SECONDS:
            return                       # too brief to be a real activity

        samples = current["samples"]
        mean_confidence = (current["confidence_sum"] / samples) if samples else 0.0
        event = ActivityEvent(current["started"], now, crew, current["label"],
                              mean_confidence, samples)

        self._history.append(event)
        self._totals[event.activity] = self._totals.get(event.activity, 0.0) + event.duration
        self._write(event)

    def retire(self, crew: str, now: float = None) -> None:
        """Close a crew member's segment because their track ended."""
        now = time.time() if now is None else now
        with self._lock:
            self._close_segment(crew, now)

    def flush(self, now: float = None) -> None:
        """Close every open segment, e.g. when the system is shutting down."""
        now = time.time() if now is None else now
        with self._lock:
            for crew in list(self._open.keys()):
                self._close_segment(crew, now)
            self._open.clear()

    def close(self) -> None:
        self.flush()
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    # -------------------------------------------------------------- reading --
    def recent(self, limit: int = 25) -> list:
        """Most recent completed segments, newest first.

        Segments are ordered by when they began. With several crew members
        closing segments at different moments, insertion order would show
        start times jumping backwards and forwards down the table.
        """
        with self._lock:
            events = list(self._history)
        events.sort(key=lambda e: e.started_at, reverse=True)
        return [e.as_dict() for e in events[:limit]]

    def totals(self, include_open: bool = True, now: float = None) -> dict:
        """Cumulative seconds per activity across the session."""
        now = time.time() if now is None else now
        with self._lock:
            totals = dict(self._totals)
            if include_open:
                # Crew-seconds: two members exercising together contribute two
                # seconds per second, which is what a duty total should mean.
                for current in self._open.values():
                    open_duration = max(0.0, now - current["started"])
                    totals[current["label"]] = (totals.get(current["label"], 0.0)
                                                + open_duration)
        return {name: round(seconds, 1) for name, seconds in totals.items()}

    def timeline(self, window_seconds: float = None, now: float = None) -> dict:
        """Recent segments positioned on a time axis, including the live one.

        The open segment is included so the newest block grows in real time
        rather than only appearing once the activity has already ended.
        """
        window_seconds = window_seconds or config.TIMELINE_WINDOW_SECONDS
        now = time.time() if now is None else now
        start = now - window_seconds

        with self._lock:
            events = [e for e in self._history if e.ended_at > start]
            blocks = [{
                "activity": e.activity,
                "crew": e.crew,
                "start_epoch": max(e.started_at, start),
                "end_epoch": e.ended_at,
                "duration": round(e.duration, 1),
                "open": False,
            } for e in events]

            for crew, current in self._open.items():
                if current["started"] > now:
                    continue
                blocks.append({
                    "activity": current["label"],
                    "crew": crew,
                    "start_epoch": max(current["started"], start),
                    "end_epoch": now,
                    "duration": round(now - current["started"], 1),
                    "open": True,
                })

        for b in blocks:
            b["offset"] = round((b["start_epoch"] - start) / window_seconds, 5)
            b["width"] = round((b["end_epoch"] - b["start_epoch"]) / window_seconds, 5)
            b["clock"] = datetime.fromtimestamp(b["start_epoch"]).strftime("%H:%M:%S")

        return {
            "window_seconds": window_seconds,
            "now": round(now, 2),
            "start": round(start, 2),
            "blocks": [b for b in blocks if b["width"] > 0],
        }

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            count = len(self._history)
            open_segments = [
                {"crew": crew, "activity": c["label"],
                 "seconds": round(now - c["started"], 1)}
                for crew, c in sorted(self._open.items())
            ]
        return {
            "storage_ok": self.storage_ok,
            "detail": self.storage_detail,
            "events_logged": count,
            "open_segments": open_segments,
            "session_seconds": round(time.time() - self.session_started, 1),
            "csv_path": str(self.csv_path),
            "db_path": str(self.db_path),
        }
