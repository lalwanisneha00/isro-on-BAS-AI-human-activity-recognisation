"""The mission log: segments, noise rejection, and what reaches disk."""

import csv
import shutil
import sqlite3
import tempfile
from pathlib import Path

from _harness import Results
from app import config
from app.logbook import CSV_COLUMNS, ActivityLogger

SHIFT = [
    ("Idle", 0.80, 6.0),
    ("Exercise", 0.92, 12.0),
    ("Idle", 0.75, 0.4),            # too brief - must not be logged
    ("Experiment Operation", 0.88, 20.0),
    ("Eating/Rest", 0.71, 8.0),
    ("Anomaly (No Motion)", 0.95, 10.0),
    ("No Crew Detected", 0.00, 5.0),
]


def run() -> Results:
    r = Results("mission log")
    scratch = Path(tempfile.mkdtemp(prefix="bas_test_log_"))

    try:
        log = ActivityLogger(log_dir=scratch)
        r.check(log.storage_ok, "log storage opens", log.storage_detail)

        t = 10_000.0
        for label, confidence, held in SHIFT:
            steps = max(1, int(held * config.TARGET_FPS))
            for i in range(steps):
                # A transient system state mid-activity must not split it.
                if label == "Exercise" and i == steps // 2:
                    log.observe("CM-1", "Acquiring", 0.0, now=t)
                log.observe("CM-1", label, confidence, now=t)
                t += 1.0 / config.TARGET_FPS
        log.flush(now=t)

        events = log.recent()
        activities = [e["activity"] for e in events]

        r.check(len(events) == 6, "one row per real activity",
                f"got {len(events)}: {activities}")
        r.check(all(e["duration"] >= config.MIN_LOG_SECONDS for e in events),
                "the 0.4s blip was discarded as noise")
        exercise = [e for e in events if e["activity"] == "Exercise"]
        r.check(len(exercise) == 1, "Exercise is one segment, not two",
                f"got {len(exercise)}")
        r.check(exercise and exercise[0]["duration"] > 11.0,
                "Exercise survived the mid-activity blip",
                f"{exercise[0]['duration'] if exercise else 0}s")
        r.check(all(0.0 <= e["confidence"] <= 1.0 for e in events),
                "confidences stay within 0..1")
        r.check(all(e["crew"] == "CM-1" for e in events),
                "every row names the crew member")

        # ---- what actually reached the disk --------------------------------
        with open(log.csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        r.check(rows[0] == CSV_COLUMNS, "CSV header matches the schema",
                str(rows[0]))
        r.check(len(rows) - 1 == len(events), "CSV holds every logged segment",
                f"{len(rows) - 1} vs {len(events)}")

        connection = sqlite3.connect(log.db_path)
        db_rows = connection.execute(
            "SELECT crew, activity, duration_seconds FROM activity_events").fetchall()
        connection.close()
        r.check(len(db_rows) == len(events), "SQLite agrees with the CSV",
                f"{len(db_rows)} vs {len(events)}")

        totals = log.totals()
        r.check(abs(totals.get("Exercise", 0) - 12.0) < 0.6,
                "cumulative totals match the time actually spent",
                f"Exercise {totals.get('Exercise')}s")
        log.close()

        # ---- two crew members are logged independently ---------------------
        log = ActivityLogger(log_dir=scratch)
        t = 20_000.0
        for i in range(int(8 * config.TARGET_FPS)):
            log.observe("CM-1", "Exercise", 0.9, now=t)
            log.observe("CM-2", "Idle", 0.8, now=t)
            t += 1.0 / config.TARGET_FPS
        log.flush(now=t)
        crews = {e["crew"]: e["activity"] for e in log.recent()}
        r.check(crews.get("CM-1") == "Exercise" and crews.get("CM-2") == "Idle",
                "two crew members log separate activities", str(crews))
        totals = log.totals()
        r.check(abs(totals.get("Exercise", 0) - 8.0) < 0.6
                and abs(totals.get("Idle", 0) - 8.0) < 0.6,
                "each member contributes their own time to the totals",
                str(totals))
        log.close()

        # ---- an unwritable log must not take the system down ---------------
        broken = ActivityLogger(log_dir=scratch / "sub")
        broken.storage_ok = False
        broken.observe("CM-1", "Idle", 0.8, now=30_000.0)
        broken.observe("CM-1", "Exercise", 0.9, now=30_010.0)
        r.check(len(broken.recent()) == 1,
                "logging continues in memory when the disk is unavailable")
        broken.close()

    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return r


if __name__ == "__main__":
    run().report()
