"""Downloadable activity logs.

The console is a live view, but the deliverable a ground team actually wants
is the record afterwards. Three formats, all generated locally with nothing
leaving the machine:

* CSV      - opens straight into Excel, one row per activity segment.
* JSON     - the same data, for anything that has to read it programmatically.
* Report   - a plain-text mission-day summary: how long went on what, how
             many times the crew member changed task, and every anomaly.

The session's own history lives in memory; the full history lives in the
SQLite file the logger has been writing all along. "Current session" reads
the first, "all history" reads the second, so asking for everything really
does return everything, including previous runs.
"""

import csv
import io
import json
import sqlite3
import time
from datetime import datetime

from . import config


def _stamp() -> str:
    """A filename stamp, so repeated downloads never overwrite each other."""
    return time.strftime("%Y%m%d_%H%M%S")


def filename(kind: str, scope: str) -> str:
    extension = {"csv": "csv", "json": "json", "report": "txt"}[kind]
    return f"bas_activity_{scope}_{_stamp()}.{extension}"


def collect(logger, scope: str = "session") -> list:
    """Activity segments for the requested scope, oldest first."""
    if scope == "all":
        rows = _from_database(logger)
        if rows:
            return rows
        # No database (or an empty one): the session's own history is still
        # the truthful answer rather than an error.
    return list(reversed(logger.recent(limit=100_000)))


def _from_database(logger) -> list:
    """Every segment ever written, including previous runs."""
    if not logger.db_path.exists():
        return []
    try:
        connection = sqlite3.connect(f"file:{logger.db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT timestamp, crew, activity, duration_seconds, confidence, "
            "ended_at, samples FROM activity_events ORDER BY id"
        ).fetchall()
        connection.close()
    except Exception:
        # A locked or damaged database must not break the download; the
        # in-memory history is still worth handing over.
        return []

    collected = []
    for row in rows:
        started = row["timestamp"]
        collected.append({
            "timestamp": started,
            "ended_at": row["ended_at"],
            "clock": started[11:19] if len(started) >= 19 else started,
            "crew": row["crew"],
            "activity": row["activity"],
            "duration": round(float(row["duration_seconds"]), 1),
            "confidence": round(float(row["confidence"]), 3),
            "samples": int(row["samples"]),
        })
    return collected


# ------------------------------------------------------------------- CSV ---
CSV_HEADERS = [
    ("timestamp", "Start Time"),
    ("ended_at", "End Time"),
    ("crew", "Crew Member"),
    ("activity", "Activity"),
    ("duration", "Duration (s)"),
    ("confidence", "Confidence"),
    ("samples", "Frames Analysed"),
]


def to_csv(rows: list) -> str:
    """One row per segment, with headers a person can read."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([title for _, title in CSV_HEADERS])
    for row in rows:
        writer.writerow([row.get(key, "") for key, _ in CSV_HEADERS])
    return buffer.getvalue()


# ------------------------------------------------------------------ JSON ---
def to_json(rows: list, scope: str) -> str:
    return json.dumps({
        "mission": config.MISSION_NAME,
        "module": config.MODULE_NAME,
        "scope": scope,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "segment_count": len(rows),
        "totals_seconds": summarise(rows)["totals"],
        "segments": rows,
    }, indent=2)


# ---------------------------------------------------------------- report ---
def summarise(rows: list) -> dict:
    """Totals, transitions and anomalies - the things a report is made of."""
    totals, anomalies = {}, []
    for row in rows:
        activity = row.get("activity", "Unknown")
        totals[activity] = round(totals.get(activity, 0.0)
                                 + float(row.get("duration", 0.0)), 1)
        if activity == config.ACT_ANOMALY:
            anomalies.append(row)

    monitored = round(sum(totals.values()), 1)
    return {
        "totals": totals,
        "monitored_seconds": monitored,
        # Every segment after the first is one change of task.
        "transitions": max(0, len(rows) - 1),
        "anomalies": anomalies,
    }


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def to_report(rows: list, scope: str) -> str:
    """A plain-text mission-day summary."""
    summary = summarise(rows)
    monitored = summary["monitored_seconds"]

    lines = [
        "=" * 66,
        f"  {config.MISSION_NAME}",
        f"  {config.MODULE_NAME} - CREW ACTIVITY REPORT",
        "=" * 66,
        "",
        f"  Generated   {datetime.now().strftime('%d %b %Y, %H:%M:%S')}",
        f"  Coverage    {'Full history' if scope == 'all' else 'Current session'}",
        f"  Monitored   {_duration(monitored)}",
        f"  Segments    {len(rows)}",
        f"  Task changes{summary['transitions']:>4}",
        "",
    ]

    if not rows:
        lines += ["  No activity was recorded.", ""]
        return "\n".join(lines)

    lines += ["  TIME PER ACTIVITY", "  " + "-" * 62]
    ordered = sorted(summary["totals"].items(), key=lambda kv: -kv[1])
    for activity, seconds in ordered:
        share = (seconds / monitored * 100) if monitored else 0.0
        bar = "#" * int(round(share / 3.0))
        lines.append(f"  {activity:26} {_duration(seconds):>10} "
                     f"{share:5.1f}%  {bar}")
    lines.append("")

    anomalies = summary["anomalies"]
    lines += ["  ANOMALIES", "  " + "-" * 62]
    if anomalies:
        for row in anomalies:
            lines.append(f"  {row['clock']}  {row['crew']}  "
                         f"no motion for {_duration(row['duration'])}")
        lines.append("")
        lines.append(f"  {len(anomalies)} anomaly period(s) flagged for review.")
    else:
        lines.append("  None. Crew motion was detected throughout.")
    lines.append("")

    lines += ["  ACTIVITY LOG", "  " + "-" * 62,
              f"  {'Start':<10}{'Activity':<28}{'Duration':>10}{'Conf.':>8}"]
    for row in rows:
        lines.append(f"  {row['clock']:<10}{row['activity']:<28}"
                     f"{_duration(row['duration']):>10}"
                     f"{row['confidence'] * 100:>7.0f}%")

    lines += ["", "=" * 66,
              "  Activities are recognised from body pose and the objects in"
              " the crew",
              "  member's hands. Confidence is the system's own certainty,"
              " averaged",
              "  across each segment. Generated offline; no data left this"
              " machine.",
              "=" * 66, ""]
    return "\n".join(lines)


def build(logger, kind: str = "csv", scope: str = "session"):
    """Return (filename, media type, body) for a download."""
    kind = kind if kind in ("csv", "json", "report") else "csv"
    scope = "all" if scope == "all" else "session"
    rows = collect(logger, scope)

    if kind == "json":
        return filename(kind, scope), "application/json", to_json(rows, scope)
    if kind == "report":
        return filename(kind, scope), "text/plain; charset=utf-8", \
            to_report(rows, scope)
    return filename(kind, scope), "text/csv; charset=utf-8", to_csv(rows)
