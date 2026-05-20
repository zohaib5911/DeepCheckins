import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

import paths
paths.bootstrap_dirs()

KNOWN_LOG        = paths.KNOWN_LOG_FILE
KNOWN_PICS_DIR   = paths.KNOWN_PICS_DIR
UNKNOWN_LOG      = paths.UNKNOWN_LOG_FILE
UNKNOWN_PICS_DIR = paths.UNKNOWN_DIR
CONFIG_FILE      = os.path.join(os.path.dirname(__file__), "config.json")

app = Flask(__name__)


# ---------- parsing ----------

KNOWN_RE = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2})\]"
    r"-(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:-\[(?P<cam>[^\]]+)\])?"
    r"-(?P<video>[^-]+\.mp4)"
    r"-F:(?P<frame>\d+):\s*(?P<name>.+?)\s+--\s+(?P<score>[\d.]+)\s*$"
)

UNKNOWN_RE = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2})\]"
    r"-(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:-\[(?P<cam>[^\]]+)\])?"
    r"-(?P<video>[^-]+\.mp4)"
    r"-F:(?P<frame>\d+):\s*(?P<who>.+?)\s+---\s+(?P<score>[\d.]+)\s*$"
)


def _video_cam(video_name: str) -> str:
    base = os.path.splitext(os.path.basename(video_name or ""))[0]
    if "_" in base:
        prefix = base.split("_", 1)[0].strip()
        if prefix:
            return prefix
    return "—"


def _known_pic_filename(name: str, video: str, frame: str) -> str:
    base = os.path.splitext(os.path.basename(video))[0]
    return f"{name}_{base}_F{frame}.jpg"


def _parse_known() -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(KNOWN_LOG):
        return out
    with open(KNOWN_LOG, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = KNOWN_RE.match(line)
            if not m:
                continue
            d = m.groupdict()
            try:
                dt = datetime.strptime(f"{d['date']} {d['time']}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            cam = d.get("cam") or _video_cam(d["video"])
            pic = _known_pic_filename(d["name"], d["video"], d["frame"])
            pic_exists = os.path.exists(os.path.join(KNOWN_PICS_DIR, pic))
            out.append(
                {
                    "kind": "known",
                    "dt": dt,
                    "date": dt.date(),
                    "time_str": d["time"],
                    "name": d["name"],
                    "cam": cam,
                    "video": d["video"],
                    "frame": d["frame"],
                    "score": float(d["score"]),
                    "pic_url": url_for("known_pic", filename=pic) if pic_exists else "",
                }
            )
    return out


def _parse_unknown() -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(UNKNOWN_LOG):
        return out
    with open(UNKNOWN_LOG, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = UNKNOWN_RE.match(line)
            if not m:
                continue
            d = m.groupdict()
            try:
                dt = datetime.strptime(f"{d['date']} {d['time']}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            who = d["who"].strip()
            id_match = re.search(r"(\d+)\s*$", who)
            uid = id_match.group(1) if id_match else ""
            cam = d.get("cam") or _video_cam(d["video"])
            pic_url = ""
            if uid:
                pic_path = os.path.join(UNKNOWN_PICS_DIR, f"{uid}.jpg")
                if os.path.exists(pic_path):
                    pic_url = url_for("unknown_pic", filename=f"{uid}.jpg")
            label = f"Unknown #{uid}" if uid else "Unknown"
            out.append(
                {
                    "kind": "unknown",
                    "dt": dt,
                    "date": dt.date(),
                    "time_str": d["time"],
                    "name": label,
                    "cam": cam,
                    "video": d["video"],
                    "frame": d["frame"],
                    "score": float(d["score"]),
                    "pic_url": pic_url,
                }
            )
    return out


# ---------- system start date ----------

def _load_start_date() -> date | None:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    raw = (cfg.get("system_starting_date") or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


# ---------- summary helpers ----------

def _period_range(period: str, anchor: date) -> tuple[date, date, str]:
    if period == "daily":
        return anchor, anchor, anchor.strftime("%A, %b %d, %Y")
    if period == "monthly":
        start = anchor.replace(day=1)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1) - timedelta(days=1)
        else:
            end = start.replace(month=start.month + 1) - timedelta(days=1)
        label = start.strftime("%B %Y")
        return start, end, label
    if period == "yearly":
        start = date(anchor.year, 1, 1)
        end = date(anchor.year, 12, 31)
        return start, end, f"{anchor.year}"
    # weekly (default): ISO week (Mon..Sun) containing anchor
    start = anchor - timedelta(days=anchor.weekday())
    end = start + timedelta(days=6)
    label = f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"
    return start, end, label


def _build_summary(known: list[dict], period: str, anchor: date) -> dict:
    start, end, label = _period_range(period, anchor)
    today = date.today()
    system_start = _load_start_date()
    # Clip elapsed-days window to [max(period_start, system_start) .. min(period_end, today)]
    effective_start = max(start, system_start) if system_start else start
    effective_end = min(end, today)
    if effective_end >= effective_start:
        elapsed_days = (effective_end - effective_start).days + 1
    else:
        elapsed_days = 0
    total_days = (end - start).days + 1

    # name -> list of entries in window
    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in known:
        if start <= e["date"] <= end:
            grouped[e["name"]].append(e)

    # Ensure every known name (across whole log) is shown even if 0 entries
    all_names = sorted({e["name"] for e in known})

    rows = []
    for name in all_names:
        entries = grouped.get(name, [])
        per_day: dict[date, list[datetime]] = defaultdict(list)
        for e in entries:
            per_day[e["date"]].append(e["dt"])
        days_present = len(per_day)
        absents = max(elapsed_days - days_present, 0)
        hours = 0.0
        for times in per_day.values():
            if len(times) >= 1:
                span = (max(times) - min(times)).total_seconds() / 3600.0
                hours += span
        rows.append(
            {
                "name": name,
                "entries": len(entries),
                "absents": absents,
                "hours": round(hours, 2),
            }
        )

    # Sort: more entries first, then name
    rows.sort(key=lambda r: (-r["entries"], r["name"]))
    return {
        "rows": rows,
        "label": label,
        "total_days": total_days,
        "elapsed_days": elapsed_days,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


# ---------- routes ----------

@app.route("/")
def index():
    return redirect(url_for("summary"))


@app.route("/summary")
def summary():
    period = request.args.get("period", "daily")
    if period not in ("daily", "weekly", "monthly", "yearly"):
        period = "daily"
    known = _parse_known()
    # Anchor: today, but if today is before first log date, fall back to latest log date
    today = date.today()
    if known:
        latest = max(e["date"] for e in known)
        anchor = today if today >= min(e["date"] for e in known) else latest
    else:
        anchor = today
    data = _build_summary(known, period, anchor)
    unknown = _parse_unknown()
    start_d, end_d = date.fromisoformat(data["start"]), date.fromisoformat(data["end"])
    known_in_period = {e["name"] for e in known if start_d <= e["date"] <= end_d}
    unknown_in_period = {e["name"] for e in unknown if start_d <= e["date"] <= end_d}
    return render_template(
        "summary.html",
        period=period,
        rows=data["rows"],
        period_label=data["label"],
        total_days=data["total_days"],
        elapsed_days=data["elapsed_days"],
        unique_known=len(known_in_period),
        unique_unknown=len(unknown_in_period),
        active_page="summary",
    )


def _activity_rows() -> list[dict]:
    all_rows = _parse_known() + _parse_unknown()
    all_rows.sort(key=lambda r: r["dt"], reverse=True)
    return [
        {
            "dt_str": r["dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "name": r["name"],
            "cam": r["cam"],
            "pic_url": r["pic_url"],
            "kind": r["kind"],
        }
        for r in all_rows
    ]


@app.route("/entries")
def entries():
    return render_template("entries.html", rows=_activity_rows(), active_page="entries")


@app.route("/api/entries")
def api_entries():
    return jsonify({"rows": _activity_rows()})


@app.route("/img/known/<path:filename>")
def known_pic(filename: str):
    safe_root = os.path.realpath(KNOWN_PICS_DIR)
    target = os.path.realpath(os.path.join(safe_root, filename))
    if not target.startswith(safe_root + os.sep):
        abort(404)
    if not os.path.exists(target):
        abort(404)
    return send_from_directory(safe_root, filename)


@app.route("/img/unknown/<path:filename>")
def unknown_pic(filename: str):
    safe_root = os.path.realpath(UNKNOWN_PICS_DIR)
    target = os.path.realpath(os.path.join(safe_root, filename))
    if not target.startswith(safe_root + os.sep):
        abort(404)
    if not os.path.exists(target):
        abort(404)
    return send_from_directory(safe_root, filename)


if __name__ == "__main__":
    app.run(host=paths.APP_HOST, port=paths.APP_PORT, debug=False)
