import json
import os
import re
import shutil
from datetime import datetime


def _extract_datetime_from_video_name(video_name: str) -> tuple[str | None, str | None]:
    if not video_name:
        return None, None
    base = os.path.splitext(os.path.basename(video_name))[0]

    fr_match = re.match(
        r"^fr_(\d+s?|\d+)_([0-3]?\d)_([0-1]?\d)_(\d{4})_([0-2]?\d)_([0-5]?\d)_([0-5]?\d)$",
        base,
    )
    if fr_match:
        day = int(fr_match.group(2))
        month = int(fr_match.group(3))
        year = int(fr_match.group(4))
        hour = int(fr_match.group(5))
        minute = int(fr_match.group(6))
        second = int(fr_match.group(7))
        return f"{year:04d}-{month:02d}-{day:02d}", f"{hour:02d}:{minute:02d}:{second:02d}"

    motion_match = re.match(r"^[A-Za-z0-9_-]+_(\d{8})_(\d{6})_(\d+)s$", base)
    if motion_match:
        date_raw = motion_match.group(1)
        time_raw = motion_match.group(2)
        date_part = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        time_part = f"{time_raw[0:2]}:{time_raw[2:4]}:{time_raw[4:6]}"
        return date_part, time_part

    return None, None


def _extract_datetime_from_timestamp(timestamp: str) -> tuple[str | None, str | None]:
    if not timestamp:
        return None, None
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except Exception:
        return None, None


def _extract_camera_type(video_name: str) -> str | None:
    if not video_name:
        return None
    base = os.path.splitext(os.path.basename(video_name))[0]
    for sep in ("-", "_"):
        if sep in base:
            prefix = base.split(sep, 1)[0].strip()
            if prefix:
                return prefix
    return None


def _safe_filename(video_name: str, frame_no: int) -> str:
    base = os.path.splitext(os.path.basename(video_name))[0] if video_name else "unknown"
    base = base.replace(" ", "_")
    return f"{base}_F{frame_no}.jpg"


def _load_paths() -> tuple[str, str]:
    """Returns (known_detect_dir, known_pics_dir) from config.json."""
    try:
        import paths as _p
        return _p.KNOWN_DETECT_DIR, _p.KNOWN_PICS_DIR
    except Exception:
        # Last-resort fallback only if paths.py couldn't load — should never
        # happen in production since entry-points call bootstrap_dirs() first.
        try:
            with open(
                os.path.join(os.path.dirname(__file__), "config.json"),
                "r",
                encoding="utf-8",
            ) as f:
                config = json.load(f)
            return (
                config.get("known_detect_dir", ""),
                config.get("known_pics_dir", ""),
            )
        except Exception:
            return "", ""


def _maybe_copy_log_image(video_name: str, frame_no: int, message: str) -> None:
    if not video_name or frame_no <= 0 or not message:
        return

    user = message.split(" -- ", 1)[0].strip()
    if not user or user.lower().startswith("new unknown"):
        return

    known_detect_dir, images_dir = _load_paths()
    filename = _safe_filename(video_name, frame_no)
    src = os.path.join(known_detect_dir, user, filename)
    if not os.path.exists(src):
        return

    os.makedirs(images_dir, exist_ok=True)
    dst = os.path.join(images_dir, f"{user}_{filename}")
    if not os.path.exists(dst):
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def write_log(file_path: str, timestamp: str, frame_no: int, message: str, video_name: str = ""):
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    date_only, time_only = _extract_datetime_from_video_name(video_name)
    if not date_only or not time_only:
        ts_date, ts_time = _extract_datetime_from_timestamp(timestamp)
        date_only = date_only or ts_date or (timestamp or "")
        time_only = time_only or ts_time

    time_segment = f"-{time_only}" if time_only else ""
    camera_type = _extract_camera_type(video_name)
    type_segment = f"-[{camera_type}]" if camera_type else ""

    with open(file_path, "a", encoding="utf-8") as f:
        if video_name:
            f.write(f"[{date_only}]{time_segment}{type_segment}-{video_name}-F:{frame_no}: {message}\n")
        else:
            f.write(f"[{date_only}]{time_segment}{type_segment}-F:{frame_no}: {message}\n")

    _maybe_copy_log_image(video_name, frame_no, message)