import argparse
import os
import re
import shutil
from pathlib import Path

import paths

LOG_PATTERN = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2})\]-\d{2}:\d{2}:\d{2}(?:-\[(?P<type>[^\]]+)\])?-(?P<video>.+)-F:(?P<frame>\d+):\s+(?P<user>.+?)\s+--"
)


def _safe_filename(video_name: str, frame_no: str) -> str:
    safe_video = Path(video_name).stem if video_name else "unknown"
    safe_video = safe_video.replace(" ", "_")
    return f"{safe_video}_F{frame_no}.jpg"


def export_images(log_path: str, known_dir: str, output_dir: str) -> tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    missing = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = LOG_PATTERN.match(line)
            if not match:
                continue

            video_name = match.group("video")
            frame_no = match.group("frame")
            user = match.group("user").strip()

            filename = _safe_filename(video_name, frame_no)
            src = os.path.join(known_dir, user, filename)
            if not os.path.exists(src):
                missing += 1
                continue

            dst = os.path.join(output_dir, f"{user}_{filename}")
            shutil.copy2(src, dst)
            copied += 1

    return copied, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export images for checkins log entries from data/detections/known into an images folder."
    )
    parser.add_argument(
        "--log",
        default=paths.KNOWN_LOG_FILE,
        help="Path to the known-checkin log file",
    )
    parser.add_argument(
        "--known-dir",
        default=paths.KNOWN_DETECT_DIR,
        help="Path to the known detections folder",
    )
    parser.add_argument(
        "--out",
        default=paths.KNOWN_PICS_DIR,
        help="Output folder to write images",
    )

    args = parser.parse_args()
    copied, missing = export_images(args.log, args.known_dir, args.out)
    print(f"Copied: {copied}")
    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()
