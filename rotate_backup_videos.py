"""
One-off utility: rotate every video in the processed folder 90° anticlockwise
and write the rotated copy (same filename) into the backup folder.

Once this finishes you can flip `apply_rotation` to false in config.json — the
backed-up videos are already upright, so no runtime rotation is needed.

Usage:
    python rotate_backup_videos.py                 # use the defaults below
    python rotate_backup_videos.py <src> <dst>     # override folders
    python rotate_backup_videos.py <src> <dst> 270 # override rotation degrees

Notes:
  * The processed folder usually nests videos in date subfolders
    (processed/DD-MM-YYYY/...). This script mirrors that structure under the
    destination so same-named files from different dates don't overwrite.
  * Rotation degrees are ANTICLOCKWISE, matching the main pipeline.
"""

import os
import sys

import cv2

# ─── Defaults ────────────────────────────────────────────────────────────────
SRC_DIR = "/home/zohaib/deeepScanIN/data/videos/processed/"
DST_DIR = "/home/zohaib/deeepScanIN/data/backupVideos/"
ROTATION_DEGREES = 90  # anticlockwise

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def rotate_frame(frame, deg):
    """Rotate a frame anticlockwise by `deg`. Cardinal angles preserve the full
    image; arbitrary angles expand the canvas so nothing is clipped."""
    deg %= 360
    if deg == 0:
        return frame
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)  # positive = anticlockwise
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(frame, M, (new_w, new_h))


def rotate_one(src_path, dst_path, deg):
    """Re-encode src_path -> dst_path with every frame rotated. Returns True on
    success."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  [SKIP] Could not open {src_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rot = rotate_frame(frame, deg)
            if writer is None:
                h, w = rot.shape[:2]
                writer = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    print(f"  [FAIL] Could not open writer for {dst_path}")
                    return False
            writer.write(rot)
            frames += 1
            if total and frames % 200 == 0:
                print(f"    {frames}/{total} frames", end="\r")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames == 0:
        print(f"  [WARN] No frames read from {src_path}")
        return False
    print(f"  [OK] {frames} frames -> {dst_path}")
    return True


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else SRC_DIR
    dst_dir = sys.argv[2] if len(sys.argv) > 2 else DST_DIR
    deg = int(sys.argv[3]) if len(sys.argv) > 3 else ROTATION_DEGREES

    if not os.path.isdir(src_dir):
        print(f"[ERROR] Source folder does not exist: {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)
    print(f"[INFO] Source : {src_dir}")
    print(f"[INFO] Dest   : {dst_dir}")
    print(f"[INFO] Rotate : {deg}° anticlockwise\n")

    ok = skipped = failed = 0
    for root, _dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(VIDEO_EXTS):
                continue
            src_path = os.path.join(root, fname)
            # Mirror the subfolder structure under dst_dir.
            rel = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(dst_dir, rel)

            if os.path.exists(dst_path):
                print(f"[SKIP] Already exists: {rel}")
                skipped += 1
                continue

            print(f"[PROCESS] {rel}")
            if rotate_one(src_path, dst_path, deg):
                ok += 1
            else:
                failed += 1

    print(f"\n[DONE] rotated={ok}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
