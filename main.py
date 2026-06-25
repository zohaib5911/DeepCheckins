from database import Database, user_count , User
from HNSW import FaissIndex
import cv2
from arcFace import recognize , getEmbedding
from face_quality import BestFaceTracker, save_best_face_results
knownFaiss = None
import os 
import json
import re
from datetime import datetime, timedelta
import numpy as np
import shutil 
from pathlib import Path
from PIL import Image
import pillow_heif
from log import write_log
import os
import time

with open("config.json", "r") as f:
    config = json.load(f)

import paths
paths.bootstrap_dirs()

EMB_PER_USER = config["EMB_PER_USER"]
EMB_DIM = config["EMB_DIM"]
IMG_H = config.get("IMG_H", 128)
IMG_W = config.get("IMG_W", 128)
IMG_C = config.get("IMG_C", 3)
IMAGE_BYTES = config.get("IMAGE_BYTES", IMG_H * IMG_W * IMG_C)
base_folder = paths.REGISTRATION_DIR
knownlogpath = paths.KNOWN_LOG_FILE

# ─── Frame rotation ──────────────────────────────────────────────────────────
# Rotate each frame before processing. Degrees are ANTICLOCKWISE
# (counter-clockwise). Cardinal angles (90/180/270) use cv2.rotate so the full
# image is preserved and dimensions swap correctly; other angles fall back to
# a canvas-expanding affine rotation so nothing gets cropped.
APPLY_ROTATION   = bool(config.get("apply_rotation", False))
ROTATION_DEGREES = int(config.get("rotation_degrees", 0))


def rotate_frame_for_processing(frame):
    """Rotate `frame` anticlockwise by ROTATION_DEGREES if rotation is enabled."""
    if not APPLY_ROTATION or frame is None:
        return frame
    deg = ROTATION_DEGREES % 360
    if deg == 0:
        return frame
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Arbitrary angle: expand the canvas so corners aren't clipped.
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)  # positive = anticlockwise
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(frame, M, (new_w, new_h))


# Videos with no detected faces land here (under the configured noface dir).
NOFACE_VIDEOS_DIR = os.path.join(paths.NOFACE_DIR, "videos")


def _write_rotated_video(src_path: str, dst_path: str) -> bool:
    """Re-encode `src_path` into `dst_path` with each frame rotated per config.
    Returns True on success."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rot = rotate_frame_for_processing(frame)
            if writer is None:
                h, w = rot.shape[:2]
                writer = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
            writer.write(rot)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return writer is not None


def _save_noface_video(video_path: str) -> None:
    """Move a no-face video into the nofaces folder. If rotation is enabled the
    saved copy is rotated; otherwise the original file is moved as-is."""
    os.makedirs(NOFACE_VIDEOS_DIR, exist_ok=True)
    dst = os.path.join(NOFACE_VIDEOS_DIR, os.path.basename(video_path))
    try:
        if APPLY_ROTATION and (ROTATION_DEGREES % 360) != 0:
            if _write_rotated_video(video_path, dst):
                if os.path.exists(video_path):
                    os.remove(video_path)   # drop original so it isn't re-processed
                print(f"[NO-FACE] Saved rotated video → {dst}")
            else:
                print(f"[NO-FACE] Could not re-encode {video_path}; moving original")
                shutil.move(video_path, dst)
        else:
            shutil.move(video_path, dst)
            print(f"[NO-FACE] Moved video → {dst}")
    except Exception as e:
        print(f"[ERROR] Failed to save no-face video {video_path}: {e}")

# ─── Coordination with motion.py ─────────────────────────────────────────────
# motion.py writes "RECORDING:<unix_ms>" or "IDLE:<unix_ms>" to MOTION_STATE_FILE.
# A dedicated watcher thread polls the file at WATCHER_POLL_MS and flips a
# threading.Event — so the per-frame is_recording() check on the hot path is
# essentially a single atomic memory read (free). Detection latency from
# motion.py writing "RECORDING" to main.py noticing it is bounded by
# WATCHER_POLL_MS (≤2 ms) plus filesystem write/read overhead (~1–3 ms on
# SSD). The currently-executing frame's recognize() call cannot be preempted
# (ArcFace runs synchronous C code), so the wall-clock time before main.py
# actually returns is detection_latency + (current frame's recognize() time).

import threading as _threading

MOTION_STATE_FILE = paths.MOTION_STATE_FILE

# Tuning knobs (milliseconds).
WATCHER_POLL_MS   = 2        # dedicated watcher thread cadence
STATE_STALE_MS    = 15_000   # treat RECORDING as IDLE if heartbeat older than this
IDLE_POLL_MS      = 100      # outer-loop poll cadence while idle
RECORDING_POLL_MS = 20       # outer-loop poll cadence while paused

_motion_event   = _threading.Event()  # set ⇔ motion.py is recording
_idle_event     = _threading.Event()  # set ⇔ motion.py is idle (inverse of _motion_event)
_idle_event.set()
_watcher_stop   = _threading.Event()
_watcher_thread: _threading.Thread | None = None
_watcher_lock   = _threading.Lock()


def _read_state_file() -> tuple[str, int]:
    """Parse 'STATE:UNIX_MS'. Tolerates legacy bare 'STATE' values."""
    try:
        with open(MOTION_STATE_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except FileNotFoundError:
        return ("IDLE", 0)
    except Exception:
        return ("IDLE", 0)
    if not raw:
        return ("IDLE", 0)
    if ":" in raw:
        state, _, ts_raw = raw.partition(":")
        state = state.strip().upper()
        try:
            ts_ms = int(ts_raw.strip())
        except ValueError:
            ts_ms = 0
    else:
        state = raw.upper()
        ts_ms = 0
    if state not in ("IDLE", "RECORDING"):
        state = "IDLE"
    return (state, ts_ms)


def _current_state() -> str:
    """Read the state file once, applying staleness fallback."""
    state, ts_ms = _read_state_file()
    if state == "RECORDING" and ts_ms > 0:
        age_ms = int(time.time() * 1000) - ts_ms
        if age_ms > STATE_STALE_MS:
            state = "IDLE"
    return state


def _watcher_loop() -> None:
    """High-frequency poll → toggles _motion_event / _idle_event."""
    interval = max(WATCHER_POLL_MS, 1) / 1000.0
    while not _watcher_stop.wait(interval):
        if _current_state() == "RECORDING":
            _idle_event.clear()
            _motion_event.set()
        else:
            _motion_event.clear()
            _idle_event.set()


def start_motion_watcher() -> None:
    """Spawn the watcher thread. Idempotent."""
    global _watcher_thread
    with _watcher_lock:
        if _watcher_thread and _watcher_thread.is_alive():
            return
        # Seed the events from the current on-disk state so we don't race
        # an empty cycle before the watcher's first tick.
        if _current_state() == "RECORDING":
            _idle_event.clear()
            _motion_event.set()
        else:
            _motion_event.clear()
            _idle_event.set()
        _watcher_stop.clear()
        _watcher_thread = _threading.Thread(
            target=_watcher_loop, daemon=True, name="motion-state-watcher"
        )
        _watcher_thread.start()


def stop_motion_watcher() -> None:
    """Stop the watcher (used by tests / clean shutdown)."""
    _watcher_stop.set()


def motion_state() -> str:
    """Convenience: 'RECORDING' or 'IDLE' based on the watcher's view."""
    return "RECORDING" if _motion_event.is_set() else "IDLE"


def is_recording() -> bool:
    """Hot-path-safe check. Single atomic read of a threading.Event."""
    return _motion_event.is_set()


def wait_until_idle(timeout: float | None = None) -> bool:
    """Block until motion.py reports IDLE again.

    Uses ``_idle_event.wait()`` so the calling thread is fully parked by the
    OS — zero CPU while waiting. Returns True if IDLE was reached, False on
    timeout. (Previous implementation polled `_motion_event.is_set()` and
    spun at 100% CPU because Event.wait returns immediately when the event
    is already set — that bug burned a core on the Pi any time motion.py
    was actively recording.)
    """
    return _idle_event.wait(timeout)

db = None

delete = False
deletedpath = paths.DELETED_REGISTRATION_DIR

def _read_image_any(path: str):
    frame = cv2.imread(path)
    if frame is not None:
        return frame
    ext = Path(path).suffix.lower()
    if ext in {".heic", ".heif"}:
        try:
            heif = pillow_heif.read_heif(path)
            img = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
            rgb = np.array(img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[WARN] HEIC decode failed for {path}: {e}")
    return None


def _to_image_bytes(frame: np.ndarray) -> bytes:
    resized = cv2.resize(frame, (IMG_W, IMG_H))
    if resized.shape[2] != IMG_C:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized.tobytes()

def _extract_timestamp_from_filename(filename: str) -> datetime | None:
    match = re.search(r"(\d{8}_\d{6})", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def fixed_pics_regiss(db):
    if not delete:
        os.makedirs(deletedpath, exist_ok=True)
    for user_id in os.listdir(base_folder):
        if user_id == "deleted":
            continue
        valid_data = []
        all_images = []
        user_path = os.path.join(base_folder, user_id)
        
        if not os.path.isdir(user_path):
            continue
        
        print(f"\n[INFO] Processing User: {user_id}")
        
        for img_name in os.listdir(user_path):
            
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".heic", ".heif")):
                continue
            
            img_path = os.path.join(user_path, img_name)
            all_images.append(img_path)
            
            frame = _read_image_any(img_path)
            
            print(f"[DEBUG] Processing {img_name} for {user_id}")
            
            if frame is None:
                print(f"[WARN] Failed to read {img_path}")
                continue
            
            embedding, quality  = getEmbedding(user_id, frame)
            if embedding is None:
                print(f"[FAIL] Registration failed for {img_name} (No face/embedding)")
                continue
    
            if embedding is not None and embedding.shape == (EMB_DIM,):
                img_bytes = _to_image_bytes(frame)
                if len(img_bytes) == IMAGE_BYTES:
                    valid_data.append((quality, embedding, img_bytes))
            else:
                print(f"[FAIL] Registration failed for {img_name}")
        
        print(f"[DEBUG] Collected {len(valid_data)} valid embeddings for {user_id}")
        
        if len(valid_data) >= EMB_PER_USER:
            valid_data.sort(key=lambda x: x[0], reverse=True)
            best_embeddings = [data[1] for data in valid_data[:EMB_PER_USER]]
            best_images = [data[2] for data in valid_data[:EMB_PER_USER]]
            
            embeddings_array = np.stack(best_embeddings, axis=0)
            user = User(user_id, embeddings_array, best_images)
            db.append(user)

            print(f"[SUCCESS] Registered {user_id} using {len(best_embeddings)} best embeddings")
            
            if delete:
                for img_path in all_images:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"[CLEANUP] Deleted {os.path.basename(img_path)}")
            else:
                user_deleted_path = os.path.join(deletedpath, user_id)
                os.makedirs(user_deleted_path, exist_ok=True)                
                for img_path in all_images:
                    if os.path.exists(img_path):
                        img_name = os.path.basename(img_path)
                        dst = os.path.join(user_deleted_path, img_name)
                        shutil.move(img_path, dst)
                        print(f"[CLEANUP] Moved {img_name} → deleted/{user_id}/")            
            if os.path.exists(user_path) and not os.listdir(user_path):
                os.rmdir(user_path)
                print(f"[CLEANUP] Removed empty folder {user_id}")
        else:
            print(f"[ERROR] Not enough valid embeddings found for {user_id} (Found: {len(valid_data)}, Required: {EMB_PER_USER})")



def dynamic_pic_regis(db):
    if not delete:
        os.makedirs(deletedpath, exist_ok=True)
    for user_id in os.listdir(base_folder):
        if user_id == "deleted":
            continue
        valid_data = []
        all_images = []
        user_path = os.path.join(base_folder, user_id)

        if not os.path.isdir(user_path):
            continue
        
        print(f"\n[INFO] Processing User: {user_id}")
        
        for img_name in os.listdir(user_path):
            
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".heic", ".heif")):
                continue
            
            img_path = os.path.join(user_path, img_name)
            all_images.append(img_path)
            
            frame = _read_image_any(img_path)
            
            print(f"[DEBUG] Processing {img_name} for {user_id}")
            
            if frame is None:
                print(f"[WARN] Failed to read {img_path}")
                continue
            
            embedding, quality  = getEmbedding(user_id, frame)
            if embedding is None:
                print(f"[FAIL] Registration failed for {img_name} (No face/embedding)")
                continue
    
            if embedding is not None and embedding.shape == (EMB_DIM,):
                img_bytes = _to_image_bytes(frame)
                if len(img_bytes) == IMAGE_BYTES:
                    valid_data.append((quality, embedding, img_bytes))
            else:
                print(f"[FAIL] Registration failed for {img_name}")
        
        print(f"[DEBUG] Collected {len(valid_data)} valid embeddings for {user_id}")
        
        if len(valid_data) >= 1:
            try:
                valid_data.sort(key=lambda x: x[0], reverse=True)
                best_embeddings = [data[1] for data in valid_data[:EMB_PER_USER]]
                best_images = [data[2] for data in valid_data[:EMB_PER_USER]]
                
                if best_embeddings:
                    embeddings_array = np.stack(best_embeddings, axis=0)
                    user = User(user_id, embeddings_array, best_images)
                    db.append(user)

                    print(f"[SUCCESS] Registered {user_id} using {len(best_embeddings)} embeddings (min required: {EMB_PER_USER})")
                    
                    if delete:
                        for img_path in all_images:
                            if os.path.exists(img_path):
                                os.remove(img_path)
                                print(f"[CLEANUP] Deleted {os.path.basename(img_path)}")
                    else:
                        user_deleted_path = os.path.join(deletedpath, user_id)
                        os.makedirs(user_deleted_path, exist_ok=True)                
                        for img_path in all_images:
                            if os.path.exists(img_path):
                                img_name = os.path.basename(img_path)
                                dst = os.path.join(user_deleted_path, img_name)
                                shutil.move(img_path, dst)
                                print(f"[CLEANUP] Moved {img_name} → deleted/{user_id}/")            
                    if os.path.exists(user_path) and not os.listdir(user_path):
                        os.rmdir(user_path)
                        print(f"[CLEANUP] Removed empty folder {user_id}")
                else:
                    print(f"[ERROR] No valid embeddings to register for {user_id}")
            except Exception as e:
                print(f"[ERROR] Failed to register {user_id}: {e}")
        else:
            print(f"[WARN] No valid embeddings found for {user_id}")





DeleteFlag = False
movePath = paths.PROCESSED_VIDEOS_DIR

def _get_processed_subdir(video_path: str) -> str:
    parent_name = os.path.basename(os.path.dirname(video_path))
    if parent_name and re.match(r"\d{2}-\d{2}-\d{4}$", parent_name):
        return os.path.join(movePath, parent_name)
    filename = os.path.basename(video_path)
    date_match = re.search(r"(\d{2}-\d{2}-\d{4})", filename)
    if date_match:
        return os.path.join(movePath, date_match.group(1))
    return movePath

def video_start(video_path):
    db = Database()
    db.create()
    # fixed_pics_regiss(db)
    knownFaiss = FaissIndex(mode="known")
    knownFaiss.build_faiss(db.load_for_hnsw(), use_exact=True)
    print(f"Initialized with {user_count()} users in the database.")
    name = "Video"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    video_name = os.path.basename(video_path)
    base_ts = _extract_timestamp_from_filename(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    start_ts = base_ts or datetime.now()
    frame_id = 0
    interrupted = False
    any_face_detected = False
    best_tracker = BestFaceTracker(top_n=10)
    # is_recording() is cached (STATE_CACHE_TTL_MS), so calling it every
    # frame is essentially free — gives us ~50ms reaction time when motion
    # starts. Partial work is discarded; the video stays in place and gets
    # re-processed when we go IDLE again.
    while True:
        if is_recording():
            print(f"[INTERRUPT] Motion detected — pausing processing of {video_name}")
            interrupted = True
            break
        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame.")
            break
        frame = rotate_frame_for_processing(frame)
        frame_id += 1
        frame_ts = (start_ts + timedelta(seconds=(frame_id - 1) / fps))
        detected = recognize(
            name,
            frame,
            knownFaiss,
            db,
            frame_id,
            frame_ts.isoformat(timespec="milliseconds"),
            video_name,
            best_tracker=best_tracker,
        )
        any_face_detected = any_face_detected or bool(detected)

        # Progress: every 100 frames report current frame + best score so far.
        if frame_id % 100 == 0:
            print(f"[PROGRESS] {video_name}: frame {frame_id} | "
                  f"best score so far {best_tracker.best_score_overall():.4f}")

    cap.release()

    if interrupted:
        # Leave the video where it is; it'll be picked up again when IDLE.
        return

    print("Processing finished.")

    # No face anywhere in the video → don't treat it as processed. Save the
    # (rotated) video into the nofaces folder and stop here.
    if not any_face_detected:
        print(f"[NO-FACE] No faces detected in {video_name} — moving to nofaces folder")
        _save_noface_video(video_path)
        return

    # Save best-face results for every identified person in this video.
    for uid in best_tracker.users():
        best_score, best_frame = best_tracker.best(uid)
        save_best_face_results(
            uid,
            video_path,
            best_score,
            best_frame,
            best_tracker.top_candidates(uid),
        )

    # Move or delete processed video
    try:
        if DeleteFlag:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted {os.path.basename(video_path)}")
        else:
            if os.path.exists(video_path):
                target_dir = _get_processed_subdir(video_path)
                os.makedirs(target_dir, exist_ok=True)
                dst = os.path.join(target_dir, os.path.basename(video_path))
                shutil.move(video_path, dst)
                print(f"[CLEANUP] Moved {os.path.basename(video_path)} → {target_dir}")
    except Exception as e:
        print(f"[ERROR] Cleanup failed for {video_path}: {e}")



def cam_start():
    db = Database()
    Database().create()
    knownFaiss = FaissIndex(mode="known")
    knownFaiss.build_faiss(db.load_for_hnsw(), use_exact=True)

    print(f"Initialized with {user_count()} users in the database.")
    name = "Webcam"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            continue
        frame_id += 1
        recognize(
            name,
            frame,
            knownFaiss,
            db,
            frame_id,
            datetime.now().isoformat(timespec="milliseconds"),
            "webcam",
        )






VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _list_ready_videos(folder: str) -> list[str]:
    """List finished video files in `folder` (skips motion.py's *.part)."""
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        return []
    out = []
    for v in entries:
        # Skip in-flight motion.py writes; they end with `.mp4.part`.
        if v.endswith(".part"):
            continue
        full = os.path.join(folder, v)
        if not os.path.isfile(full):
            continue
        if not v.lower().endswith(VIDEO_EXTS):
            continue
        out.append(v)
    return sorted(out)


if __name__ == "__main__":
    pt = paths.VIDEOS_DIR.rstrip("/") + "/"
    print(f"[INFO] Watching folder: {pt}")
    print(f"[INFO] Motion state file: {MOTION_STATE_FILE}")
    # Spawn the 2ms watcher so per-frame is_recording() is a free atomic read.
    start_motion_watcher()

    idle_delay = max(IDLE_POLL_MS, 10) / 1000.0
    while True:
        # While motion.py is recording, pause processing entirely and poll
        # at ms cadence so we resume the instant it goes IDLE.
        if is_recording():
            wait_until_idle()
            continue

        videos = _list_ready_videos(pt)
        if not videos:
            time.sleep(idle_delay)
            continue

        for video in videos:
            # Re-check between videos so we yield immediately when motion starts.
            if is_recording():
                break
            video_path = os.path.join(pt, video)
            print(f"\n[INFO] New video detected: {video}")
            try:
                video_start(video_path)
            except Exception as e:
                print(f"[ERROR] video_start failed for {video}: {e}")
        time.sleep(idle_delay)
