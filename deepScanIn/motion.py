#!/usr/bin/env python3
"""
Motion Detection Camera System
================================
Detects motion → records exactly 10 seconds → saves as timestamped MP4.
Production-grade: robust error handling, auto-recovery, storage management.
"""

import os
import sys
import atexit
import cv2
import time
import signal
import logging
import argparse
import threading
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, List
import json


# ─────────────────────── PROCESS-STATE SIGNALLING ─────────────────────────────
# A tiny file-based IPC: motion.py writes "RECORDING:<unix_ms>" while a clip is
# being captured, "IDLE:<unix_ms>" the rest of the time. main.py polls this
# file and only processes videos while we are IDLE.
#
# The trailing unix-millisecond heartbeat lets main.py detect a crashed
# motion.py: if state == RECORDING but the heartbeat is older than
# STATE_STALE_MS, the consumer should treat the system as IDLE rather than
# blocking forever. While a clip is being captured, a background thread
# refreshes the heartbeat every HEARTBEAT_INTERVAL_MS so the timestamp stays
# fresh under multi-second recordings.

_STATE_LOCK = threading.Lock()
_STATE_FILE_PATH: Optional[str] = None
_RECORDING_COUNT = 0  # supports multi-camera (each clip increments/decrements)
_HEARTBEAT_STOP: Optional[threading.Event] = None
_HEARTBEAT_THREAD: Optional[threading.Thread] = None

HEARTBEAT_INTERVAL_MS = 500   # refresh cadence while recording


def _now_ms() -> int:
    return int(time.time() * 1000)


def _write_state(state: str) -> None:
    if not _STATE_FILE_PATH:
        return
    payload = f"{state}:{_now_ms()}"
    try:
        parent = os.path.dirname(_STATE_FILE_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{_STATE_FILE_PATH}.tmp"
        # Atomic write: write to .tmp then rename, so a reader never sees
        # a half-written state line. We intentionally do NOT fsync — this is
        # a transient signalling file, not durable state. Skipping fsync
        # drops the round-trip from ~20ms to <1ms, which is what lets
        # main.py react in single-digit ms.
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, _STATE_FILE_PATH)
    except Exception:
        # Never let a transient FS error kill the recording loop.
        pass


def _heartbeat_loop(stop_event: threading.Event) -> None:
    """While recording, refresh the timestamp so consumers know we're alive."""
    interval = max(0.05, HEARTBEAT_INTERVAL_MS / 1000.0)
    while not stop_event.wait(interval):
        with _STATE_LOCK:
            if _RECORDING_COUNT > 0:
                _write_state("RECORDING")
            else:
                # Race: counter dropped to zero just as we woke up. Stop.
                return


def init_state_file(path: str) -> None:
    global _STATE_FILE_PATH
    _STATE_FILE_PATH = path
    _write_state("IDLE")
    atexit.register(lambda: _write_state("IDLE"))


def record_start_signal() -> None:
    """Called when a clip begins. Idempotent across concurrent cameras."""
    global _RECORDING_COUNT, _HEARTBEAT_STOP, _HEARTBEAT_THREAD
    with _STATE_LOCK:
        _RECORDING_COUNT += 1
        if _RECORDING_COUNT == 1:
            _write_state("RECORDING")
            _HEARTBEAT_STOP = threading.Event()
            _HEARTBEAT_THREAD = threading.Thread(
                target=_heartbeat_loop,
                args=(_HEARTBEAT_STOP,),
                daemon=True,
                name="motion-state-heartbeat",
            )
            _HEARTBEAT_THREAD.start()


def record_end_signal() -> None:
    """Called when a clip ends. Final camera flips state back to IDLE."""
    global _RECORDING_COUNT, _HEARTBEAT_STOP, _HEARTBEAT_THREAD
    stop_event: Optional[threading.Event] = None
    thread: Optional[threading.Thread] = None
    with _STATE_LOCK:
        _RECORDING_COUNT = max(0, _RECORDING_COUNT - 1)
        if _RECORDING_COUNT == 0:
            stop_event = _HEARTBEAT_STOP
            thread = _HEARTBEAT_THREAD
            _HEARTBEAT_STOP = None
            _HEARTBEAT_THREAD = None
            _write_state("IDLE")
    # Join outside the lock to avoid deadlock with the heartbeat thread.
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)

# Suppress OpenCV / TF noise
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'


# ─────────────────────────── CONFIGURATION ────────────────────────────────────

@dataclass
class Config:
    """All tunable parameters in one place. Edit here or pass a JSON file."""

    # ── Storage ──────────────────────────────────────────────────────────────
    recording_dir: str = "recordings"
    max_storage_gb: float = 10.0          # hard cap; oldest files deleted first
    cleanup_trigger_gb: float = 8.0       # start cleanup before hitting the cap

    # ── Recording ────────────────────────────────────────────────────────────
    clip_duration_sec: float = 10.0       # fixed recording window after motion
    pre_buffer_sec: float = 1.0           # seconds of frames kept before trigger
    codec: str = "mp4v"                   # fourcc code; avc1 for H.264 if ffmpeg

    # ── Camera ───────────────────────────────────────────────────────────────
    camera_index: int = 0                 # -1 = auto-scan 0-9
    target_fps: int = 20
    frame_width: int = 0                  # 0 = use camera default
    frame_height: int = 0
    rotation: int = 0                     # 0 | 90 | 180 | 270
    camera_retries: int = 5
    camera_retry_delay: float = 2.0
    warmup_frames: int = 20

    # ── Motion Detection ─────────────────────────────────────────────────────
    movement_threshold: int = 25          # pixel diff threshold (0-255)
    min_contour_area: int = 800           # px² – ignore tiny blobs
    max_changed_ratio: float = 0.85       # ignore full-frame changes (light switch)
    min_changed_ratio: float = 0.0008     # ignore sensor noise
    confirm_frames: int = 2               # N consecutive detections to confirm
    cooldown_sec: float = 5.0             # seconds to ignore motion after clip ends

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = ""                    # empty = console only

    # ── Hardware / Identity ────────────────────────────────────────────────
    hardware_id: str = ""                # e.g. "0458:6006"
    camera_type: str = "motion"          # e.g. "in", "out", "playground"

    # ── Multi-camera support ───────────────────────────────────────────────
    cameras: List[dict] = field(default_factory=list)

    # ── JSON persistence ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ─────────────────────────── LOGGING ──────────────────────────────────────────

def build_logger(cfg: Config) -> logging.Logger:
    fmt = "[%(asctime)s] [%(levelname)-8s] %(message)s"
    dfmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg.log_file:
        handlers.append(logging.FileHandler(cfg.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format=fmt, datefmt=dfmt, handlers=handlers
    )
    return logging.getLogger("motion")


# ─────────────────────────── STORAGE MANAGER ──────────────────────────────────

class StorageManager:
    """Tracks disk usage; deletes oldest clips when quota is approached."""

    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self.root = Path(cfg.recording_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _videos(self) -> List[Path]:
        return sorted(self.root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)

    def _total_gb(self) -> float:
        try:
            return sum(p.stat().st_size for p in self._videos()) / 1e9
        except OSError:
            return 0.0

    def ensure_space(self) -> None:
        """Remove oldest clips until we are under the trigger threshold."""
        with self._lock:
            gb = self._total_gb()
            if gb < self.cfg.cleanup_trigger_gb:
                return
            self.log.warning(f"Storage {gb:.2f} GB ≥ trigger {self.cfg.cleanup_trigger_gb} GB — cleaning up")
            for path in self._videos():
                if self._total_gb() < self.cfg.cleanup_trigger_gb:
                    break
                try:
                    size = path.stat().st_size / 1e6
                    path.unlink()
                    self.log.info(f"Deleted {path.name} ({size:.1f} MB)")
                except OSError as exc:
                    self.log.error(f"Cannot delete {path.name}: {exc}")

    def unique_path(self, ts: datetime, duration: float) -> Path:
        date_folder = ts.strftime("%d-%m-%Y")
        self.root.mkdir(parents=True, exist_ok=True)
        target_dir = self.root / date_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_type = (self.cfg.camera_type or "motion").strip().replace(" ", "_")
        if not safe_type:
            safe_type = "motion"
        name = f"{safe_type}_{ts.strftime('%Y%m%d_%H%M%S')}_{duration:.0f}s.mp4"
        return target_dir / name

    def cleanup_partials(self) -> None:
        """Remove abandoned *.mp4.part files (process killed mid-write)."""
        try:
            for part in self.root.glob("*.mp4.part"):
                try:
                    part.unlink()
                    self.log.info(f"Removed stale partial: {part.name}")
                except OSError as exc:
                    self.log.warning(f"Cannot remove stale partial {part.name}: {exc}")
        except Exception as exc:
            self.log.debug(f"cleanup_partials error: {exc}")


# ─────────────────────────── MOTION DETECTOR ──────────────────────────────────

class MotionDetector:
    """
    Dual-method detector:
      1. Frame-diff + contour area (fast, deterministic)
      2. MOG2 background subtractor (adaptive to slow lighting changes)

    Both must agree (OR logic) to flag motion. A sliding confirmation
    buffer prevents single-frame glitches from triggering a recording.
    """

    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self._buf: deque = deque(maxlen=max(cfg.confirm_frames, 1))
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=20, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    def _ratio_ok(self, ratio: float) -> bool:
        return self.cfg.min_changed_ratio <= ratio <= self.cfg.max_changed_ratio

    # ── public API ────────────────────────────────────────────────────────────

    def check(
        self,
        prev: Optional[np.ndarray],
        curr: np.ndarray,
        *,
        update_bg: bool = True,
    ) -> bool:
        """
        Feed one frame pair and return True if confirmed motion.
        Pass update_bg=False while recording to keep the model stable.
        """
        result = False

        if prev is not None:
            result = result or self._frame_diff(prev, curr)

        lr = 0.005 if update_bg else 0.0
        result = result or self._mog2_check(curr, lr)

        self._buf.append(result)
        confirmed = sum(self._buf) >= max(1, self.cfg.confirm_frames // 2 + 1)
        return confirmed

    def _frame_diff(self, prev: np.ndarray, curr: np.ndarray) -> bool:
        try:
            diff = cv2.absdiff(self._gray(prev), self._gray(curr))
            _, thresh = cv2.threshold(diff, self.cfg.movement_threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel)
            thresh = cv2.dilate(thresh, self._kernel, iterations=2)

            total = thresh.shape[0] * thresh.shape[1]
            ratio = np.count_nonzero(thresh) / total
            if not self._ratio_ok(ratio):
                return False

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return any(cv2.contourArea(c) >= self.cfg.min_contour_area for c in contours)
        except Exception as exc:
            self.log.debug(f"frame_diff error: {exc}")
            return False

    def _mog2_check(self, frame: np.ndarray, lr: float) -> bool:
        try:
            mask = self._mog2.apply(frame, learningRate=lr)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            total = mask.shape[0] * mask.shape[1]
            ratio = np.count_nonzero(mask) / total
            if not self._ratio_ok(ratio):
                return False
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return any(cv2.contourArea(c) >= self.cfg.min_contour_area for c in contours)
        except Exception as exc:
            self.log.debug(f"mog2 error: {exc}")
            return False

    def reset(self) -> None:
        self._buf.clear()
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=20, detectShadows=False
        )


# ─────────────────────────── CAMERA MANAGER ───────────────────────────────────

class CameraManager:
    """Opens the camera; auto-retries on failure; exposes read_frame()."""

    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self._cap: Optional[cv2.VideoCapture] = None
        self._active_index: Optional[int] = None

    # ── public ────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Find + open a working camera. Returns True on success."""
        hw_indices = _indices_for_hardware_id(self.cfg.hardware_id)
        if hw_indices:
            indices = hw_indices
            self.log.info(f"Using hardware_id {self.cfg.hardware_id} → indices {indices}")
        else:
            indices = (
                [self.cfg.camera_index]
                if self.cfg.camera_index >= 0
                else list(range(10))
            )
        for attempt in range(1, self.cfg.camera_retries + 1):
            for idx in indices:
                if self._try_open(idx):
                    return True
            if attempt < self.cfg.camera_retries:
                self.log.warning(
                    f"No camera found (attempt {attempt}/{self.cfg.camera_retries}), "
                    f"retrying in {self.cfg.camera_retry_delay}s…"
                )
                time.sleep(self.cfg.camera_retry_delay)
        self.log.error("Camera open failed after all retries.")
        return False

    def warmup(self) -> bool:
        self.log.info(f"Warming up camera ({self.cfg.warmup_frames} frames)…")
        for i in range(self.cfg.warmup_frames):
            ok, _ = self.read_frame()
            if not ok:
                self.log.error(f"Warmup failed at frame {i}")
                return False
        self.log.info("Camera ready ✓")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        try:
            ret, frame = self._cap.read()
            return ret, frame if ret else None
        except Exception as exc:
            self.log.error(f"read_frame error: {exc}")
            return False, None

    @property
    def fps(self) -> float:
        if self._cap is None:
            return float(self.cfg.target_fps)
        v = self._cap.get(cv2.CAP_PROP_FPS)
        return v if 1 < v <= 120 else float(self.cfg.target_fps)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self.log.debug("Camera released.")

    # ── private ───────────────────────────────────────────────────────────────

    def _try_open(self, idx: int) -> bool:
        self.log.debug(f"Trying camera index {idx}…")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False
        self._apply_settings(cap)
        self._cap = cap
        self._active_index = idx
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.log.info(f"Camera #{idx} opened: {w}×{h} @ {fps:.1f} fps")
        return True

    def _apply_settings(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FPS, self.cfg.target_fps)
        if self.cfg.frame_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        if self.cfg.frame_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal latency


# ─────────────────────────── VIDEO WRITER ─────────────────────────────────────

_ROTATE = {
    90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


class ClipWriter:
    """
    Writes a single MP4 clip from a list of frames.
    File is named: motion_YYYYMMDD_HHMMSS_<duration>s.mp4
    """

    def __init__(self, cfg: Config, storage: StorageManager, log: logging.Logger):
        self.cfg = cfg
        self.storage = storage
        self.log = log

    def save(self, frames: List[np.ndarray], fps: float, triggered_at: datetime) -> bool:
        if not frames:
            self.log.warning("save() called with empty frame list — skipping.")
            return False

        duration = len(frames) / max(fps, 1)
        path = self.storage.unique_path(triggered_at, duration)
        # Write to a .part sidecar so consumers don't pick up a half-written clip.
        temp_path = path.parent / (path.name + ".part")

        sample = self._rotate(frames[0])
        h, w = sample.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            self.log.error(f"VideoWriter failed to open: {temp_path}")
            return False

        try:
            for f in frames:
                writer.write(self._rotate(f))
        finally:
            writer.release()

        if not temp_path.exists() or temp_path.stat().st_size < 1024:
            self.log.error(f"Output file missing or empty: {temp_path}")
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            return False

        try:
            os.replace(str(temp_path), str(path))
        except OSError as exc:
            self.log.error(f"Cannot finalize clip {path.name}: {exc}")
            return False

        mb = path.stat().st_size / 1e6
        self.log.info(
            f"Saved → {path.name}  "
            f"({len(frames)} frames, {duration:.1f}s, {mb:.1f} MB)"
        )
        return True

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        code = _ROTATE.get(self.cfg.rotation)
        return cv2.rotate(frame, code) if code is not None else frame


# ─────────────────────────── MAIN SYSTEM ──────────────────────────────────────

class MotionCameraSystem:
    """
    Orchestrates the full pipeline:
      idle → monitor → MOTION DETECTED → record 10 s → save → cooldown → idle
    """

    def __init__(self, cfg: Config, *, register_signals: bool = True):
        self.cfg = cfg
        self.log = build_logger(cfg)
        self.storage = StorageManager(cfg, self.log)
        self.camera = CameraManager(cfg, self.log)
        self.detector = MotionDetector(cfg, self.log)
        self.writer = ClipWriter(cfg, self.storage, self.log)
        self._running = False
        self._shutdown_event = threading.Event()

        if register_signals:
            # Register OS signals for clean exit
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.log.info("═" * 60)
        self.log.info("  Motion Detection System — starting")
        self.log.info(f"  Recordings → {Path(self.cfg.recording_dir).resolve()}")
        self.log.info(f"  Clip length → {self.cfg.clip_duration_sec} s")
        self.log.info(f"  Cooldown    → {self.cfg.cooldown_sec} s")
        self.log.info("═" * 60)

        # Sweep any half-written clips left behind by a prior crash/kill.
        self.storage.cleanup_partials()

        if not self.camera.open():
            self.log.critical("Cannot open camera. Exiting.")
            sys.exit(1)

        if not self.camera.warmup():
            self.log.critical("Camera warmup failed. Exiting.")
            self.camera.release()
            sys.exit(1)

        self._running = True
        self._monitor_loop()

    def _handle_signal(self, signum, _frame) -> None:
        self.log.info(f"Signal {signum} received — shutting down…")
        self._running = False
        self._shutdown_event.set()
        # Explicitly flip the IPC state back to IDLE so any consumer (main.py)
        # stops waiting immediately, instead of relying on atexit (which does
        # not fire on SIGKILL or some hosted-process terminations).
        try:
            _write_state("IDLE")
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        self._shutdown_event.set()
        self.camera.release()

    def _shutdown(self) -> None:
        self._running = False
        self.camera.release()
        cv2.destroyAllWindows()
        self.log.info("System stopped cleanly.")

    # ── monitoring loop ───────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        """
        Idle state: read frames, feed detector.
        On confirmed motion: enter _record_clip(), then cooldown.
        Handles camera reconnection automatically.
        """
        fps = self.camera.fps
        pre_buf_len = max(1, int(self.cfg.pre_buffer_sec * fps))
        pre_buffer: deque = deque(maxlen=pre_buf_len)

        prev_frame: Optional[np.ndarray] = None
        in_cooldown_until = 0.0

        self.log.info("Monitoring for motion…  (Ctrl-C to stop)")

        while self._running:
            ok, frame = self.camera.read_frame()

            if not ok or frame is None:
                self.log.error("Frame read failed — attempting camera reconnect…")
                self.camera.release()
                time.sleep(1.0)
                if not self.camera.open():
                    self.log.critical("Reconnect failed. Exiting.")
                    break
                prev_frame = None
                pre_buffer.clear()
                self.detector.reset()
                continue

            pre_buffer.append(frame)

            # ── cooldown period ───────────────────────────────────────────────
            now = time.time()
            if now < in_cooldown_until:
                prev_frame = frame
                remaining = in_cooldown_until - now
                # Sleep briefly, keep draining frames
                time.sleep(min(0.05, remaining))
                continue

            # ── motion check ──────────────────────────────────────────────────
            if self.detector.check(prev_frame, frame):
                triggered_at = datetime.now()
                self.log.info(
                    f"══ MOTION DETECTED @ {triggered_at.strftime('%Y-%m-%d %H:%M:%S')} ══"
                )

                # Record the fixed-length clip
                self._record_clip(
                    trigger_frame=frame,
                    pre_frames=list(pre_buffer),
                    triggered_at=triggered_at,
                )

                # Async storage cleanup (doesn't block the camera loop)
                threading.Thread(
                    target=self.storage.ensure_space, daemon=True
                ).start()

                # Reset state
                self.detector.reset()
                pre_buffer.clear()
                prev_frame = None
                in_cooldown_until = time.time() + self.cfg.cooldown_sec
                self.log.info(f"Cooldown {self.cfg.cooldown_sec}s started.")
                continue

            prev_frame = frame

    # ── clip recording ────────────────────────────────────────────────────────

    def _record_clip(
        self,
        trigger_frame: np.ndarray,
        pre_frames: List[np.ndarray],
        triggered_at: datetime,
    ) -> None:
        """
        Record exactly `clip_duration_sec` seconds of video starting from the
        motion trigger, prepend the pre-roll buffer, then save.
        """
        fps = self.camera.fps
        total_frames_needed = int(self.cfg.clip_duration_sec * fps)

        clip: List[np.ndarray] = list(pre_frames) + [trigger_frame]

        self.log.info(
            f"Recording {self.cfg.clip_duration_sec}s clip "
            f"(need {total_frames_needed} frames @ {fps:.1f} fps)…"
        )

        record_start_signal()
        record_start = time.time()
        deadline = record_start + self.cfg.clip_duration_sec
        frames_recorded = 1  # trigger_frame already added

        try:
            while time.time() < deadline and self._running:
                ok, frame = self.camera.read_frame()

                if not ok or frame is None:
                    self.log.warning("Frame drop during recording.")
                    time.sleep(0.01)
                    continue

                clip.append(frame)
                frames_recorded += 1
        finally:
            record_end_signal()

        elapsed = time.time() - record_start
        self.log.info(
            f"Recording finished: {frames_recorded} frames in {elapsed:.1f}s "
            f"(effective {frames_recorded / max(elapsed, 0.001):.1f} fps)"
        )

        # Save asynchronously so we're back monitoring ASAP
        saved_frames = list(clip)  # copy before thread starts
        save_thread = threading.Thread(
            target=self._save_async,
            args=(saved_frames, fps, triggered_at),
            daemon=True,
        )
        save_thread.start()

    def _save_async(
        self,
        frames: List[np.ndarray],
        fps: float,
        triggered_at: datetime,
    ) -> None:
        try:
            self.writer.save(frames, fps, triggered_at)
        except Exception as exc:
            self.log.error(f"Async save failed: {exc}", exc_info=True)


# ─────────────────────────── ENTRY POINT ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Motion Detection Camera System — records 10-second clips on motion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",          help="Path to JSON config file")
    p.add_argument("--save-config",     help="Write current config to file and exit")
    p.add_argument("--recording-dir",   default="recordings",   help="Where to store MP4 clips")
    p.add_argument("--clip-duration",   type=float, default=10.0, help="Clip length in seconds")
    p.add_argument("--cooldown",        type=float, default=0.0,  help="Cooldown between clips (s)")
    p.add_argument("--camera-index",    type=int,   default=0,    help="Camera index (-1 = auto)")
    p.add_argument("--rotation",        type=int,   default=90,    choices=[0, 90, 180, 270])
    p.add_argument("--log-level",       default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file",        default="",  help="Optional log file path")
    return p.parse_args()


def _apply_overrides(cfg: Config, overrides: dict) -> Config:
    updated = replace(cfg)
    for k, v in overrides.items():
        if hasattr(updated, k):
            setattr(updated, k, v)
    return updated


def _build_camera_configs(cfg: Config) -> List[Config]:
    if not cfg.cameras:
        return [cfg]
    result: List[Config] = []
    for cam in cfg.cameras:
        if not isinstance(cam, dict):
            continue
        overrides = {
            "camera_index": cam.get("camera_index", cfg.camera_index),
            "camera_type": cam.get("camera_type", cfg.camera_type),
            "hardware_id": cam.get("hardware_id", cfg.hardware_id),
            "recording_dir": cam.get("recording_dir", cfg.recording_dir),
            "rotation": cam.get("rotation", cfg.rotation),
            "target_fps": cam.get("target_fps", cfg.target_fps),
            "frame_width": cam.get("frame_width", cfg.frame_width),
            "frame_height": cam.get("frame_height", cfg.frame_height),
            "log_level": cam.get("log_level", cfg.log_level),
            "log_file": cam.get("log_file", cfg.log_file),
        }
        result.append(_apply_overrides(cfg, overrides))
    return result


def _list_connected_camera_indices() -> List[int]:
    indices: List[int] = []
    for path in glob.glob("/sys/class/video4linux/video*"):
        try:
            name = os.path.basename(path)
            idx = int(name.replace("video", ""))
            indices.append(idx)
        except Exception:
            continue
    return sorted(indices)


def _read_hardware_id_for_index(index: int) -> str:
    uevent_path = f"/sys/class/video4linux/video{index}/device/uevent"
    try:
        with open(uevent_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRODUCT="):
                    value = line.strip().split("=", 1)[1]
                    parts = value.split("/")
                    if len(parts) >= 2:
                        vid = int(parts[0])
                        pid = int(parts[1])
                        return f"{vid:04x}:{pid:04x}"
    except Exception:
        return ""
    return ""


def _indices_for_hardware_id(hw_id: str) -> List[int]:
    if not hw_id:
        return []
    target = hw_id.strip().lower()
    matches: List[int] = []
    for idx in _list_connected_camera_indices():
        found = _read_hardware_id_for_index(idx).lower()
        if found and found == target:
            matches.append(idx)
    return matches


def _next_unknown_type(existing: List[str]) -> str:
    n = 1
    while True:
        candidate = f"unknown_{n}"
        if candidate not in existing:
            return candidate
        n += 1


def _sync_cameras_with_config(cfg: Config, config_path: str | None) -> Config:
    connected = _list_connected_camera_indices()
    existing = cfg.cameras or []
    known_by_index = {c.get("camera_index"): c for c in existing if isinstance(c, dict)}
    existing_types = [c.get("camera_type", "") for c in existing if isinstance(c, dict)]
    updated = list(existing)

    for idx in connected:
        if idx in known_by_index:
            continue
        hw_id = _read_hardware_id_for_index(idx)
        cam_type = _next_unknown_type(existing_types)
        existing_types.append(cam_type)
        updated.append(
            {
                "camera_index": idx,
                "camera_type": cam_type,
                "hardware_id": hw_id,
            }
        )

    cfg.cameras = updated

    if config_path:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg.__dict__, f, indent=2)
        except Exception:
            pass
    return cfg


def main() -> None:
    args = parse_args()

    # Import paths only inside main() — keeps `from motion import ...` cheap
    # and lets the project be imported even when config.json is incomplete
    # (e.g. during tests). bootstrap_dirs() ensures every configured directory
    # exists before we touch the filesystem.
    import paths
    paths.bootstrap_dirs()

    # Load or build config
    if args.config and os.path.exists(args.config):
        cfg = Config.load(args.config)
    else:
        cfg = Config()

    # CLI overrides. paths.VIDEOS_DIR is the project-wide source of truth so
    # motion.py and main.py always agree on where clips live.
    cfg.recording_dir   = paths.VIDEOS_DIR
    cfg.clip_duration_sec = args.clip_duration
    cfg.cooldown_sec    = args.cooldown
    cfg.camera_index    = args.camera_index
    cfg.rotation        = args.rotation
    cfg.log_level       = args.log_level
    cfg.log_file        = args.log_file

    config_path = args.config if args.config and os.path.exists(args.config) else None
    cfg = _sync_cameras_with_config(cfg, config_path)

    # Initialize the state file early so main.py can begin processing
    # immediately when motion.py starts up.
    init_state_file(paths.MOTION_STATE_FILE)

    if args.save_config:
        cfg.save(args.save_config)
        print(f"Config saved to {args.save_config}")
        return

    camera_cfgs = _build_camera_configs(cfg)
    if len(camera_cfgs) == 1:
        MotionCameraSystem(camera_cfgs[0]).start()
        return

    systems: List[MotionCameraSystem] = []
    threads: List[threading.Thread] = []
    for cam_cfg in camera_cfgs:
        system = MotionCameraSystem(cam_cfg, register_signals=False)
        systems.append(system)
        t = threading.Thread(target=system.start, daemon=True)
        threads.append(t)
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        for system in systems:
            system.stop()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()