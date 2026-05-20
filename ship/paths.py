"""
Single source of truth for every filesystem path the project uses.

All paths come from config.json — no module hard-codes a directory or file.
Entry-points (main.py, motion.py, app.py, editor.py, reset.py) call
``bootstrap_dirs()`` at startup so every configured directory exists before
anything tries to write to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).with_name("config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg: dict[str, Any] = json.load(_f)


def _require(key: str) -> str:
    val = _cfg.get(key)
    if not val:
        raise KeyError(
            f"config.json is missing required path '{key}'. "
            f"Add it before starting the system."
        )
    return str(val)


def _opt(key: str, default: str = "") -> str:
    return str(_cfg.get(key, default) or "")


# ─── Files ───────────────────────────────────────────────────────────────────
MODEL_PATH        = _require("model_path")
DATABASE_PATH     = _require("database_path")
UNKNOWN_BIN       = _require("unknown_bin")
KNOWN_LOG_FILE    = _require("known_log_file")
UNKNOWN_LOG_FILE  = _require("unknown_log_file")
DEBUG_LOG_FILE    = _require("debug_log_file")
MOTION_STATE_FILE = _require("motion_state_file")

# ─── Directories ─────────────────────────────────────────────────────────────
REGISTRATION_DIR         = _require("registration_dir")
DELETED_REGISTRATION_DIR = _require("deleted_registration_dir")
VIDEOS_DIR               = _require("videos_dir")
PROCESSED_VIDEOS_DIR     = _require("processed_videos_dir")
UNKNOWN_DIR              = _require("unknown_dir")
UNKNOWN_DEBUG_DIR        = _require("unknown_debug_dir")
NOFACE_DIR               = _require("noface_dir")
DEBUG_FAIL_DIR           = _require("debug_fail_dir")
KNOWN_DETECT_DIR         = _require("known_detect_dir")
KNOWN_PICS_DIR           = _require("known_pics_dir")

# ─── Network ─────────────────────────────────────────────────────────────────
APP_HOST    = _opt("app_host", "0.0.0.0")
APP_PORT    = int(_cfg.get("app_port", 5000))
EDITOR_HOST = _opt("editor_host", "0.0.0.0")
EDITOR_PORT = int(_cfg.get("editor_port", 5050))


_DIRS = (
    REGISTRATION_DIR,
    DELETED_REGISTRATION_DIR,
    VIDEOS_DIR,
    PROCESSED_VIDEOS_DIR,
    UNKNOWN_DIR,
    UNKNOWN_DEBUG_DIR,
    NOFACE_DIR,
    DEBUG_FAIL_DIR,
    KNOWN_DETECT_DIR,
    KNOWN_PICS_DIR,
)

_FILES_WITH_PARENT = (
    DATABASE_PATH,
    UNKNOWN_BIN,
    KNOWN_LOG_FILE,
    UNKNOWN_LOG_FILE,
    DEBUG_LOG_FILE,
    MOTION_STATE_FILE,
)


def bootstrap_dirs() -> None:
    """Create every configured directory + each file's parent directory.

    Idempotent and cheap to call multiple times. Entry-points should call
    this exactly once on startup so downstream code never has to defensively
    mkdir before writing.
    """
    for d in _DIRS:
        if d:
            os.makedirs(d, exist_ok=True)
    for f in _FILES_WITH_PARENT:
        if not f:
            continue
        parent = os.path.dirname(f)
        if parent:
            os.makedirs(parent, exist_ok=True)
