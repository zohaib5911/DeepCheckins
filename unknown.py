import json
from datetime import datetime
import os
import time
import cv2
import numpy as np
import struct
from HNSW import FaissIndex
from database import INFO
from typing import  List
from log import write_log
users: List[INFO] = []
from collections import defaultdict

_logged_unknown_by_video = defaultdict(set)


def _should_log_unknown(video_name: str, uid: int) -> bool:
    if not video_name:
        return True
    if uid in _logged_unknown_by_video[video_name]:
        return False
    _logged_unknown_by_video[video_name].add(uid)
    return True


with open("config.json", "r") as f:
    config = json.load(f)

import paths

unknownFolder = paths.UNKNOWN_DIR
unknownlogpath = paths.UNKNOWN_LOG_FILE
unknown_debug_dir = paths.UNKNOWN_DEBUG_DIR
EMB_DIM = config["EMB_DIM"]
cooling_period = config["cooling_period"]

UNKNOWN_BIN = paths.UNKNOWN_BIN

def save_unknown_image(name: str, frame: np.ndarray) -> str:
    os.makedirs(unknownFolder, exist_ok=True)
    file_path = os.path.join(unknownFolder, f"{name}.jpg")
    if isinstance(frame, np.ndarray):
        cv2.imwrite(file_path, frame)
    else:
        raise TypeError("frame must be a numpy array (OpenCV image)")
    return file_path


def _safe_unknown_filename(video_name: str, frame_no: int) -> str:
    base = os.path.splitext(video_name)[0] if video_name else "unknown"
    base = base.replace(" ", "_")
    return f"{base}_F{frame_no}.jpg"


def _save_unknown_debug(uid: int, frame: np.ndarray, timestamp: str, frame_no: int, message: str, video_name: str) -> None:
    folder = os.path.join(unknown_debug_dir, str(uid))
    os.makedirs(folder, exist_ok=True)
    img_path = os.path.join(folder, _safe_unknown_filename(video_name, frame_no))
    try:
        cv2.imwrite(img_path, frame)
    except Exception:
        pass
    log_path = os.path.join(folder, "log.txt")
    write_log(log_path, timestamp, frame_no, message, video_name)

def write_unknown(unk_id: int, embedding: np.ndarray):
    embedding = np.asarray(embedding, dtype=np.float32)

    os.makedirs(os.path.dirname(UNKNOWN_BIN), exist_ok=True)
    with open(UNKNOWN_BIN, "ab") as f:
        f.write(struct.pack("<I", unk_id))
        f.write(embedding.tobytes())


def load_unknown_bin():
    if not os.path.exists(UNKNOWN_BIN):
        return []
    record_size = 4 + (EMB_DIM * 4)
    with open(UNKNOWN_BIN, "rb") as f:
        data = f.read()

    records = []

    for i in range(0, len(data), record_size):
        chunk = data[i:i + record_size]
        if len(chunk) != record_size:
            continue

        unk_id = struct.unpack("<I", chunk[:4])[0]
        emb = np.frombuffer(chunk[4:], dtype=np.float32)

        records.append((unk_id, emb))

    return records


def get_next_unknown_id() -> int:
    records = load_unknown_bin()
    if not records:
        return 1
    return max(r[0] for r in records) + 1


def add_unknown( embedding: np.ndarray):
    unk_id = get_next_unknown_id()
    write_unknown(unk_id, embedding)
    return unk_id

def unknown_handler(frame, embedding, threshold=0.75, timestamp: str = "", frame_no: int = 0, video_name: str = ""):
    unknownFaiss = FaissIndex(mode="unknown")
    unknownFaiss.build_faiss(load_unknown_bin(), use_exact=True)
    uid , score = unknownFaiss.search_faiss_best(embedding)
    if uid is not None and score >= threshold:
        now = time.time()
        for user in users:
            if user.userid == uid:
                if now - user.lastseen > cooling_period:
                    user.lastseen = now
                    save_unknown_image(uid, frame)
                    # add in log
                    if _should_log_unknown(video_name, uid):
                        write_log(unknownlogpath, timestamp, frame_no, f"{uid} --- {score:.2f}", video_name)
                        _save_unknown_debug(uid, frame, timestamp, frame_no, f"{uid} --- {score:.2f}", video_name)
                return uid

    new_id = add_unknown(embedding)
    save_unknown_image(new_id, frame)
    # add in log
    unknownFaiss.build_faiss(load_unknown_bin(), use_exact=True)
    users.append(INFO(new_id, time.time()))
    if _should_log_unknown(video_name, new_id):
        write_log(unknownlogpath, timestamp, frame_no, f"New unknown {new_id} --- {score:.2f}", video_name)
        _save_unknown_debug(new_id, frame, timestamp, frame_no, f"New unknown {new_id} --- {score:.2f}", video_name)
    return new_id
