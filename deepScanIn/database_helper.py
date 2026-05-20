import json
import os
import struct
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import pillow_heif
import shutil
import tkinter as tk

from arcFace import getEmbedding
from database import User, Database

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

STRUCT_PREFIX = "<"
VALID_FMT = config.get("VALID_FMT", "B")
ID_FMT = config.get("ID_FMT", "64s")
COUNT_FMT = config.get("COUNT_FMT", "B")

VALID_BYTES = struct.calcsize(STRUCT_PREFIX + VALID_FMT)
ID_BYTES = struct.calcsize(STRUCT_PREFIX + ID_FMT)
COUNT_BYTES = struct.calcsize(STRUCT_PREFIX + COUNT_FMT)

EMB_PER_USER = config["EMB_PER_USER"]
EMB_DIM = config["EMB_DIM"]
EMB_DTYPE = np.dtype(config["EMB_DTYPE"])
MAX_USERS = config["MAX_USERS"]

import paths

database_path = paths.DATABASE_PATH

IMG_H = config.get("IMG_H", 128)
IMG_W = config.get("IMG_W", 128)
IMG_C = config.get("IMG_C", 3)
IMAGE_BYTES = config.get("IMAGE_BYTES", IMG_H * IMG_W * IMG_C)

EMB_BYTES = EMB_DTYPE.itemsize * EMB_DIM * EMB_PER_USER
IMG_BLOCK_BYTES = EMB_PER_USER * IMAGE_BYTES
USER_RECORD_SIZE = VALID_BYTES + ID_BYTES + COUNT_BYTES + EMB_BYTES + IMG_BLOCK_BYTES

base_folder = paths.REGISTRATION_DIR
deletedpath = paths.DELETED_REGISTRATION_DIR
delete = config.get("delete", False)


def _pack_id(user_id) -> bytes:
    if ID_FMT.endswith("s"):
        encoded = str(user_id).encode("utf-8")[:ID_BYTES]
        return encoded.ljust(ID_BYTES, b"\x00")
    return struct.pack(STRUCT_PREFIX + ID_FMT, int(user_id))


def _unpack_id(raw: bytes):
    if ID_FMT.endswith("s"):
        return raw.rstrip(b"\x00").decode("utf-8", errors="ignore")
    return struct.unpack(STRUCT_PREFIX + ID_FMT, raw)[0]


def _iter_user_ids(db_path: str = database_path) -> list:
    user_ids = []
    if not os.path.exists(db_path):
        return user_ids
    with open(db_path, "rb") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid_bytes = f.read(VALID_BYTES)
            if not valid_bytes:
                break
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, valid_bytes)[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            user_ids.append(_unpack_id(raw_id))
    return sorted(user_ids)


def get_user_embedding_count(user_id, db_path: str = database_path) -> int:
    if not os.path.exists(db_path):
        return 0
    with open(db_path, "rb") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            current_id = _unpack_id(raw_id)
            if current_id == user_id:
                count = struct.unpack(STRUCT_PREFIX + COUNT_FMT, f.read(COUNT_BYTES))[0]
                return int(count)
    return 0


def _read_user_image_bytes(user_id, image_index: int = 0, db_path: str = database_path) -> bytes | None:
    if not os.path.exists(db_path):
        return None
    if image_index < 0 or image_index >= EMB_PER_USER:
        raise ValueError("Invalid image index")
    with open(db_path, "rb") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            current_id = _unpack_id(raw_id)
            if current_id == user_id:
                count = struct.unpack(STRUCT_PREFIX + COUNT_FMT, f.read(COUNT_BYTES))[0]
                if image_index >= int(count):
                    return None
                img_offset = offset + VALID_BYTES + ID_BYTES + COUNT_BYTES + EMB_BYTES
                f.seek(img_offset + image_index * IMAGE_BYTES)
                return f.read(IMAGE_BYTES)
    return None


def _write_user_image_bytes(user_id, image_bytes: bytes, image_index: int = 0, db_path: str = database_path) -> bool:
    if not os.path.exists(db_path):
        return False
    if len(image_bytes) != IMAGE_BYTES:
        raise ValueError("Invalid image size")
    if image_index < 0 or image_index >= EMB_PER_USER:
        raise ValueError("Invalid image index")
    with open(db_path, "r+b") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            current_id = _unpack_id(raw_id)
            if current_id == user_id:
                img_offset = offset + VALID_BYTES + ID_BYTES + COUNT_BYTES + EMB_BYTES
                f.seek(img_offset + image_index * IMAGE_BYTES)
                f.write(image_bytes)
                return True
    return False


def _get_user_image_array(user_id, image_index: int = 0, db_path: str = database_path) -> np.ndarray | None:
    img_bytes = _read_user_image_bytes(user_id, image_index, db_path)
    if img_bytes is None:
        return None
    return np.frombuffer(img_bytes, dtype=np.uint8).reshape((IMG_H, IMG_W, IMG_C))


def _get_screen_size() -> tuple[int, int] | None:
    try:
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return None


def _show_user_image(user_id, image_index: int = 0, db_path: str = database_path) -> bool:
    img_array = _get_user_image_array(user_id, image_index, db_path)
    if img_array is None:
        return False
    window_name = f"User ID {user_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    screen_size = _get_screen_size()
    if screen_size is not None:
        cv2.resizeWindow(window_name, screen_size[0] // 2, screen_size[1] // 2)
    cv2.imshow(window_name, img_array)
    cv2.waitKey(1)
    return True


def _close_image_window():
    cv2.destroyAllWindows()


def _resize_to_square(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return cv2.resize(frame, (target_w, target_h))

    if w > h:
        x1 = (w - h) // 2
        x2 = x1 + h
        cropped = frame[:, x1:x2]
    else:
        y1 = (h - w) // 2
        y2 = y1 + w
        cropped = frame[y1:y2, :]

    return cv2.resize(cropped, (target_w, target_h))


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
    resized = _resize_to_square(frame, IMG_W, IMG_H)
    if resized.shape[2] != IMG_C:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized.tobytes()


def dropBYID(user_id, db_path: str = database_path) -> bool:
    if not os.path.exists(db_path):
        return False

    with open(db_path, "r+b") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            current_id = _unpack_id(raw_id)
            if current_id == user_id:
                f.seek(offset)
                f.write(struct.pack(STRUCT_PREFIX + VALID_FMT, 0))
                return True

    return False


def replaceEmbedding(user_id, embedding_no: int, new_embedding: np.ndarray, db_path: str = database_path) -> bool:
    if not os.path.exists(db_path):
        return False

    if embedding_no < 0 or embedding_no >= EMB_PER_USER:
        raise ValueError("Invalid embedding number")

    with open(db_path, "r+b") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            current_id = _unpack_id(raw_id)
            if current_id == user_id:
                emb_offset = (
                    offset
                    + VALID_BYTES
                    + ID_BYTES
                    + COUNT_BYTES
                    + (embedding_no * EMB_DIM * EMB_DTYPE.itemsize)
                )
                f.seek(emb_offset)
                f.write(new_embedding.astype(EMB_DTYPE).tobytes())
                return True

    return False


def remove_embedding(user_id, embedding_no: int, db_path: str = database_path) -> bool:
    if not os.path.exists(db_path):
        return False

    if embedding_no < 0 or embedding_no >= EMB_PER_USER:
        raise ValueError("Invalid embedding number")

    if EMB_PER_USER <= 1:
        raise ValueError("Cannot remove embedding: only one embedding per user is available, so there is no other embedding to duplicate.")

    with open(db_path, "r+b") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)
            valid = struct.unpack(STRUCT_PREFIX + VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue
            raw_id = f.read(ID_BYTES)
            current_id = _unpack_id(raw_id)
            if current_id == user_id:
                if embedding_no == 0:
                    alt_idx = 1
                elif embedding_no == EMB_PER_USER - 1:
                    alt_idx = EMB_PER_USER - 2
                else:
                    alt_idx = embedding_no - 1

                alt_emb_offset = (
                    offset
                    + VALID_BYTES
                    + ID_BYTES
                    + COUNT_BYTES
                    + (alt_idx * EMB_DIM * EMB_DTYPE.itemsize)
                )
                f.seek(alt_emb_offset)
                alt_embedding = f.read(EMB_DIM * EMB_DTYPE.itemsize)

                emb_offset = (
                    offset
                    + VALID_BYTES
                    + ID_BYTES
                    + COUNT_BYTES
                    + (embedding_no * EMB_DIM * EMB_DTYPE.itemsize)
                )
                f.seek(emb_offset)
                f.write(alt_embedding)

                img_base = (
                    offset
                    + VALID_BYTES
                    + ID_BYTES
                    + COUNT_BYTES
                    + EMB_BYTES
                )
                f.seek(img_base + alt_idx * IMAGE_BYTES)
                alt_image = f.read(IMAGE_BYTES)
                f.seek(img_base + embedding_no * IMAGE_BYTES)
                f.write(alt_image)
                return True

    return False


def _replace_embedding_from_image(user_id, embedding_no: int, image_path: str, db_path: str = database_path) -> bool:
    frame = _read_image_any(image_path)
    if frame is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return False
    result = getEmbedding(user_id, frame)
    if result is None:
        print("[ERROR] No valid embedding found in image")
        return False
    embedding, _quality = result
    if embedding is None or embedding.shape != (EMB_DIM,):
        print("[ERROR] No valid embedding found in image")
        return False
    if not replaceEmbedding(user_id, embedding_no, embedding.astype(EMB_DTYPE), db_path):
        return False
    resized = _resize_to_square(frame, IMG_W, IMG_H)
    _write_user_image_bytes(user_id, resized.tobytes(), embedding_no, db_path)
    return True


def _replace_user_image_from_file(user_id, image_path: str, image_index: int = 0, db_path: str = database_path) -> bool:
    frame = _read_image_any(image_path)
    if frame is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return False
    resized = _resize_to_square(frame, IMG_W, IMG_H)
    return _write_user_image_bytes(user_id, resized.tobytes(), image_index, db_path)


def register_to_dataset(db, user_path: str) -> tuple[bool, str]:
    if not os.path.isdir(user_path):
        return False, "Folder path not found."
    if db is None:
        db = Database()
        db.create()
    user_id = Path(user_path).name
    valid_data = []
    all_images = []
    for root, _dirs, files in os.walk(user_path):
        for img_name in files:
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg", ".heic", ".heif")):
                continue

            img_path = os.path.join(root, img_name)
            all_images.append(img_path)

            frame = _read_image_any(img_path)

            print(f"[DEBUG] Processing {img_name} for {user_id}")

            if frame is None:
                print(f"[WARN] Failed to read {img_path}")
                continue

            result = getEmbedding(user_id, frame)
            if result is None:
                print(f"[FAIL] Registration failed for {img_name} (No face/embedding)")
                continue
            embedding, quality = result
            if embedding is None:
                print(f"[FAIL] Registration failed for {img_name} (No face/embedding)")
                continue

            if embedding is not None and embedding.shape == (EMB_DIM,):
                valid_data.append((quality, embedding, frame))
            else:
                print(f"[FAIL] Registration failed for {img_name}")

    print(f"[DEBUG] Collected {len(valid_data)} valid embeddings for {user_id}")
    
    if len(valid_data) >= 1:
        valid_data.sort(key=lambda x: x[0], reverse=True)
        best_embeddings = [data[1] for data in valid_data[:EMB_PER_USER]]
        best_frames = [data[2] for data in valid_data[:EMB_PER_USER]]

        embeddings_array = np.stack(best_embeddings, axis=0)

        image_bytes_list = [_to_image_bytes(frame) for frame in best_frames]

        user = User(user_id, embeddings_array, image_bytes_list)
        db.append(user)
        print(f"[SUCCESS] Registered user {user_id} using {len(best_embeddings)} embeddings")

        if delete:
            for img_path in all_images:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"[CLEANUP] Deleted {os.path.basename(img_path)}")
        else:
            user_deleted_path = os.path.join(deletedpath, str(user_id))
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
        return True, f"Registered user {user_id}."
    else:
        return False, f"Need at least 1 valid image; found {len(valid_data)}."


def showPicture(user_id, db_path: str = database_path):
    if not _show_user_image(user_id, db_path):
        print(f"[ERROR] User ID {user_id} not found in database.")
        return
    cv2.waitKey(0)
    _close_image_window()


def _browse_users(db_path: str = database_path):
    user_ids = _iter_user_ids(db_path)
    if not user_ids:
        print("[INFO] No users found in database.")
        return

    idx = 0
    help_text = "←/→: prev/next | q/esc: exit"
    window_name = "User Browser"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    screen_size = _get_screen_size()
    if screen_size is not None:
        cv2.resizeWindow(window_name, screen_size[0] // 2, screen_size[1] // 2)
    while True:
        user_id = user_ids[idx]
        img_array = _get_user_image_array(user_id, db_path)
        if img_array is None:
            print("[WARN] Failed to show image for user.")
            idx = (idx + 1) % len(user_ids)
            continue

        display = img_array.copy()
        cv2.putText(
            display,
            f"User ID: {user_id} ({idx + 1}/{len(user_ids)})",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            help_text,
            (6, IMG_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(0)

        if key in (27, ord("q")):
            _close_image_window()
            return
        if key in (81, 2424832):
            idx = (idx - 1) % len(user_ids)
            continue
        if key in (83, 2555904):
            idx = (idx + 1) % len(user_ids)
            continue


def dataBaseEdit(db):
    while True:
        print("\n[DATABASE TOOL]")
        print("1. Browse users (images one-by-one)")
        print("2. Show user by ID")
        print("3. Drop user by ID")
        print("4. Remove user embedding")
        print("5. Replace embedding from image")
        print("6. Update user picture from image")
        print("7. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            _browse_users(db.path)
        elif choice == "2":
            try:
                user_id = input("Enter User ID: ").strip()
                showPicture(user_id, db.path)
            except ValueError:
                print("[ERROR] Invalid User ID.")
        elif choice == "3":
            user_id = input("Enter User ID to remove: ").strip()
            if dropBYID(user_id, db.path):
                print(f"[SUCCESS] User ID {user_id} removed.")
            else:
                print(f"[ERROR] User ID {user_id} not found.")
        elif choice == "4":
            try:
                user_id = input("Enter User ID: ").strip()
                embedding_no = int(input(f"Embedding Number (0-{EMB_PER_USER - 1}): "))
                if remove_embedding(user_id, embedding_no, db.path):
                    print(f"[SUCCESS] Embedding {embedding_no} removed.")
                else:
                    print("[ERROR] Failed to remove embedding.")
            except ValueError:
                print("[ERROR] Invalid input.")
        elif choice == "5":
            try:
                user_id = input("Enter User ID: ").strip()
                embedding_no = int(input(f"Embedding Number (0-{EMB_PER_USER - 1}): "))
                image_path = input("Image path: ").strip()
                if _replace_embedding_from_image(user_id, embedding_no, image_path, db.path):
                    print(f"[SUCCESS] Embedding {embedding_no} updated from image.")
                else:
                    print("[ERROR] Failed to update embedding from image.")
            except ValueError:
                print("[ERROR] Invalid input.")
        elif choice == "6":
            try:
                user_id = input("Enter User ID: ").strip()
                image_path = input("Image path: ").strip()
                if _replace_user_image_from_file(user_id, image_path, db.path):
                    print("[SUCCESS] User picture updated.")
                else:
                    print("[ERROR] Failed to update user picture.")
            except ValueError:
                print("[ERROR] Invalid input.")
        elif choice == "7":
            return
        else:
            print("[ERROR] Invalid choice.")
