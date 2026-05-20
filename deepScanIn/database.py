import os
import struct
import numpy as np
from typing import List, Dict
import json
from typing import Any

with open("config.json", "r") as f:
    config = json.load(f)

import paths

db_path = paths.DATABASE_PATH
EMB_PER_USER = config["EMB_PER_USER"]
EMB_DIM = config["EMB_DIM"]
EMB_DTYPE = config["EMB_DTYPE"]
MAX_USERS = config["MAX_USERS"]
IMG_H = config.get("IMG_H", 128)
IMG_W = config.get("IMG_W", 128)
IMG_C = config.get("IMG_C", 3)
IMAGE_BYTES = config.get("IMAGE_BYTES", IMG_H * IMG_W * IMG_C)

STRUCT_PREFIX = "<"
VALID_FMT = STRUCT_PREFIX + "B"
ID_FMT = STRUCT_PREFIX + "64s"
COUNT_FMT = STRUCT_PREFIX + "B"
VALID_BYTES = struct.calcsize(VALID_FMT)
ID_BYTES = struct.calcsize(ID_FMT)
COUNT_BYTES = struct.calcsize(COUNT_FMT)
EMB_BYTES = np.dtype(EMB_DTYPE).itemsize * EMB_DIM * EMB_PER_USER
IMG_BLOCK_BYTES = IMAGE_BYTES * EMB_PER_USER
USER_RECORD_SIZE = VALID_BYTES + ID_BYTES + COUNT_BYTES + EMB_BYTES + IMG_BLOCK_BYTES
DB_SIZE = USER_RECORD_SIZE * MAX_USERS


class User:
    def __init__(self, user_id: str, embeddings: np.ndarray, images: List[bytes]):
        # Allow variable number of embeddings (1 to EMB_PER_USER)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] >= 1 and embeddings.shape[0] <= EMB_PER_USER
        assert embeddings.shape[1] == EMB_DIM
        assert embeddings.dtype == np.dtype(EMB_DTYPE)
        assert len(images) == embeddings.shape[0]
        for img in images:
            assert isinstance(img, (bytes, bytearray))
            assert len(img) == IMAGE_BYTES

        self.id = str(user_id)
        self.embeddings = embeddings
        self.images = images
        self.num_embeddings = embeddings.shape[0]

class Database:
    def __init__(self, path: str = db_path):
        self.path = path

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def _ensure_parent_dir(self) -> None:
        parent_dir = os.path.dirname(self.path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _encode_id(self, user_id: str) -> bytes:
        encoded = user_id.encode("utf-8")[:ID_BYTES]
        return encoded.ljust(ID_BYTES, b"\x00")

    def _decode_id(self, raw: bytes) -> str:
        return raw.rstrip(b"\x00").decode("utf-8", errors="ignore")

    def create(self):
        self._ensure_parent_dir()

        if os.path.exists(self.path):
            print("Database already exists")
            return

        with open(self.path, "wb") as f:
            f.write(b"\x00" * DB_SIZE)

        print(f"Created database: {self.path}")

    def append(self, user: User) -> int:
        self._ensure_parent_dir()

        if not os.path.exists(self.path):
            self.create()

        with open(self.path, "r+b") as f:
            for slot in range(MAX_USERS):
                offset = slot * USER_RECORD_SIZE
                f.seek(offset)

                valid = struct.unpack(VALID_FMT, f.read(VALID_BYTES))[0]

                if valid == 0:
                    f.seek(offset)

                    f.write(struct.pack(VALID_FMT, 1))

                    f.write(struct.pack(ID_FMT, self._encode_id(user.id)))

                    f.write(struct.pack(COUNT_FMT, user.num_embeddings))

                    # Pad embeddings to EMB_PER_USER size
                    padded_embeddings = np.zeros((EMB_PER_USER, EMB_DIM), dtype=EMB_DTYPE)
                    padded_embeddings[:user.num_embeddings] = user.embeddings
                    f.write(padded_embeddings.astype(EMB_DTYPE).tobytes())

                    # Pad images to EMB_PER_USER size
                    padded_images = [bytes(IMAGE_BYTES) for _ in range(EMB_PER_USER)]
                    for i, img in enumerate(user.images[:user.num_embeddings]):
                        padded_images[i] = bytes(img)
                    f.write(b"".join(padded_images))

                    print(f"Inserted user {user.id} at slot {slot} with {user.num_embeddings} embeddings")
                    return slot

        raise RuntimeError("Database FULL")

    def load_all(self) -> List[User]:
        users: List[User] = []

        if not os.path.exists(self.path):
            return users

        with open(self.path, "rb") as f:
            for slot in range(MAX_USERS):
                offset = slot * USER_RECORD_SIZE
                f.seek(offset)

                valid = struct.unpack(VALID_FMT, f.read(VALID_BYTES))[0]
                if valid == 0:
                    continue

                raw_id = f.read(ID_BYTES)
                user_id = self._decode_id(raw_id)

                num_embeddings = struct.unpack(COUNT_FMT, f.read(COUNT_BYTES))[0]

                emb_raw = f.read(EMB_BYTES)
                embeddings_full = np.frombuffer(emb_raw, dtype=EMB_DTYPE).reshape(
                    EMB_PER_USER, EMB_DIM
                )
                embeddings = embeddings_full[:num_embeddings]

                img_raw = f.read(IMG_BLOCK_BYTES)
                images = []
                for i in range(num_embeddings):
                    start = i * IMAGE_BYTES
                    end = start + IMAGE_BYTES
                    images.append(img_raw[start:end])

                users.append(User(user_id, embeddings, images))

        return users

    def load_for_hnsw(self) -> Dict[str, np.ndarray]:
        user_embeddings: Dict[str, np.ndarray] = {}

        if not os.path.exists(self.path):
            return user_embeddings

        with open(self.path, "rb") as f:
            for slot in range(MAX_USERS):
                offset = slot * USER_RECORD_SIZE
                f.seek(offset)

                valid = struct.unpack(VALID_FMT, f.read(VALID_BYTES))[0]
                if valid == 0:
                    continue

                raw_id = f.read(ID_BYTES)
                user_id = self._decode_id(raw_id)

                num_embeddings = struct.unpack(COUNT_FMT, f.read(COUNT_BYTES))[0]

                emb_raw = f.read(EMB_BYTES)
                embeddings_full = np.frombuffer(emb_raw, dtype=EMB_DTYPE).reshape(
                    EMB_PER_USER, EMB_DIM
                )
                embeddings = embeddings_full[:num_embeddings]

                user_embeddings[user_id] = embeddings

                f.seek(IMG_BLOCK_BYTES, 1)

        return user_embeddings
    
    def addEmbtoUser(self, user_id: str, new_embedding: np.ndarray, image_bytes: bytes) -> bool:
        if not os.path.exists(self.path):
            raise RuntimeError("Database does not exist")

        if new_embedding.shape != (EMB_DIM,) or new_embedding.dtype != np.dtype(EMB_DTYPE):
            raise ValueError("Invalid embedding shape or dtype")
        if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) != IMAGE_BYTES:
            raise ValueError("Invalid image bytes")

        with open(self.path, "r+b") as f:
            for slot in range(MAX_USERS):
                offset = slot * USER_RECORD_SIZE
                f.seek(offset)

                valid = struct.unpack(VALID_FMT, f.read(VALID_BYTES))[0]
                if valid == 0:
                    continue

                raw_id = f.read(ID_BYTES)
                current_user_id = self._decode_id(raw_id)

                if current_user_id == user_id:
                    num_embeddings = struct.unpack(COUNT_FMT, f.read(COUNT_BYTES))[0]

                    if num_embeddings >= EMB_PER_USER:
                        print(f"User {user_id} already has maximum embeddings")
                        return False

                    emb_offset = offset + VALID_BYTES + ID_BYTES + COUNT_BYTES + (num_embeddings * EMB_DIM * np.dtype(EMB_DTYPE).itemsize)
                    f.seek(emb_offset)
                    f.write(new_embedding.astype(EMB_DTYPE).tobytes())

                    img_offset = offset + VALID_BYTES + ID_BYTES + COUNT_BYTES + EMB_BYTES + (num_embeddings * IMAGE_BYTES)
                    f.seek(img_offset)
                    f.write(bytes(image_bytes))

                    f.seek(offset + VALID_BYTES + ID_BYTES)
                    f.write(struct.pack(COUNT_FMT, num_embeddings + 1))

                    print(f"Added embedding to user {user_id}, now has {num_embeddings + 1} embeddings")
                    return True

        print(f"User {user_id} not found in database")
        return False

def user_count() -> int:
    if not os.path.exists(db_path):
        raise RuntimeError("Database does not exist")
    count = 0

    with open(db_path, "rb") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)

            valid = struct.unpack(VALID_FMT, f.read(VALID_BYTES))[0]
            if valid != 0:
                count += 1
    return count


class INFO:
    def __init__(self, userid, lastseen):
        self.userid = userid
        self.lastseen = lastseen


def getNumberOfEmbeddings(user_id):
    if not os.path.exists(db_path):
        raise RuntimeError("Database does not exist")

    with open(db_path, "rb") as f:
        for slot in range(MAX_USERS):
            offset = slot * USER_RECORD_SIZE
            f.seek(offset)

            valid = struct.unpack(VALID_FMT, f.read(VALID_BYTES))[0]
            if valid == 0:
                continue

            raw_id = f.read(ID_BYTES)
            current_user_id = raw_id.rstrip(b"\x00").decode("utf-8", errors="ignore")

            if current_user_id == user_id:
                num_embeddings = struct.unpack(COUNT_FMT, f.read(COUNT_BYTES))[0]
                return num_embeddings

    return 0