from __future__ import annotations

import time
from io import BytesIO
from tempfile import NamedTemporaryFile
from pathlib import Path

import cv2
import json
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from database_helper import (
    _get_user_image_array,
    _iter_user_ids,
    get_user_embedding_count,
    _replace_embedding_from_image,
    _replace_user_image_from_file,
    dropBYID,
    remove_embedding,
    register_to_dataset,
)
from database import Database

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

import paths
paths.bootstrap_dirs()

app = Flask(__name__, template_folder="templates")
app.secret_key = "db-browser"

DB_PATH = paths.DATABASE_PATH


def _ensure_db_exists() -> None:
    if not DB_PATH:
        return
    db = Database(DB_PATH)
    if not db.exists():
        db.create()


def _wrap_index(index: int, total: int) -> int:
    if total <= 0:
        return 0
    return index % total


def _save_upload_to_temp(file_storage) -> str | None:
    if file_storage is None or file_storage.filename == "":
        return None
    suffix = ""
    if "." in file_storage.filename:
        suffix = "." + file_storage.filename.rsplit(".", 1)[-1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_storage.read())
        return tmp.name


@app.route("/")
def index():
    _ensure_db_exists()
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return render_template("db_browser.html", empty=True)
    return redirect(url_for("user_view", index=0))


@app.route("/user/<int:index>")
def user_view(index: int):
    _ensure_db_exists()
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return render_template("db_browser.html", empty=True)
    original_index = index
    index = _wrap_index(index, len(user_ids))
    if index != original_index:
        return redirect(url_for("user_view", index=index))
    user_id = user_ids[index]
    emb_per_user = int(cfg.get("EMB_PER_USER", 1))
    actual_count = max(1, min(emb_per_user, get_user_embedding_count(user_id, DB_PATH)))
    image_indices = list(range(actual_count))
    status = request.args.get("status")
    error = request.args.get("error")
    return render_template(
        "db_browser.html",
        empty=False,
        user_id=user_id,
        index=index,
        total=len(user_ids),
        image_indices=image_indices,
        max_embedding_index=emb_per_user,
        images_per_row=5,
        status=status,
        error=error,
        ts=int(time.time()),
    )


@app.route("/user/<user_id>/image/<int:image_index>")
def user_image(user_id: str, image_index: int):
    _ensure_db_exists()
    img = _get_user_image_array(user_id, image_index, DB_PATH)
    if img is None:
        abort(404)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        abort(500)
    return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg", max_age=0)


@app.route("/user/<int:index>/drop", methods=["POST"])
def drop_user(index: int):
    _ensure_db_exists()
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return redirect(url_for("index"))
    index = _wrap_index(index, len(user_ids))
    user_id = user_ids[index]
    dropBYID(user_id, DB_PATH)
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return redirect(url_for("index"))
    index = _wrap_index(index, len(user_ids))
    return redirect(url_for("user_view", index=index))


@app.route("/user/<int:index>/remove-embedding", methods=["POST"])
def remove_embedding_action(index: int):
    _ensure_db_exists()
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return redirect(url_for("index"))
    index = _wrap_index(index, len(user_ids))
    user_id = user_ids[index]
    try:
        embedding_no = int(request.form.get("embedding_no", ""))
        remove_embedding(user_id, embedding_no - 1, DB_PATH)
    except Exception:
        pass
    return redirect(url_for("user_view", index=index))


@app.route("/user/<int:index>/replace-embedding", methods=["POST"])
def replace_embedding_action(index: int):
    _ensure_db_exists()
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return redirect(url_for("index"))
    index = _wrap_index(index, len(user_ids))
    user_id = user_ids[index]
    try:
        embedding_no = int(request.form.get("embedding_no", ""))
        file = request.files.get("image")
        temp_path = _save_upload_to_temp(file)
        if temp_path:
            _replace_embedding_from_image(user_id, embedding_no - 1, temp_path, DB_PATH)
            _replace_user_image_from_file(user_id, temp_path, embedding_no - 1, DB_PATH)
    except Exception:
        pass
    return redirect(url_for("user_view", index=index))


@app.route("/add-user", methods=["POST"])
def add_user_action():
    _ensure_db_exists()
    files = request.files.getlist("folder")
    if not files:
        return redirect(url_for("index"))

    temp_root = Path("/tmp") / f"db_add_{int(time.time())}"
    temp_root.mkdir(parents=True, exist_ok=True)
    top_folder = None
    for file in files:
        if not file or not file.filename:
            continue
        rel_path = Path(file.filename)
        if top_folder is None and len(rel_path.parts) > 0:
            top_folder = rel_path.parts[0]
        target = temp_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        file.save(target)

    image_files = []
    for p in temp_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif"}:
            image_files.append(p)

    if len(image_files) < 1:
        user_ids = _iter_user_ids(DB_PATH)
        if not user_ids:
            return redirect(url_for("index"))
        target_index = len(user_ids) - 1
        return redirect(
            url_for(
                "user_view",
                index=target_index,
                error=f"Need at least 1 image; found {len(image_files)}.",
            )
        )

    db = Database()
    db.create()
    user_folder = temp_root / top_folder if top_folder else temp_root
    ok, message = register_to_dataset(db, str(user_folder))
    user_ids = _iter_user_ids(DB_PATH)
    if not user_ids:
        return render_template(
            "db_browser.html",
            empty=True,
            status=message if ok else None,
            error=None if ok else message,
        )
    target_index = len(user_ids) - 1
    if ok:
        return redirect(url_for("user_view", index=target_index, status=message))
    return redirect(url_for("user_view", index=target_index, error=message))


if __name__ == "__main__":
    app.run(host=paths.EDITOR_HOST, port=paths.EDITOR_PORT, debug=False)
