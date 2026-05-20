import os
os.environ.setdefault("GLOG_minloglevel", "2") 
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_LOGGING_CPP_MIN_LOG_LEVEL", "2")
import sys
import time
import cv2
from database import INFO
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis
from unknown import  unknown_handler 
from database import Database, getNumberOfEmbeddings

from checks import (
    is_blurry,
    validFacePosition_mediapipe,
    is_face_brightness_ok,
    compute_quality
)
from log import write_log
import json
from collections import defaultdict
from pathlib import Path

with open("config.json", "r") as f:
    config = json.load(f)

import paths

knownThreshold = config["threshold"]
MODEL_PATH = paths.MODEL_PATH
print_debug_info = True
min_face_size = config["min_face_size"]
knownlogpath = paths.KNOWN_LOG_FILE
debuglogpath = paths.DEBUG_LOG_FILE
noface_dir = paths.NOFACE_DIR
debug_fail_dir = paths.DEBUG_FAIL_DIR
known_detect_dir = paths.KNOWN_DETECT_DIR
EMB_PER_USER = config["EMB_PER_USER"]
EMB_DIM = config["EMB_DIM"]

_logged_users_by_video = defaultdict(set)


def _should_log_user(video_name: str, user_id: str) -> bool:
    if not video_name:
        return True
    if user_id in _logged_users_by_video[video_name]:
        return False
    _logged_users_by_video[video_name].add(user_id)
    return True
IMG_H = config.get("IMG_H", 128)
IMG_W = config.get("IMG_W", 128)
IMG_C = config.get("IMG_C", 3)
IMAGE_BYTES = config.get("IMAGE_BYTES", IMG_H * IMG_W * IMG_C)

sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

app = FaceAnalysis(name="buffalo_l")  
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=5,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
)

landmarker = None
if os.path.exists(MODEL_PATH):
    try:
        landmarker = FaceLandmarker.create_from_options(options)
        print("[INFO] MediaPipe Face Landmarker initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize landmarker: {e}")
        landmarker = None
else:
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    landmarker = None


class userData:
    def __init__(self):
        self.id = None
        self.confidence = 0.0


def _safe_filename(video_name: str, frame_no: int) -> str:
    safe_video = Path(video_name).stem if video_name else "unknown"
    safe_video = safe_video.replace(" ", "_")
    return f"{safe_video}_F{frame_no}.jpg"


def _save_image(folder: str, video_name: str, frame_no: int, image: np.ndarray) -> None:
    os.makedirs(folder, exist_ok=True)
    filename = _safe_filename(video_name, frame_no)
    path = os.path.join(folder, filename)
    try:
        cv2.imwrite(path, image)
    except Exception:
        pass


def _save_debug_failure(failure: str, video_name: str, frame_no: int, image: np.ndarray) -> None:
    folder = os.path.join(debug_fail_dir, failure)
    _save_image(folder, video_name, frame_no, image)


def _to_image_bytes(frame: np.ndarray) -> bytes:
    resized = cv2.resize(frame, (IMG_W, IMG_H))
    if resized.shape[2] != IMG_C:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized.tobytes()


def align_face_for_arcface(face_crop, face_landmarks, x1_p, y1_p, w_img, h_img):
    """Align face using 5-point landmarks for better ArcFace detection"""
    try:
        # Get key landmarks (left eye, right eye, nose, left mouth, right mouth)
        # MediaPipe landmark indices
        LEFT_EYE = 33   # or 133
        RIGHT_EYE = 263  # or 362
        NOSE = 1
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291
        
        landmarks_5 = np.array([
            [face_landmarks[LEFT_EYE].x * w_img - x1_p, 
             face_landmarks[LEFT_EYE].y * h_img - y1_p],
            [face_landmarks[RIGHT_EYE].x * w_img - x1_p, 
             face_landmarks[RIGHT_EYE].y * h_img - y1_p],
            [face_landmarks[NOSE].x * w_img - x1_p, 
             face_landmarks[NOSE].y * h_img - y1_p],
            [face_landmarks[LEFT_MOUTH].x * w_img - x1_p, 
             face_landmarks[LEFT_MOUTH].y * h_img - y1_p],
            [face_landmarks[RIGHT_MOUTH].x * w_img - x1_p, 
             face_landmarks[RIGHT_MOUTH].y * h_img - y1_p]
        ], dtype=np.float32)
        
        # Standard reference points (for 112x112 face)
        ref_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Scale reference to crop size
        scale = min(face_crop.shape[0], face_crop.shape[1]) / 112.0
        ref_points_scaled = ref_points * scale
        
        # Estimate affine transform
        tform = cv2.estimateAffinePartial2D(landmarks_5, ref_points_scaled)[0]
        
        # Apply alignment
        aligned = cv2.warpAffine(face_crop, tform, 
                                 (face_crop.shape[1], face_crop.shape[0]),
                                 flags=cv2.INTER_LINEAR)
        return aligned
    except:
        return face_crop  # Return original if alignment fails

visulize = False

def recognize(name, frame, knownFaiss , db : Database, frame_no: int = 0, timestamp: str = "", video_name: str = ""):
    global knownThreshold
    if landmarker is None:
        return frame
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    except Exception:
        frame_bgr = np.ascontiguousarray(frame)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)

    result = landmarker.detect(mp_img)
    n_frame = frame.copy()
    
    if not result.face_landmarks:
        if print_debug_info:
            print(f"[INFO] {name}: No faces detected")
        write_log(debuglogpath, timestamp, frame_no, "No faces detected", video_name)
        _save_image(noface_dir, video_name, frame_no, frame)
        return n_frame
    
    h_img, w_img = frame.shape[:2]

    for face_landmarks in result.face_landmarks:
        xs = [lm.x for lm in face_landmarks]
        ys = [lm.y for lm in face_landmarks]
        x1 = int(min(xs) * w_img)
        x2 = int(max(xs) * w_img)
        y1 = int(min(ys) * h_img)
        y2 = int(max(ys) * h_img)
        
        # Increased padding for better ArcFace detection
        pad = 0.35  # Increased from 0.25
        bw, bh = x2 - x1, y2 - y1
        x1_p = max(0, x1 - int(bw * pad))
        y1_p = max(0, y1 - int(bh * pad))
        x2_p = min(w_img, x2 + int(bw * pad))
        y2_p = min(h_img, y2 + int(bh * pad))
    
        face_crop = frame[y1_p:y2_p, x1_p:x2_p]
        
        quality = compute_quality(face_landmarks)
        pos_valid = validFacePosition_mediapipe(face_landmarks, frame.shape)
        
        if print_debug_info:
            print(f"[DEBUG] {name}: Face position check - {'PASS' if pos_valid else 'FAIL'} (quality: {quality:.3f})")
        
        if not pos_valid:
            write_log(debuglogpath, timestamp, frame_no, "Face position check failed", video_name)
            _save_debug_failure("position", video_name, frame_no, face_crop)
            continue
        
        h_c, w_c = face_crop.shape[:2]
        size_valid = (w_c >= min_face_size) and (h_c >= min_face_size)
        if not size_valid:
            write_log(debuglogpath, timestamp, frame_no, "Face size check failed", video_name)
            _save_debug_failure("size", video_name, frame_no, face_crop)
            if print_debug_info:
                print(f"[DEBUG] {name}: Face size check - FAIL (crop size: {w_c}x{h_c})")
            continue
        
        # not_blurry = is_blurry(face_crop)
        # if not not_blurry:
        #     if print_debug_info:
        #         print(f"[DEBUG] {name}: Face blurry check - FAIL")
        #     write_log(debuglogpath, timestamp, frame_no, "Face blur check failed", video_name)
        #     _save_debug_failure("blur", video_name, frame_no, face_crop)
        #     continue
        
        brightness_ok = is_face_brightness_ok(face_crop)
        if not brightness_ok:
            write_log(debuglogpath, timestamp, frame_no, "Face brightness check failed", video_name)
            _save_debug_failure("brightness", video_name, frame_no, face_crop)
            if print_debug_info:
                print(f"[DEBUG] {name}: Face brightness check - FAIL")
            continue

        faces_arc = []
        
        faces_arc = app.get(face_crop)
        
        if len(faces_arc) == 0:
            aligned_crop = align_face_for_arcface(face_crop, face_landmarks, x1_p, y1_p, w_img, h_img)
            faces_arc = app.get(aligned_crop)
            if len(faces_arc) > 0 and print_debug_info:
                print(f"[INFO] {name}: ArcFace succeeded with alignment")
        
        if len(faces_arc) == 0:
            enhanced = cv2.convertScaleAbs(face_crop, alpha=1.2, beta=10)
            faces_arc = app.get(enhanced)
            if len(faces_arc) > 0 and print_debug_info:
                print(f"[INFO] {name}: ArcFace succeeded with enhancement")
        
        if len(faces_arc) == 0:
            target_size = 224
            resized = cv2.resize(face_crop, (target_size, target_size), 
                                interpolation=cv2.INTER_CUBIC)
            faces_arc = app.get(resized)
            if len(faces_arc) > 0 and print_debug_info:
                print(f"[INFO] {name}: ArcFace succeeded with resize")

        if len(faces_arc) == 0:
            if print_debug_info:
                print(f"[WARNING] {name}: ArcFace FAILED (all strategies)")
            write_log(debuglogpath, timestamp, frame_no, "ArcFace failed", video_name)
            _save_debug_failure("arcface", video_name, frame_no, face_crop)
            continue

        face_embedding = faces_arc[0].embedding
        
        if print_debug_info:
            norm = np.linalg.norm(face_embedding)
            print(f"[SUCCESS] ArcFace embedding extracted. Norm={norm:.4f}")
        
        log_ts = timestamp
        uid, best_score = knownFaiss.search_faiss_best(face_embedding)
        if uid is not None and best_score >= knownThreshold :
            # flag = True
            # for user in users:
            #     if user.userid == uid:
            #         if time.time() - user.lastseen > config["cooling_period"]:
            #             user.lastseen = time.time()
            #             write_log(knownlogpath, f"{uid} -- {best_score:.4f}")
            #         flag = False
            #         break
            # if flag:
            #     users.append(INFO(uid, time.time()))
            should_log = _should_log_user(video_name, uid)
            if getNumberOfEmbeddings(uid) < EMB_PER_USER and face_embedding is not None  and face_embedding.shape == (EMB_DIM,):
                img_bytes = _to_image_bytes(face_crop)
                if len(img_bytes) == IMAGE_BYTES:
                    db.addEmbtoUser(uid, face_embedding, img_bytes)
                if should_log:
                    write_log(knownlogpath, log_ts, frame_no, f"{uid} -- {best_score:.2f} (embedding added)", video_name)
                knownFaiss.build_faiss(db.load_for_hnsw())
            
            if should_log:
                write_log(knownlogpath, log_ts, frame_no, f"{uid} -- {best_score:.2f}", video_name)
            user_folder = os.path.join(known_detect_dir, uid)
            _save_image(user_folder, video_name, frame_no, face_crop)
            user_log = os.path.join(user_folder, "log.txt")
            if should_log:
                write_log(user_log, log_ts, frame_no, f"{uid} -- {best_score:.2f}", video_name)
            color = (0, 255, 0)   
            label = f"{uid} ({best_score:.3f})"
        else:
            uid = unknown_handler(face_crop, face_embedding, threshold=knownThreshold, timestamp=log_ts, frame_no=frame_no, video_name=video_name)
            best_score = 0.0 if uid is None else best_score
            color = (0, 0, 255)   # RED (unknown)
            label = f"{uid}"

        if visulize:
            # Draw bounding box
            cv2.rectangle(n_frame, (x1_p, y1_p), (x2_p, y2_p), color, 2)

            # Draw label background (for clean text)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(n_frame, (x1_p, y1_p - th - 10), (x1_p + tw, y1_p), color, -1)
            # Put text
            cv2.putText(
                n_frame,
                label,
                (x1_p, y1_p - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            # ---------------- LANDMARKS ----------------
            for lm in face_landmarks:
                lx = int(lm.x * w_img)
                ly = int(lm.y * h_img)

                cv2.circle(n_frame, (lx, ly), 1, color, -1)

    if visulize:
        display_frame = cv2.resize(n_frame, (960, 540))
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 960, 540)
        cv2.imshow(name, display_frame)
        cv2.waitKey(1)
             
            
                

def getEmbedding(name, frame):
    if landmarker is None:
        print(f"[ERROR] {name}: Landmarker not initialized")
        return None
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    except Exception:
        frame_bgr = np.ascontiguousarray(frame)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)

    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        print(f"[FAIL-1] {name}: No faces detected")
        return None

    h_img, w_img = frame.shape[:2]
    face_landmarks = result.face_landmarks[0]

    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]

    x1 = int(min(xs) * w_img)
    x2 = int(max(xs) * w_img)
    y1 = int(min(ys) * h_img)
    y2 = int(max(ys) * h_img)

    pad = 0.5
    bw, bh = x2 - x1, y2 - y1

    x1_p = max(0, x1 - int(bw * pad))
    y1_p = max(0, y1 - int(bh * pad))
    x2_p = min(w_img, x2 + int(bw * pad))
    y2_p = min(h_img, y2 + int(bh * pad))

    face_crop = frame[y1_p:y2_p, x1_p:x2_p]

    if face_crop.size == 0:
        return None
    quality = compute_quality(face_landmarks)
    quality_threshold = config.get("quality_threshold", 0.5)
    if True:
        print(f"[DEBUG] {name}: Quality score = {quality:.3f}, threshold = {quality_threshold}")

    if quality < quality_threshold:
        print(f"[FAIL] {name}: Face rejected due to LOW QUALITY (score={quality:.3f})")
        return None
    pos_valid = validFacePosition_mediapipe(face_landmarks, frame.shape)

    if not pos_valid:
        print(f"[FAIL] {name}: Invalid face position")
        return None

    h_c, w_c = face_crop.shape[:2]
    if w_c < min_face_size or h_c < min_face_size:
        print(f"[FAIL] {name}: Face too small ({w_c}x{h_c}, min={min_face_size})")
        return None
    print(f"[DEBUG] {name}: Face crop size = {w_c}x{h_c}")
    # if is_blurry(face_crop):
    #     print(f"[FAIL] {name}: Face is blurry")
    #     return None
    # if not is_face_brightness_ok(face_crop):
    #     print(f"[FAIL] {name}: Face brightness check failed")
    #     return None
    
    target_size = 224
    face_crop_resized = cv2.resize(face_crop, (target_size, target_size))
    
    face_crop_rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
    
    face_crop_rgb_eq = np.zeros_like(face_crop_rgb)
    for i in range(3):
        face_crop_rgb_eq[:, :, i] = cv2.equalizeHist(face_crop_rgb[:, :, i])
    
    try:
        faces_arc = app.get(face_crop_rgb_eq)
    except Exception as e:
        print(f"[ERROR] {name}: ArcFace exception: {e}")
        return None

    if len(faces_arc) == 0:
        print(f"[WARNING] {name}: ArcFace failed on resized crop, trying original size")
        try:
            face_crop_rgb_orig = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces_arc = app.get(face_crop_rgb_orig)
        except Exception as e:
            print(f"[ERROR] {name}: ArcFace exception on fallback: {e}")
            return None
        
        if len(faces_arc) == 0:
            print(f"[WARNING] {name}: ArcFace failed on original crop, trying full frame")
            # Last resort: try full frame
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply histogram equalization to frame
                frame_rgb_eq = np.zeros_like(frame_rgb)
                for i in range(3):
                    frame_rgb_eq[:, :, i] = cv2.equalizeHist(frame_rgb[:, :, i])
                faces_arc = app.get(frame_rgb_eq)
            except Exception as e:
                print(f"[ERROR] {name}: ArcFace exception on full frame: {e}")
                return None
            
            if len(faces_arc) == 0:
                print(f"[WARNING] {name}: ArcFace could not detect face in any format")
                return None

    face_embedding = faces_arc[0].embedding

    return face_embedding , quality


