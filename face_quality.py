"""
Face validation + quality scoring + best-face bookkeeping.

This module sits BETWEEN MediaPipe Face Landmarker detection and ArcFace
embedding extraction. It does three jobs:

  1. validate_face()            — reject false / hallucinated MediaPipe faces
                                   (back-of-head, "completed" half-faces, etc).
  2. score_face()               — score a *valid* face on frontality, sharpness,
                                   size, eye-openness, symmetry and confidence.
  3. BestFaceTracker /          — remember the best-scoring frame per identified
     save_best_face_results()     person and write the results into Testing/.

Nothing here touches ArcFace. ArcFace stays the identity matcher; the scoring
below is purely geometric / image based.
"""

import os
import math
import shutil
from collections import defaultdict

import cv2
import numpy as np

# ─── MediaPipe Face Landmarker landmark indices ──────────────────────────────
NOSE_TIP        = 1
LEFT_EAR        = 234
RIGHT_EAR       = 454
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER  = 133
RIGHT_EYE_INNER = 362
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM = 374
CHIN            = 152
LEFT_CHEEK      = 123
RIGHT_CHEEK     = 352
LEFT_MOUTH      = 61
RIGHT_MOUTH     = 291

# Project root — anchors the Testing/ folder regardless of the cwd we run from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TESTING_DIR  = os.path.join(PROJECT_ROOT, "Testing")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 1 — Face validation layer
# ══════════════════════════════════════════════════════════════════════════════
def validate_face(face_landmarks):
    """Run every false-detection check on a MediaPipe face.

    Returns ``(True, None)`` if the face passes ALL checks, otherwise
    ``(False, reason)`` where ``reason`` is a short string for logging.

    Landmarks are MediaPipe's normalised coordinates (x, y in 0..1).
    """
    if face_landmarks is None:
        return False, "no_landmarks"

    try:
        nose = face_landmarks[NOSE_TIP]

        # ── Check 1: yaw angle (most important) ──────────────────────────────
        left_ear = face_landmarks[LEFT_EAR]
        right_ear = face_landmarks[RIGHT_EAR]
        ear_width = abs(right_ear.x - left_ear.x)
        if ear_width < 0.005:
            return False, "yaw_ear_width"   # can't see both ears
        nose_ratio = (nose.x - left_ear.x) / ear_width
        if nose_ratio < 0.15 or nose_ratio > 0.85:
            return False, "yaw_ratio"       # turned too far / back of head

        # ── Check 2: eye distance sanity ─────────────────────────────────────
        left_eye = face_landmarks[LEFT_EYE_OUTER]
        right_eye = face_landmarks[RIGHT_EYE_OUTER]
        eye_dist = math.hypot(right_eye.x - left_eye.x, right_eye.y - left_eye.y)
        if eye_dist < 0.015:
            return False, "eye_distance"    # collapsed mesh on back of head

        # ── Check 3: facial geometry order ───────────────────────────────────
        eye_center_y = (left_eye.y + right_eye.y) / 2.0
        chin = face_landmarks[CHIN]
        if not (nose.y > eye_center_y):
            return False, "geometry_nose_above_eyes"
        if not (chin.y > nose.y):
            return False, "geometry_chin_above_nose"

        # ── Check 4: symmetry (hallucinated half-faces) ──────────────────────
        left_cheek = face_landmarks[LEFT_CHEEK]
        right_cheek = face_landmarks[RIGHT_CHEEK]
        d_left = abs(nose.x - left_cheek.x)
        d_right = abs(nose.x - right_cheek.x)
        mn = min(d_left, d_right)
        if mn < 1e-6:
            return False, "symmetry"
        if (max(d_left, d_right) / mn) > 3.5:
            return False, "symmetry"        # side face MediaPipe "completed"

        return True, None
    except Exception as e:
        return False, f"exception:{e}"


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 2 — Quality scoring
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS = {
    "frontality":   0.30,
    "sharpness":    0.25,
    "face_size":    0.15,
    "eye_openness": 0.10,
    "symmetry":     0.10,
    "confidence":   0.10,
}


class FaceQualityScore:
    """One scored face detection from one frame."""

    def __init__(self, frame_no, total, breakdown, face_bbox, angles):
        self.frame_no = frame_no
        self.total = total                # 0.0 .. 1.0
        self.breakdown = breakdown        # dict: metric -> score
        self.face_bbox = face_bbox        # (x, y, w, h) tight, pixel coords
        self.angles = angles              # (yaw, pitch, roll) in degrees


def _frontality(face_landmarks):
    """Score = exp(-yaw²/450)·exp(-pitch²/800)·exp(-roll²/800). Returns
    (score, (yaw, pitch, roll))."""
    nose = face_landmarks[NOSE_TIP]
    left_ear = face_landmarks[LEFT_EAR]
    right_ear = face_landmarks[RIGHT_EAR]
    left_eye = face_landmarks[LEFT_EYE_OUTER]
    right_eye = face_landmarks[RIGHT_EYE_OUTER]
    chin = face_landmarks[CHIN]

    # Yaw — from nose position between the ears (0.5 == centred).
    ear_width = abs(right_ear.x - left_ear.x)
    nose_ratio = 0.5 if ear_width < 1e-6 else (nose.x - left_ear.x) / ear_width
    yaw = (nose_ratio - 0.5) * 180.0

    # Pitch — from nose vertical position between the eye line and the chin.
    eye_center_y = (left_eye.y + right_eye.y) / 2.0
    span = chin.y - eye_center_y
    p = 0.5 if abs(span) < 1e-6 else (nose.y - eye_center_y) / span
    pitch = (p - 0.5) * 90.0

    # Roll — angle of the eye line.
    roll = math.degrees(math.atan2(right_eye.y - left_eye.y,
                                   right_eye.x - left_eye.x))

    score = (math.exp(-(yaw ** 2) / 450.0)
             * math.exp(-(pitch ** 2) / 800.0)
             * math.exp(-(roll ** 2) / 800.0))
    return score, (yaw, pitch, roll)


def _sharpness(frame, face_bbox):
    """Laplacian variance of the face crop, normalised to 0..1."""
    x, y, w, h = face_bbox
    x = max(0, x)
    y = max(0, y)
    crop = frame[y:y + h, x:x + w]
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return 1.0 - math.exp(-lap_var / 150.0)


def _face_size(face_bbox, frame_shape):
    """Face-area / frame-area, scored so 10% of the frame == 1.0."""
    _, _, w, h = face_bbox
    fh, fw = frame_shape[:2]
    frame_area = fh * fw
    if frame_area <= 0:
        return 0.0
    ratio = (w * h) / frame_area
    if ratio < 0.005:                     # < 0.5% of frame
        return 0.1
    return min(1.0, ratio / 0.10)


def _eye_aspect_ratio(face_landmarks, top, bottom, outer, inner):
    vertical = abs(face_landmarks[top].y - face_landmarks[bottom].y)
    horizontal = math.hypot(
        face_landmarks[outer].x - face_landmarks[inner].x,
        face_landmarks[outer].y - face_landmarks[inner].y,
    )
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _eye_openness(face_landmarks):
    """Average EAR of both eyes. EAR<=0.10 -> 0.1, EAR>=0.18 -> 1.0."""
    left = _eye_aspect_ratio(face_landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                             LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right = _eye_aspect_ratio(face_landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                              RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    ear = (left + right) / 2.0
    if ear <= 0.10:
        return 0.1
    if ear >= 0.18:
        return 1.0
    # Linear ramp between the closed (0.1) and open (1.0) end points.
    return 0.1 + (ear - 0.10) / (0.18 - 0.10) * (1.0 - 0.1)


def _symmetry(face_landmarks):
    """Average of cheek and mouth left/right balance about the nose."""
    nose = face_landmarks[NOSE_TIP]

    def balance(li, ri):
        d_left = abs(nose.x - face_landmarks[li].x)
        d_right = abs(nose.x - face_landmarks[ri].x)
        mx = max(d_left, d_right)
        return 0.0 if mx < 1e-6 else min(d_left, d_right) / mx

    cheek = balance(LEFT_CHEEK, RIGHT_CHEEK)
    mouth = balance(LEFT_MOUTH, RIGHT_MOUTH)
    return (cheek + mouth) / 2.0


def _confidence(face_landmarks):
    """Proxy for detection confidence: fraction of landmarks inside 0..1."""
    total = 0
    inside = 0
    for lm in face_landmarks:
        total += 1
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            inside += 1
    return 0.0 if total == 0 else inside / total


def score_face(face_landmarks, frame, face_bbox, frame_no=0):
    """Score a single valid face. ``face_bbox`` is (x, y, w, h) in pixels."""
    frontality, angles = _frontality(face_landmarks)
    breakdown = {
        "frontality":   frontality,
        "sharpness":    _sharpness(frame, face_bbox),
        "face_size":    _face_size(face_bbox, frame.shape),
        "eye_openness": _eye_openness(face_landmarks),
        "symmetry":     _symmetry(face_landmarks),
        "confidence":   _confidence(face_landmarks),
    }
    total = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)
    return FaceQualityScore(frame_no, total, breakdown, face_bbox, angles)


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM 2/3 — Best-face tracking + saving
# ══════════════════════════════════════════════════════════════════════════════
class BestFaceTracker:
    """Keeps the top-N scoring frames per identified user for one video."""

    def __init__(self, top_n=10):
        self.top_n = top_n
        self._cands = defaultdict(list)   # user_id -> [(FaceQualityScore, frame)]

    def add(self, user_id, score, frame):
        bucket = self._cands[user_id]
        bucket.append((score, frame.copy()))
        bucket.sort(key=lambda t: t[0].total, reverse=True)
        del bucket[self.top_n:]           # keep only the best top_n

    def best(self, user_id):
        bucket = self._cands[user_id]
        return bucket[0] if bucket else (None, None)

    def top_candidates(self, user_id):
        return [score for score, _ in self._cands[user_id]]

    def users(self):
        return list(self._cands.keys())

    def best_score_overall(self):
        best = 0.0
        for bucket in self._cands.values():
            if bucket:
                best = max(best, bucket[0][0].total)
        return best


def _write_quality_report(path, user_id, video_file, best_score, top_candidates):
    b = best_score.breakdown
    yaw, pitch, roll = best_score.angles
    lines = [
        f"User ID      : {user_id}",
        f"Video        : {video_file}",
        f"Best Frame   : #{best_score.frame_no}",
        f"Total Score  : {best_score.total:.4f}",
        "--------------------------",
        f"Frontality   : {b['frontality']:.4f} "
        f"(yaw: {yaw:.1f}°, pitch: {pitch:.1f}°, roll: {roll:.1f}°)",
        f"Sharpness    : {b['sharpness']:.4f}",
        f"Face Size    : {b['face_size']:.4f}",
        f"Eye Openness : {b['eye_openness']:.4f}",
        f"Symmetry     : {b['symmetry']:.4f}",
        f"Confidence   : {b['confidence']:.4f}",
        "--------------------------",
        "Top 5 Candidates:",
    ]
    for i, cand in enumerate(top_candidates[:5], start=1):
        lines.append(f"  #{i}: Frame {cand.frame_no}  | Score {cand.total:.4f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_best_face_results(user_id, video_path, best_score, best_frame,
                           top_candidates, testing_root=TESTING_DIR):
    """Write the best-face artefacts into Testing/{userId}_{videoName}/.

    Overwrites any previous results for the same user + video.
    """
    if best_score is None or best_frame is None:
        print(f"[BEST-FACE] No candidates to save for user {user_id}")
        return None

    video_file = os.path.basename(video_path)
    video_name = os.path.splitext(video_file)[0]
    folder_name = f"{user_id}_{video_name}"
    folder_path = os.path.join(testing_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Best frame (full).
    cv2.imwrite(os.path.join(folder_path, "best_face.jpg"), best_frame)

    # Face crop with 10% padding, clamped to frame edges.
    x, y, w, h = best_score.face_bbox
    fh, fw = best_frame.shape[:2]
    px, py = int(w * 0.10), int(h * 0.10)
    cx1, cy1 = max(0, x - px), max(0, y - py)
    cx2, cy2 = min(fw, x + w + px), min(fh, y + h + py)
    crop = best_frame[cy1:cy2, cx1:cx2]
    if crop.size > 0:
        cv2.imwrite(os.path.join(folder_path, "best_face_crop.jpg"), crop)

    # Copy the source video.
    try:
        shutil.copy2(video_path, os.path.join(folder_path, "source_video.mp4"))
    except Exception as e:
        print(f"[BEST-FACE] Could not copy source video for {user_id}: {e}")

    # Quality report.
    _write_quality_report(
        os.path.join(folder_path, "quality_report.txt"),
        user_id, video_file, best_score, top_candidates,
    )

    print(f"[BEST-FACE] Saved {folder_name} "
          f"(frame #{best_score.frame_no}, score {best_score.total:.4f})")
    return folder_path
