import time
import cv2 
import os
import numpy as np
import json

with open("config.json", "r") as f:
    config = json.load(f)

min_face_size = config["min_face_size"]

GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
BLUE = '\033[94m'


def valid_face_size(face_crop):
    global min_face_size
    h, w = face_crop.shape[:2]
    if w < min_face_size or h < min_face_size:
        return False
    aspect_ratio = w / (h + 1e-6)
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False

    return True

FACE_POSITION_CONSTANTS = {
    "MAX_ROLL_DEGREES": 80,
    "YAW_RATIO_MIN": 0.30,
    "YAW_RATIO_MAX": 1.5,
    "PITCH_RATIO_MIN": 0.2,
    "PITCH_RATIO_MAX": 2.8
}

def crop_frame(frame):
    h, w, _ = frame.shape
    top_crop = int(0.15 * h)
    right_crop = int(0.05 * w)
    cropped_frame = frame[top_crop:h, 0:w - right_crop]
    return cropped_frame


BRIGHTNESS_THRESHOLDS = {
    'MIN_BRIGHTNESS': 30,   
    'MAX_BRIGHTNESS': 220,  
    'MIN_CONTRAST_STD': 18.5
}

BLUR_THRESHOLD = 30 

def is_face_brightness_ok(face_image):
    if face_image is None or face_image.size == 0:
        return False
    
    if face_image.dtype != np.uint8:
        face_image = np.clip(face_image, 0, 255).astype(np.uint8)
    
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    
    if mean_brightness > BRIGHTNESS_THRESHOLDS['MAX_BRIGHTNESS']:
        return False
    
    if mean_brightness < BRIGHTNESS_THRESHOLDS['MIN_BRIGHTNESS']:
        return False
    
    if std_brightness < BRIGHTNESS_THRESHOLDS['MIN_CONTRAST_STD']:  
        return False
    
    return True
import cv2
import numpy as np
from typing import Optional

def is_blurry(
    image_region: np.ndarray,
    threshold: float = 100.0,
    method: str = 'laplacian'
) -> Optional[bool]:
    if image_region is None or not isinstance(image_region, np.ndarray):
        return None
    
    if len(image_region.shape) < 2:
        return None
    
    if image_region.size == 0:
        return None
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_region
    if method == 'laplacian':
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
    elif method == 'gradient':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        blur_score = (sobelx**2 + sobely**2).var()
        threshold *= 2  
    else:
        raise ValueError(f"Unknown method: {method}")
    return blur_score < threshold

def rotate_frame(frame, angle = 180):
    if frame is None:
        return frame
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated

def is_face_in_cropped_frame(x1, y1, x2, y2, cropped_frame, min_ratio=0.6):
    if cropped_frame is None:
        return False
    cropped_h, cropped_w, _ = cropped_frame.shape
    face_w = x2 - x1
    face_h = y2 - y1
    face_area = face_w * face_h

    if face_area <= 0:
        return False
    overlap_x1 = max(x1, 0)
    overlap_x2 = min(x2, cropped_w)
    overlap_w = max(0, overlap_x2 - overlap_x1)

    overlap_area = overlap_w * face_h
    overlap_ratio = overlap_area / face_area

    return overlap_ratio >= min_ratio


def scale_face_3x(face_img):
    if face_img is None or face_img.size == 0:
        return None

    h, w = face_img.shape[:2]

    scaled_face = cv2.resize(
        face_img,
        (w * 3, h * 3),
        interpolation=cv2.INTER_CUBIC
    )

    return scaled_face


UPSCALE = 5
def upscale(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * UPSCALE, h * UPSCALE),
                       interpolation=cv2.INTER_CUBIC)

def compute_quality(face_landmarks):
    pts = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks])
    spread = np.std(pts[:, :2])
    score = min(1.0, spread * 10)

    return score


def validFacePosition_mediapipe(face_landmarks, frame_shape):
    if face_landmarks is None:
        return False

    try:
        h, w = frame_shape[:2]
        def pt(i):
            return np.array([
                face_landmarks[i].x * w,
                face_landmarks[i].y * h
            ], dtype=np.float32)

        left_eye = pt(33)
        right_eye = pt(263)
        nose = pt(1)
        left_mouth = pt(61)
        right_mouth = pt(291)
        for x, y in [left_eye, right_eye, nose, left_mouth, right_mouth]:
            if not (0 <= x <= w and 0 <= y <= h):
                return False
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]

        roll_degrees = abs(np.degrees(np.arctan2(dy, dx)))
        if roll_degrees > FACE_POSITION_CONSTANTS['MAX_ROLL_DEGREES']:
            return False

        def dist(a, b):
            return np.linalg.norm(a - b)

        left_eye_nose = dist(left_eye, nose)
        right_eye_nose = dist(right_eye, nose)

        yaw_ratio = left_eye_nose / (right_eye_nose + 1e-6)
        if not (FACE_POSITION_CONSTANTS['YAW_RATIO_MIN'] <= yaw_ratio <= FACE_POSITION_CONSTANTS['YAW_RATIO_MAX']):
            return False

        eye_mid = (left_eye[1] + right_eye[1]) / 2
        mouth_mid = (left_mouth[1] + right_mouth[1]) / 2

        pitch_ratio = (nose[1] - eye_mid) / (mouth_mid - nose[1] + 1e-6)
        if not (FACE_POSITION_CONSTANTS['PITCH_RATIO_MIN'] <= pitch_ratio <= FACE_POSITION_CONSTANTS['PITCH_RATIO_MAX']):
            return False
        return True

    except Exception as e:
        return True  
    


def cropframe(frame):
    roi = config["roi"]
    h, w = frame.shape[:2]
    x1 = int(roi["x"] * w)
    y1 = int(roi["y"] * h)
    x2 = int((roi["x"] + roi["width"]) * w)
    y2 = int((roi["y"] + roi["height"]) * h)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return frame 

    cropped = frame[y1:y2, x1:x2]

    return cropped