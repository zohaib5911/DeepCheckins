#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp


CONFIG_PATH = Path(__file__).with_name("config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MODEL_PATH = config["model_path"]
FACE_POSITION_CONSTANTS = {
    "MAX_ROLL_DEGREES": 80,
    "YAW_RATIO_MIN": 0.30,
    "YAW_RATIO_MAX": 1.5,
    "PITCH_RATIO_MIN": 0.2,
    "PITCH_RATIO_MAX": 2.8
}


def _load_landmarker():
    base_options = mp.tasks.BaseOptions
    face_landmarker = mp.tasks.vision.FaceLandmarker
    face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode

    options = face_landmarker_options(
        base_options=base_options(model_asset_path=MODEL_PATH),
        running_mode=running_mode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return face_landmarker.create_from_options(options)


def _pt(face_landmarks, idx: int, w: int, h: int) -> np.ndarray:
    return np.array([
        face_landmarks[idx].x * w,
        face_landmarks[idx].y * h,
    ], dtype=np.float32)


def _compute_ratios(face_landmarks, w: int, h: int) -> Tuple[float, float, float]:
    left_eye  = _pt(face_landmarks, 33,  w, h)
    right_eye = _pt(face_landmarks, 263, w, h)
    nose      = _pt(face_landmarks, 1,   w, h)
    left_mouth  = _pt(face_landmarks, 61,  w, h)
    right_mouth = _pt(face_landmarks, 291, w, h)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_degrees = abs(np.degrees(np.arctan2(dy, dx)))

    def dist(a, b):
        return np.linalg.norm(a - b)

    left_eye_nose  = dist(left_eye,  nose)
    right_eye_nose = dist(right_eye, nose)
    yaw_ratio      = left_eye_nose / (right_eye_nose + 1e-6)

    eye_mid   = (left_eye[1]   + right_eye[1])  / 2
    mouth_mid = (left_mouth[1] + right_mouth[1]) / 2
    pitch_ratio = (nose[1] - eye_mid) / (mouth_mid - nose[1] + 1e-6)

    return roll_degrees, yaw_ratio, pitch_ratio


def _check_pass(roll: float, yaw: float, pitch: float) -> Tuple[bool, str]:
    issues = []

    # Roll checks
    if roll > FACE_POSITION_CONSTANTS["MAX_ROLL_DEGREES"]:
        issues.append(
            f"roll > {FACE_POSITION_CONSTANTS['MAX_ROLL_DEGREES']:.2f}"
        )

    # Yaw checks
    if not (
        FACE_POSITION_CONSTANTS["YAW_RATIO_MIN"]
        <= yaw
        <= FACE_POSITION_CONSTANTS["YAW_RATIO_MAX"]
    ):
        issues.append(
            f"yaw not in [{FACE_POSITION_CONSTANTS['YAW_RATIO_MIN']:.2f}, "
            f"{FACE_POSITION_CONSTANTS['YAW_RATIO_MAX']:.2f}]"
        )

    # Pitch checks
    if not (
        FACE_POSITION_CONSTANTS["PITCH_RATIO_MIN"]
        <= pitch
        <= FACE_POSITION_CONSTANTS["PITCH_RATIO_MAX"]
    ):
        issues.append(
            f"pitch not in [{FACE_POSITION_CONSTANTS['PITCH_RATIO_MIN']:.2f}, "
            f"{FACE_POSITION_CONSTANTS['PITCH_RATIO_MAX']:.2f}]"
        )

    if issues:
        return False, "; ".join(issues)

    return True, "PASS"


def _draw_points(frame: np.ndarray, face_landmarks, w: int, h: int) -> None:
    points = {
        "left_eye":    33,
        "right_eye":   263,
        "nose":        1,
        "left_mouth":  61,
        "right_mouth": 291,
    }
    for name, idx in points.items():
        p = _pt(face_landmarks, idx, w, h)
        cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)


def _measure(text: str, font, scale: float, thick: int) -> int:
    """Return the pixel width of a rendered string."""
    return cv2.getTextSize(text, font, scale, thick)[0][0]


def _annotate(
    frame: np.ndarray,
    roll: float,
    yaw: float,
    pitch: float,
    passed: bool,
    issue: str,
) -> np.ndarray:
    """
    Appends a professional black info-panel below the image.
    The panel (and image canvas) are widened automatically if the image
    is too narrow to fit all text — no hardcoded widths anywhere.

    Layout:
      ┌─────────────────────────────────────┐
      │  ● STATUS : PASS / FAIL             │
      │  ─────────────────────────────────  │
      │    Roll        :  xx.xx deg         │
      │    Yaw ratio   :  x.xx              │
      │    Pitch ratio :  x.xx              │
      │  [FAIL only] ─────────────────────  │
      │    Issue  : <description, wrapped>  │
      └─────────────────────────────────────┘
    """
    img_h, img_w = frame.shape[:2]

    # ── constants ────────────────────────────────────────────────────────────
    FONT        = cv2.FONT_HERSHEY_DUPLEX
    FONT_TITLE  = 1.1
    FONT_BODY   = 0.72
    FONT_ISSUE  = 0.65
    THICK_TITLE = 2
    THICK_BODY  = 1

    PAD_X       = 28    # left / right inner margin
    PAD_TOP     = 22
    PAD_BOT     = 22
    LINE_H      = 36    # vertical advance per body line
    SEP_H       = 14    # space above/below separator line

    COL_BG    = (0,   0,   0  )
    COL_PASS  = (50,  220, 100)
    COL_FAIL  = (50,  80,  240)
    COL_LABEL = (160, 160, 160)
    COL_VALUE = (240, 240, 240)
    COL_SEP   = (60,  60,  60 )
    COL_DOT   = COL_PASS if passed else COL_FAIL
    status_col = COL_PASS if passed else COL_FAIL
    status_text = "PASS" if passed else "FAIL"

    # ── metric rows ───────────────────────────────────────────────────────────
    metrics = [
        ("Roll",        f"{roll:.2f} deg"),
        ("Yaw ratio",   f"{yaw:.2f}"),
        ("Pitch ratio", f"{pitch:.2f}"),
    ]

    # ── PASS 1: measure every piece of text to find required panel width ──────
    dot_r   = 9
    dot_diam = dot_r * 2
    gap_after_dot = 14

    # status row width
    status_label = "STATUS : "
    title_row_w = (
        PAD_X + dot_diam + gap_after_dot
        + _measure(status_label, FONT, FONT_TITLE, THICK_TITLE)
        + _measure(status_text,  FONT, FONT_TITLE, THICK_TITLE)
        + PAD_X
    )

    # metric rows: align colon at the same x for all rows
    max_label_w = max(
        _measure(k + "  ", FONT, FONT_BODY, THICK_BODY)
        for k, _ in metrics
    )
    colon_w = _measure(": ", FONT, FONT_BODY, THICK_BODY)
    max_metric_row_w = max(
        PAD_X + max_label_w + colon_w
        + _measure(v, FONT, FONT_BODY, THICK_BODY)
        + PAD_X
        for _, v in metrics
    )

    # issue prefix (used for indentation of wrapped lines)
    issue_prefix    = "Issue  : "
    issue_prefix_w  = _measure(issue_prefix, FONT, FONT_ISSUE, THICK_BODY)

    # minimum panel width = widest of all rows (before wrapping)
    if not passed:
        # measure full issue string unwrapped to know the minimum we need
        full_issue_w = PAD_X + issue_prefix_w + _measure(issue, FONT, FONT_ISSUE, THICK_BODY) + PAD_X
    else:
        full_issue_w = 0

    # required panel width (content drives it, not the image)
    required_w = max(title_row_w, max_metric_row_w, full_issue_w)
    # canvas width = at least the image width, at least required_w
    canvas_w   = max(img_w, required_w)

    # ── word-wrap issue text to fit canvas ────────────────────────────────────
    issue_lines: list[str] = []
    if not passed:
        avail_px = canvas_w - PAD_X - issue_prefix_w - PAD_X
        words, current_line = issue.split(), ""
        for word in words:
            trial = (current_line + " " + word).strip()
            if _measure(trial, FONT, FONT_ISSUE, THICK_BODY) <= avail_px:
                current_line = trial
            else:
                if current_line:
                    issue_lines.append(current_line)
                current_line = word
        if current_line:
            issue_lines.append(current_line)

    # ── compute panel height ──────────────────────────────────────────────────
    title_h  = LINE_H + 4
    sep1_h   = SEP_H * 2 + 1
    body_h   = len(metrics) * LINE_H
    sep2_h   = (SEP_H * 2 + 1) if issue_lines else 0
    issue_h  = len(issue_lines) * LINE_H if issue_lines else 0
    panel_h  = PAD_TOP + title_h + sep1_h + body_h + sep2_h + issue_h + PAD_BOT

    # ── widen image canvas if needed ──────────────────────────────────────────
    if canvas_w > img_w:
        # pad image on the right with black so widths match
        pad_right = canvas_w - img_w
        frame = cv2.copyMakeBorder(
            frame, 0, 0, 0, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    # ── build panel ───────────────────────────────────────────────────────────
    panel = np.zeros((panel_h, canvas_w, 3), dtype=np.uint8)

    # accent bar
    cv2.rectangle(panel, (0, 0), (canvas_w, 4), COL_DOT, -1)

    cursor_y = PAD_TOP

    # status row
    dot_cx = PAD_X + dot_r
    dot_cy = cursor_y + dot_r + 4
    cv2.circle(panel, (dot_cx, dot_cy), dot_r, COL_DOT, -1)

    text_x   = dot_cx + dot_r + gap_after_dot
    text_y   = cursor_y + title_h - 4
    label_w  = _measure(status_label, FONT, FONT_TITLE, THICK_TITLE)
    cv2.putText(panel, status_label, (text_x, text_y),
                FONT, FONT_TITLE, COL_LABEL, THICK_TITLE, cv2.LINE_AA)
    cv2.putText(panel, status_text, (text_x + label_w, text_y),
                FONT, FONT_TITLE, status_col, THICK_TITLE, cv2.LINE_AA)
    cursor_y += title_h

    # separator 1
    cursor_y += SEP_H
    cv2.line(panel, (PAD_X, cursor_y), (canvas_w - PAD_X, cursor_y), COL_SEP, 1)
    cursor_y += SEP_H

    # metric rows
    colon_x = PAD_X + max_label_w
    for key, val in metrics:
        cv2.putText(panel, key,    (PAD_X,              cursor_y),
                    FONT, FONT_BODY, COL_LABEL, THICK_BODY, cv2.LINE_AA)
        cv2.putText(panel, ": ",   (colon_x,            cursor_y),
                    FONT, FONT_BODY, COL_LABEL, THICK_BODY, cv2.LINE_AA)
        cv2.putText(panel, val,    (colon_x + colon_w,  cursor_y),
                    FONT, FONT_BODY, COL_VALUE, THICK_BODY, cv2.LINE_AA)
        cursor_y += LINE_H

    # issue block
    if issue_lines:
        cursor_y += SEP_H
        cv2.line(panel, (PAD_X, cursor_y), (canvas_w - PAD_X, cursor_y), COL_SEP, 1)
        cursor_y += SEP_H

        cv2.putText(panel, issue_prefix, (PAD_X, cursor_y),
                    FONT, FONT_ISSUE, COL_LABEL, THICK_BODY, cv2.LINE_AA)
        cv2.putText(panel, issue_lines[0], (PAD_X + issue_prefix_w, cursor_y),
                    FONT, FONT_ISSUE, COL_FAIL, THICK_BODY, cv2.LINE_AA)
        cursor_y += LINE_H

        for extra in issue_lines[1:]:
            cv2.putText(panel, extra, (PAD_X + issue_prefix_w, cursor_y),
                        FONT, FONT_ISSUE, COL_FAIL, THICK_BODY, cv2.LINE_AA)
            cursor_y += LINE_H

    # ── stack ────────────────────────────────────────────────────────────────
    return np.vstack([frame, panel])


def _load_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img


def _to_mp_image(frame: np.ndarray):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def process_folder(input_dir: Path, output_dir: Path, upscale: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.txt"

    landmarker = _load_landmarker()

    with open(report_path, "w", encoding="utf-8") as report:
        for path in sorted(input_dir.glob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            img = _load_image(path)
            if img is None:
                report.write(f"{path.name}: FAILED to load image\n")
                continue

            mp_img = _to_mp_image(img)
            result = landmarker.detect(mp_img)
            if not result.face_landmarks:
                report.write(f"{path.name}: NO FACE\n")
                continue

            h, w = img.shape[:2]
            face_landmarks = result.face_landmarks[0]
            roll, yaw, pitch = _compute_ratios(face_landmarks, w, h)
            passed, issue   = _check_pass(roll, yaw, pitch)

            annotated = img.copy()
            _draw_points(annotated, face_landmarks, w, h)
            annotated = _annotate(annotated, roll, yaw, pitch, passed, issue)

            if upscale > 1:
                new_h, new_w = annotated.shape[:2]
                annotated = cv2.resize(
                    annotated,
                    (new_w * upscale, new_h * upscale),
                    interpolation=cv2.INTER_CUBIC,
                )

            out_path = output_dir / path.name
            cv2.imwrite(str(out_path), annotated)

            report.write(
                f"{path.name}: {'PASS' if passed else 'FAIL'} | "
                f"roll={roll:.2f}, yaw={yaw:.2f}, pitch={pitch:.2f} | {issue}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Debug face position checks on images."
    )
    parser.add_argument("--input",   required=True, help="Folder with images")
    parser.add_argument("--output",  required=True, help="Output folder for annotated images and report.txt")
    parser.add_argument("--upscale", type=int, default=2, help="Upscale factor for annotated images")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Input folder not found: {input_dir}")

    process_folder(input_dir, output_dir, args.upscale)


if __name__ == "__main__":
    main()