import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import sys
from typing import List, Tuple

# ================= CONFIG =================
SCALE_FACTOR = 10
CLIP_VALUE = 225
MOVEMENT_THRESHOLD = 30
MIN_REGION_CHANGE = 0.12
GRID_SIZE = (4, 4)
GLOBAL_CHANGE_THRESHOLD = 0.85
BLUR_KERNEL = (21, 21)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v')


def _safe_resize(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))


def _draw_label(frame: np.ndarray, text: str, y: int = 30, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def _collect_active_regions(
    thresh: np.ndarray,
    grid_size: Tuple[int, int],
    min_region_change: float,
) -> List[Tuple[int, int, float, int, int, int, int]]:
    """
    Return list of active regions as:
    (row, col, ratio, y1, y2, x1, x2)
    """
    h, w = thresh.shape
    grid_rows, grid_cols = grid_size

    if grid_rows <= 0 or grid_cols <= 0:
        return []

    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)

    active_regions: List[Tuple[int, int, float, int, int, int, int]] = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            y1, y2 = y_edges[i], y_edges[i + 1]
            x1, x2 = x_edges[j], x_edges[j + 1]
            region = thresh[y1:y2, x1:x2]

            region_total = region.size
            if region_total == 0:
                continue

            region_changed = int(np.sum(region > 0))
            region_ratio = region_changed / region_total

            if region_ratio > min_region_change:
                active_regions.append((i, j, region_ratio, y1, y2, x1, x2))

    return active_regions

# ================= DEBUG ANALYZER =================
def analyze_video(video_path: str) -> None:
    """
    Analyze video and show all detection mechanisms:
    - Difference
    - Centralization
    - Hard Scale 10x
    - Clip 225
    - Grayscale display
    - Global changes detection
    - Partial changes (region grid)
    """
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    delay = max(1, int(1000 / fps))
    
    print(f"[VIDEO] {video_path}")
    print(f"[INFO] FPS: {fps:.2f}, Delay: {delay}ms")
    print("[CONTROLS] SPACE=pause, Q=quit, R=restart, F=faster, S=slower")
    print("[ANALYSIS] Showing 7 windows with detection pipeline...")
    
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        cap.release()
        return
    
    paused = False
    current_delay = delay
    
    while True:
        if not paused:
            ret, curr_frame = cap.read()
            
            if not ret:
                print("[END] Video finished")
                break
            
            # ===== STEP 1: GRAYSCALE =====
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # ===== STEP 2: BLUR =====
            prev_blur = cv2.GaussianBlur(prev_gray, BLUR_KERNEL, 0)
            curr_blur = cv2.GaussianBlur(curr_gray, BLUR_KERNEL, 0)
            
            # ===== STEP 3: DIFFERENCE =====
            frame_diff = cv2.absdiff(prev_blur, curr_blur)
            
            # ===== STEP 4: CENTRALIZATION =====
            diff_mean = np.mean(frame_diff)
            frame_diff_centered = np.clip(frame_diff.astype(np.float32) - diff_mean + 128, 0, 255).astype(np.uint8)
            
            # ===== STEP 5: HARD SCALE 10x =====
            frame_diff_scaled = frame_diff_centered.astype(np.float32) * SCALE_FACTOR
            
            # ===== STEP 6: CLIP AT 225 =====
            frame_diff_clipped = np.clip(frame_diff_scaled, 0, CLIP_VALUE).astype(np.uint8)
            
            # ===== STEP 7: THRESHOLD =====
            _, thresh = cv2.threshold(frame_diff_clipped, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # ===== STEP 8: GLOBAL CHANGES =====
            total_pixels = thresh.shape[0] * thresh.shape[1]
            changed_pixels = np.sum(thresh > 0)
            global_ratio = changed_pixels / total_pixels
            
            # ===== STEP 9: PARTIAL CHANGES (REGIONS) =====
            h, w = thresh.shape
            grid_rows, grid_cols = GRID_SIZE
            active_regions = _collect_active_regions(thresh, GRID_SIZE, MIN_REGION_CHANGE)
            y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
            x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)
            
            # ===== VISUALIZATIONS =====
            
            # Window 1: Original current frame
            display_curr = _safe_resize(curr_frame)
            _draw_label(display_curr, "1. Current Frame (Original)")
            cv2.imshow("1. Current Frame", display_curr)
            
            # Window 2: Grayscale
            display_gray = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
            display_gray = _safe_resize(display_gray)
            _draw_label(display_gray, "2. Grayscale")
            cv2.imshow("2. Grayscale", display_gray)
            
            # Window 3: Raw difference
            display_diff = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)
            display_diff = _safe_resize(display_diff)
            _draw_label(display_diff, "3. Difference (Raw)")
            cv2.imshow("3. Difference (Raw)", display_diff)
            
            # Window 4: Centralized difference
            display_centered = cv2.cvtColor(frame_diff_centered, cv2.COLOR_GRAY2BGR)
            display_centered = _safe_resize(display_centered)
            _draw_label(display_centered, f"4. Centralized (mean={diff_mean:.1f})")
            cv2.imshow("4. Centralized", display_centered)
            
            # Window 5: Scaled 10x + Clipped 225
            display_scaled = cv2.cvtColor(frame_diff_clipped, cv2.COLOR_GRAY2BGR)
            display_scaled = _safe_resize(display_scaled)
            _draw_label(display_scaled, "5. 10x Scaled + Clip 225")
            cv2.imshow("5. Scaled & Clipped", display_scaled)
            
            # Window 6: Threshold mask
            display_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            display_thresh = _safe_resize(display_thresh)
            _draw_label(display_thresh, f"6. Threshold (changed={global_ratio*100:.1f}%)")
            
            # Global change indicator
            if global_ratio > GLOBAL_CHANGE_THRESHOLD:
                cv2.putText(display_thresh, "GLOBAL CHANGE DETECTED", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("6. Threshold Mask", display_thresh)
            
            # Window 7: Partial regions analysis
            display_regions = cv2.cvtColor(frame_diff_clipped, cv2.COLOR_GRAY2BGR)
            display_regions = _safe_resize(display_regions)
            
            # Draw grid
            scale_y = DISPLAY_HEIGHT / h
            scale_x = DISPLAY_WIDTH / w
            
            for i in range(grid_rows + 1):
                y = int(y_edges[i] * scale_y)
                cv2.line(display_regions, (0, y), (DISPLAY_WIDTH, y), (100, 100, 100), 1)
            
            for j in range(grid_cols + 1):
                x = int(x_edges[j] * scale_x)
                cv2.line(display_regions, (x, 0), (x, DISPLAY_HEIGHT), (100, 100, 100), 1)
            
            # Highlight active regions
            for i, j, ratio, y1, y2, x1, x2 in active_regions:
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                
                cv2.rectangle(display_regions, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_regions, f"{ratio*100:.0f}%", (x1+5, y1+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            _draw_label(display_regions, f"7. Partial Regions ({len(active_regions)} active)")
            
            # Movement decision
            if len(active_regions) >= 1 and global_ratio <= GLOBAL_CHANGE_THRESHOLD:
                cv2.putText(display_regions, "MOVEMENT DETECTED", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display_regions, "NO MOVEMENT", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("7. Partial Regions", display_regions)
            
            prev_frame = curr_frame.copy()
        
        # Handle keyboard
        key = cv2.waitKey(current_delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            print("[QUIT] User quit")
            break
        elif key == ord(' '):
            paused = not paused
            print(f"[PAUSE] {'Paused' if paused else 'Resumed'}")
        elif key == ord('r'):
            print("[RESTART] Restarting video...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, prev_frame = cap.read()
            if not ret:
                print("[ERROR] Could not restart video")
                break
            paused = False
        elif key == ord('f'):
            current_delay = max(1, current_delay // 2)
            print(f"[SPEED] Faster (delay={current_delay}ms)")
        elif key == ord('s'):
            current_delay = min(1000, current_delay * 2)
            print(f"[SPEED] Slower (delay={current_delay}ms)")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[COMPLETE] Analysis finished")

# ================= MAIN =================
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python debug_video.py <path_to_video_or_folder>")
        print("Example: python debug_video.py recorded_videos/")
        print("Example: python debug_video.py recorded_videos/fr_15_26_4_2026_14_30_45.mp4")
        return
    
    path = sys.argv[1]
    
    # Check if it's a folder
    if os.path.isdir(path):
        videos = sorted(
            f for f in os.listdir(path)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )
        
        if not videos:
            print(f"[ERROR] No videos found in {path}")
            return
        
        print(f"[FOUND] {len(videos)} videos in {path}")
        print("[SELECT] Choose video to analyze:")
        
        for idx, video in enumerate(videos, 1):
            print(f"  {idx}. {video}")
        
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(videos):
                video_path = os.path.join(path, videos[choice])
                analyze_video(video_path)
            else:
                print("[ERROR] Invalid choice")
        except ValueError:
            print("[ERROR] Invalid input")
    
    # Check if it's a file
    elif os.path.isfile(path):
        analyze_video(path)
    
    else:
        print(f"[ERROR] Path not found: {path}")

if __name__ == "__main__":
    main()
