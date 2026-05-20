import cv2

def find_camera():
    """Find and return the first available camera"""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[CAM] Found camera {i}")
                return cap
            cap.release()
    return None

def main():
    # Find camera
    cap = find_camera()
    if cap is None:
        print("[FATAL] No camera found")
        return
    
    print("[SYSTEM] Camera found - Press 'q' to quit")
    
    # Display camera feed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Rotate frame 90 degrees counter-clockwise (like original code)
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Show the frame
        cv2.imshow('Camera Feed', rotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Shutdown")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user")
