#!/usr/bin/env python3
"""
Simple webcam preview script using OpenCV.
Press 'q' to quit.
"""

import sys
import time
import cv2


def main():
    """Open webcam and display live feed with FPS overlay."""
    

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam.", file=sys.stderr)
        print("Please check that:", file=sys.stderr)
        print("  - A webcam is connected", file=sys.stderr)
        print("  - No other application is using it", file=sys.stderr)
        print("  - Camera permissions are granted", file=sys.stderr)
        return 1
    

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam opened successfully at {actual_width}x{actual_height}")
    print("Press 'q' to quit")
    
    
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
       
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to read frame from webcam.", file=sys.stderr)
                break
            
         
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            
         
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            

            cv2.imshow('Webcam Preview', frame)
            

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
    
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

