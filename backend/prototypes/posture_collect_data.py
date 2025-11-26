"""
Posture data collection script for StudyWise.

Usage:
    python backend/prototypes/posture_collect_data.py

Keys:
    1 -> label = neutral
    2 -> label = slouch
    3 -> label = lean
    0 -> label = none (pause labeling)
    q -> quit

Output:
    data/posture_data.csv with columns:
        timestamp, neck_deg, ear_deg, shoulder_y, label
"""

import csv
import os
import sys
import time
from collections import deque
from typing import Deque, Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    print("ERROR: Failed to import mediapipe. Install with: pip install mediapipe", file=sys.stderr)
    sys.exit(1)

ROLLING_WINDOW_FRAMES = 15
VISIBILITY_THRESHOLD = 0.5


def angle_to_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    ang = abs(np.degrees(np.arctan2(dx, -dy)))
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)


def rotate_point(p: Tuple[int, int], center: Tuple[int, int], deg: float) -> Tuple[int, int]:
    th = np.radians(deg)
    cos, sin = float(np.cos(th)), float(np.sin(th))
    x, y = float(p[0] - center[0]), float(p[1] - center[1])
    xr = x * cos - y * sin
    yr = x * sin + y * cos
    return (int(xr + center[0]), int(yr + center[1]))


def get_landmark_xy(landmarks, idx: int, width: int, height: int) -> Optional[Tuple[int, int, float]]:
    lm = landmarks[idx]
    if lm.visibility is None or lm.visibility < VISIBILITY_THRESHOLD:
        return None
    x_px = int(lm.x * width)
    y_px = int(lm.y * height)
    if x_px < 0 or y_px < 0 or x_px > width or y_px > height:
        return None
    return (x_px, y_px, lm.visibility)


def median_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [v for v in values if v is not None]
    if not arr:
        return None
    return float(np.median(np.array(arr, dtype=np.float32)))


def main() -> int:
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "posture_data.csv")
    new_file = not os.path.exists(csv_path)

    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["timestamp", "neck_deg", "ear_deg", "shoulder_y", "label"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened at {actual_width}x{actual_height}")
    print("Keys: 1=neutral, 2=slouch, 3=lean, 0=none, q=quit")

    neck_window: Deque[Optional[float]] = deque(maxlen=ROLLING_WINDOW_FRAMES)
    ear_window: Deque[Optional[float]] = deque(maxlen=ROLLING_WINDOW_FRAMES)

    current_label = "none"
    last_time = time.time()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.", file=sys.stderr)
                break

            now = time.time()
            dt = now - last_time
            last_time = now

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            frame = frame_bgr
            h, w = frame.shape[:2]

            present = False
            neck_deg_frame: Optional[float] = None
            ear_deg_frame: Optional[float] = None
            shoulder_y: Optional[float] = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                PL = mp_pose.PoseLandmark

                ls = get_landmark_xy(lms, PL.LEFT_SHOULDER.value, w, h)
                rs = get_landmark_xy(lms, PL.RIGHT_SHOULDER.value, w, h)
                nose = get_landmark_xy(lms, PL.NOSE.value, w, h)

                le = get_landmark_xy(lms, PL.LEFT_EAR.value, w, h)
                if le is None:
                    le = get_landmark_xy(lms, PL.LEFT_EYE_OUTER.value, w, h)
                re = get_landmark_xy(lms, PL.RIGHT_EAR.value, w, h)
                if re is None:
                    re = get_landmark_xy(lms, PL.RIGHT_EYE_OUTER.value, w, h)

                has_head = (nose is not None) or (le is not None) or (re is not None)

                if ls is not None and rs is not None and has_head:
                    present = True

                    ls_xy = (int(ls[0]), int(ls[1]))
                    rs_xy = (int(rs[0]), int(rs[1]))
                    mid_shoulder = (
                        int(0.5 * (ls_xy[0] + rs_xy[0])),
                        int(0.5 * (ls_xy[1] + rs_xy[1])),
                    )

                    roll_deg = float(
                        np.degrees(
                            np.arctan2(rs_xy[1] - ls_xy[1], rs_xy[0] - ls_xy[0])
                        )
                    )

                    nose_xy = (int(nose[0]), int(nose[1])) if nose is not None else None
                    le_xy = (int(le[0]), int(le[1])) if le is not None else None
                    re_xy = (int(re[0]), int(re[1])) if re is not None else None

                    ls_n = rotate_point(ls_xy, mid_shoulder, -roll_deg)
                    rs_n = rotate_point(rs_xy, mid_shoulder, -roll_deg)
                    nose_n = rotate_point(nose_xy, mid_shoulder, -roll_deg) if nose_xy else None
                    le_n = rotate_point(le_xy, mid_shoulder, -roll_deg) if le_xy else None
                    re_n = rotate_point(re_xy, mid_shoulder, -roll_deg) if re_xy else None

                    if nose_n is not None:
                        neck_deg_frame = angle_to_vertical(mid_shoulder, nose_n)

                    side_vals = []
                    if le_n is not None:
                        side_vals.append(angle_to_vertical(ls_n, le_n))
                    if re_n is not None:
                        side_vals.append(angle_to_vertical(rs_n, re_n))
                    if side_vals:
                        ear_deg_frame = float(np.mean(side_vals))

                    # average shoulder y in original coords
                    shoulder_y = 0.5 * (ls_xy[1] + rs_xy[1])

            neck_window.append(neck_deg_frame)
            ear_window.append(ear_deg_frame)
            neck_deg_med = median_ignore_none(neck_window)
            ear_deg_med = median_ignore_none(ear_window)

            # UI overlays
            y0, dy = 30, 28
            txt_label = f"label: {current_label}"
            txt_present = f"present: {'yes' if present else 'no'}"
            txt_neck = f"neck_deg: {neck_deg_med:.1f}" if neck_deg_med is not None else "neck_deg: --"
            txt_ear = f"ear_deg: {ear_deg_med:.1f}" if ear_deg_med is not None else "ear_deg: --"
            txt_sh = f"shoulder_y: {shoulder_y:.1f}" if shoulder_y is not None else "shoulder_y: --"

            for i, text in enumerate([txt_label, txt_present, txt_neck, txt_ear, txt_sh]):
                cv2.putText(
                    frame,
                    text,
                    (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Posture Data Collection", frame)

            # --- write row if we have a valid label and features ---
            if (
                current_label in ("neutral", "slouch", "lean")
                and present
                and neck_deg_med is not None
                and ear_deg_med is not None
                and shoulder_y is not None
            ):
                writer.writerow(
                    [now, float(neck_deg_med), float(ear_deg_med), float(shoulder_y), current_label]
                )
                # flush occasionally
                f.flush()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("1"):
                current_label = "neutral"
                print("Label set to neutral")
            elif key == ord("2"):
                current_label = "slouch"
                print("Label set to slouch")
            elif key == ord("3"):
                current_label = "lean"
                print("Label set to lean")
            elif key == ord("0"):
                current_label = "none"
                print("Label set to none (not recording)")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            pose.close()
        except Exception:
            pass
        f.close()
        print(f"Saved data to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
