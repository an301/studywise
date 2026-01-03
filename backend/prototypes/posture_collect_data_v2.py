"""
Improved Posture Data Collection for StudyWise v2.

Features:
- Calibration step to establish YOUR baseline neutral posture
- Normalized features (distance-invariant)
- Visual guides showing what each posture should look like
- Better feature set for distinguishing slouch vs lean

Usage:
    python backend/prototypes/posture_collect_data_v2.py

Keys:
    c -> calibrate (do this first!)
    1 -> label = neutral
    2 -> label = slouch  
    3 -> label = lean
    4 -> label = away
    0 -> label = none (pause)
    q -> quit
"""

import csv
import os
import sys
import time
from collections import deque
from typing import Deque, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    print("ERROR: Failed to import mediapipe", file=sys.stderr)
    sys.exit(1)

ROLLING_WINDOW_FRAMES = 10
VISIBILITY_THRESHOLD = 0.5

# Colors (BGR)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED = (0, 0, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)

LABEL_COLORS = {
    "neutral": COLOR_GREEN,
    "slouch": COLOR_YELLOW,
    "lean": COLOR_RED,
    "away": COLOR_GRAY,
    "none": COLOR_WHITE,
}

LABEL_DESCRIPTIONS = {
    "neutral": "Sit upright, head balanced over shoulders",
    "slouch": "Round your back, let shoulders droop forward",
    "lean": "Push your head/chin forward toward screen",
    "away": "Step away or look away from camera",
    "none": "Not recording - press 1/2/3/4 to start",
}


def angle_to_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Angle between vector p1->p2 and vertical (0-90 degrees)."""
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    ang = abs(np.degrees(np.arctan2(dx, -dy)))
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Euclidean distance between two points."""
    return float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))


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


def extract_features(
    landmarks, mp_pose, w: int, h: int
) -> Optional[dict]:
    """Extract comprehensive posture features from landmarks."""
    PL = mp_pose.PoseLandmark
    lms = landmarks
    
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
    
    if ls is None or rs is None or not has_head:
        return None
    
    ls_xy = (int(ls[0]), int(ls[1]))
    rs_xy = (int(rs[0]), int(rs[1]))
    mid_shoulder = (
        int(0.5 * (ls_xy[0] + rs_xy[0])),
        int(0.5 * (ls_xy[1] + rs_xy[1])),
    )
    
    # Shoulder width (used for normalization)
    shoulder_width = distance(ls_xy, rs_xy)
    if shoulder_width < 10:
        return None
    
    # Roll compensation
    roll_deg = float(np.degrees(np.arctan2(rs_xy[1] - ls_xy[1], rs_xy[0] - ls_xy[0])))
    
    nose_xy = (int(nose[0]), int(nose[1])) if nose is not None else None
    le_xy = (int(le[0]), int(le[1])) if le is not None else None
    re_xy = (int(re[0]), int(re[1])) if re is not None else None
    
    # Roll-normalized points
    ls_n = rotate_point(ls_xy, mid_shoulder, -roll_deg)
    rs_n = rotate_point(rs_xy, mid_shoulder, -roll_deg)
    nose_n = rotate_point(nose_xy, mid_shoulder, -roll_deg) if nose_xy else None
    le_n = rotate_point(le_xy, mid_shoulder, -roll_deg) if le_xy else None
    re_n = rotate_point(re_xy, mid_shoulder, -roll_deg) if re_xy else None
    
    features = {
        "present": True,
        "shoulder_width": shoulder_width,
        "roll_deg": roll_deg,
        "raw_points": {
            "ls": ls_xy, "rs": rs_xy, "mid_shoulder": mid_shoulder,
            "nose": nose_xy, "le": le_xy, "re": re_xy,
        }
    }
    
    # Feature 1: Neck angle (shoulder midpoint to nose)
    if nose_n is not None:
        features["neck_angle"] = angle_to_vertical(mid_shoulder, nose_n)
        # Normalized nose-to-shoulder distance (head forward/back)
        nose_dist = distance(mid_shoulder, nose_n)
        features["nose_dist_norm"] = nose_dist / shoulder_width
        # How far forward is nose relative to shoulder line (horizontal offset)
        features["nose_forward_norm"] = (nose_n[1] - mid_shoulder[1]) / shoulder_width
    
    # Feature 2: Ear-shoulder angles
    ear_angles = []
    ear_dists = []
    if le_n is not None:
        ear_angles.append(angle_to_vertical(ls_n, le_n))
        ear_dists.append(distance(ls_n, le_n) / shoulder_width)
    if re_n is not None:
        ear_angles.append(angle_to_vertical(rs_n, re_n))
        ear_dists.append(distance(rs_n, re_n) / shoulder_width)
    
    if ear_angles:
        features["ear_angle"] = float(np.mean(ear_angles))
        features["ear_dist_norm"] = float(np.mean(ear_dists))
    
    # Feature 3: Head tilt (difference between left and right ear heights)
    if le_n is not None and re_n is not None:
        features["head_tilt"] = (le_n[1] - re_n[1]) / shoulder_width
    
    # Feature 4: Shoulder Y position (normalized by frame height)
    features["shoulder_y_norm"] = mid_shoulder[1] / h
    
    return features


def draw_guide(frame: np.ndarray, label: str) -> None:
    """Draw visual guide for the current posture being collected."""
    h, w = frame.shape[:2]
    guide_w, guide_h = 200, 150
    x1, y1 = w - guide_w - 10, h - guide_h - 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + guide_w, y1 + guide_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw stick figure based on posture
    cx, cy = x1 + guide_w // 2, y1 + 40
    head_r = 20
    
    color = LABEL_COLORS.get(label, COLOR_WHITE)
    
    if label == "neutral":
        # Upright posture
        cv2.circle(frame, (cx, cy), head_r, color, 2)  # head
        cv2.line(frame, (cx, cy + head_r), (cx, cy + 60), color, 2)  # spine
        cv2.line(frame, (cx - 30, cy + 35), (cx + 30, cy + 35), color, 2)  # shoulders
    elif label == "slouch":
        # Rounded back, shoulders forward
        cv2.circle(frame, (cx, cy + 10), head_r, color, 2)  # head lower
        cv2.ellipse(frame, (cx, cy + 40), (8, 30), 0, -90, 90, color, 2)  # curved spine
        cv2.line(frame, (cx - 25, cy + 45), (cx + 25, cy + 45), color, 2)  # shoulders
    elif label == "lean":
        # Head pushed forward
        cv2.circle(frame, (cx + 25, cy), head_r, color, 2)  # head forward
        cv2.line(frame, (cx + 15, cy + head_r), (cx, cy + 60), color, 2)  # angled neck
        cv2.line(frame, (cx - 30, cy + 35), (cx + 30, cy + 35), color, 2)  # shoulders
    elif label == "away":
        cv2.putText(frame, "?", (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Label text
    cv2.putText(frame, label.upper(), (x1 + 10, y1 + guide_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    desc = LABEL_DESCRIPTIONS.get(label, "")[:25]
    cv2.putText(frame, desc, (x1 + 10, y1 + guide_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)


def main() -> int:
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "posture_data_v2.csv")
    new_file = not os.path.exists(csv_path)
    
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    
    # New feature columns
    feature_cols = [
        "timestamp", "neck_angle", "ear_angle", "nose_dist_norm", 
        "ear_dist_norm", "nose_forward_norm", "shoulder_y_norm", 
        "head_tilt", "roll_deg", "label"
    ]
    if new_file:
        writer.writerow(feature_cols)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.", file=sys.stderr)
        return 1
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened at {actual_width}x{actual_height}")
    print("\n=== INSTRUCTIONS ===")
    print("1. Press 'c' to CALIBRATE first (sit in neutral position)")
    print("2. Then press 1/2/3/4 to record each posture")
    print("Keys: 1=neutral, 2=slouch, 3=lean, 4=away, 0=pause, c=calibrate, q=quit")
    
    # Smoothing windows
    windows: dict = {k: deque(maxlen=ROLLING_WINDOW_FRAMES) for k in [
        "neck_angle", "ear_angle", "nose_dist_norm", "ear_dist_norm",
        "nose_forward_norm", "shoulder_y_norm", "head_tilt", "roll_deg"
    ]}
    
    current_label = "none"
    calibrated = False
    calibration_baseline = {}
    calib_samples: List[dict] = []
    calibrating = False
    calib_start = 0.0
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    sample_count = {k: 0 for k in ["neutral", "slouch", "lean", "away"]}
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.", file=sys.stderr)
                break
            
            now = time.time()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            
            frame = frame_bgr
            h, w = frame.shape[:2]
            
            features = None
            if results.pose_landmarks:
                features = extract_features(results.pose_landmarks.landmark, mp_pose, w, h)
            
            # Update smoothing windows
            for key in windows:
                val = features.get(key) if features else None
                windows[key].append(val)
            
            # Get smoothed values
            smoothed = {k: median_ignore_none(windows[k]) for k in windows}
            present = features is not None
            
            # Calibration handling
            if calibrating:
                elapsed = now - calib_start
                remaining = 3.0 - elapsed
                
                if features:
                    calib_samples.append(features)
                
                # Draw calibration progress
                cv2.putText(frame, f"CALIBRATING... {remaining:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_CYAN, 3)
                cv2.putText(frame, "Sit upright in neutral position!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
                
                if elapsed >= 3.0:
                    if len(calib_samples) > 10:
                        # Compute baseline from calibration samples
                        for key in ["neck_angle", "ear_angle", "nose_dist_norm", 
                                    "ear_dist_norm", "nose_forward_norm"]:
                            vals = [s.get(key) for s in calib_samples if s.get(key) is not None]
                            if vals:
                                calibration_baseline[key] = float(np.median(vals))
                        calibrated = True
                        print(f"\n=== CALIBRATION COMPLETE ===")
                        print(f"Baseline: {calibration_baseline}")
                    else:
                        print("Calibration failed - not enough samples. Try again.")
                    calibrating = False
                    calib_samples = []
            
            # Draw UI
            y0, dy = 30, 28
            label_color = LABEL_COLORS.get(current_label, COLOR_WHITE)
            
            # Status bar
            status = "CALIBRATED" if calibrated else "NOT CALIBRATED (press 'c')"
            status_color = COLOR_GREEN if calibrated else COLOR_RED
            cv2.putText(frame, status, (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Current label (big)
            txt_label = f"[ {current_label.upper()} ]"
            cv2.putText(frame, txt_label, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
            
            # Recording indicator
            is_recording = current_label in ("neutral", "slouch", "lean", "away")
            rec_text = "RECORDING" if is_recording else "PAUSED"
            rec_color = (0, 0, 255) if is_recording else (128, 128, 128)
            cv2.putText(frame, rec_text, (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
            
            # Sample counts
            counts_text = f"n: {sample_count['neutral']} | s: {sample_count['slouch']} | l: {sample_count['lean']} | a: {sample_count['away']}"
            cv2.putText(frame, counts_text, (10, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
            
            # Feature values
            txt_present = f"present: {'yes' if present else 'no'}"
            txt_neck = f"neck_angle: {smoothed['neck_angle']:.1f}" if smoothed['neck_angle'] else "neck_angle: --"
            txt_ear = f"ear_angle: {smoothed['ear_angle']:.1f}" if smoothed['ear_angle'] else "ear_angle: --"
            txt_nose = f"nose_forward: {smoothed['nose_forward_norm']:.2f}" if smoothed['nose_forward_norm'] else "nose_forward: --"
            
            for i, text in enumerate([txt_present, txt_neck, txt_ear, txt_nose]):
                cv2.putText(frame, text, (10, y0 + (i + 3) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 1)
            
            # Draw visual guide
            if current_label != "none":
                draw_guide(frame, current_label)
            
            # Draw pose keypoints
            if features and "raw_points" in features:
                pts = features["raw_points"]
                for key in ["ls", "rs", "nose", "le", "re"]:
                    if pts.get(key):
                        cv2.circle(frame, pts[key], 5, COLOR_GREEN, -1)
                if pts.get("mid_shoulder") and pts.get("nose"):
                    cv2.line(frame, pts["mid_shoulder"], pts["nose"], COLOR_CYAN, 2)
            
            cv2.imshow("Posture Data Collection v2", frame)
            
            # Write data
            if current_label == "away":
                # Away: write sentinel values
                writer.writerow([now, -1, -1, -1, -1, -1, -1, -1, -1, "away"])
                f.flush()
                sample_count["away"] += 1
            elif current_label in ("neutral", "slouch", "lean") and present:
                # Only write if we have all features
                if all(smoothed.get(k) is not None for k in ["neck_angle", "ear_angle"]):
                    row = [
                        now,
                        smoothed.get("neck_angle", -1),
                        smoothed.get("ear_angle", -1),
                        smoothed.get("nose_dist_norm", -1),
                        smoothed.get("ear_dist_norm", -1),
                        smoothed.get("nose_forward_norm", -1),
                        smoothed.get("shoulder_y_norm", -1),
                        smoothed.get("head_tilt", -1),
                        smoothed.get("roll_deg", -1),
                        current_label,
                    ]
                    writer.writerow(row)
                    f.flush()
                    sample_count[current_label] += 1
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("c") and not calibrating:
                calibrating = True
                calib_start = time.time()
                calib_samples = []
                print("\nStarting calibration - sit upright for 3 seconds...")
            elif key == ord("1"):
                current_label = "neutral"
                print(">>> Label: NEUTRAL - sit upright, head over shoulders")
            elif key == ord("2"):
                current_label = "slouch"
                print(">>> Label: SLOUCH - round your back, let shoulders droop")
            elif key == ord("3"):
                current_label = "lean"
                print(">>> Label: LEAN - push your head/chin forward")
            elif key == ord("4"):
                current_label = "away"
                print(">>> Label: AWAY - step away or look away")
            elif key == ord("0"):
                current_label = "none"
                print(">>> PAUSED - not recording")
    
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        f.close()
        print(f"\nSaved data to {csv_path}")
        print(f"Total samples: {sum(sample_count.values())}")
        print(f"Per class: {sample_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

