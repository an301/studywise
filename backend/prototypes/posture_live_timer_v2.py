"""
StudyWise Live Posture + Focus Monitor v2

Improved version with:
- Better feature extraction
- Clearer visual feedback
- State stability (no flickering)
- Posture alerts after sustained bad posture

Usage:
    python backend/prototypes/posture_live_timer_v2.py

Controls:
    q -> quit
    r -> reset focus timer
    c -> calibrate
"""

import os
import sys
import time
from collections import deque
from typing import Deque, Iterable, Optional, Tuple

import cv2
import joblib
import numpy as np

try:
    import mediapipe as mp
except Exception:
    print("ERROR: Failed to import mediapipe", file=sys.stderr)
    sys.exit(1)

ROLLING_WINDOW_FRAMES = 10
VISIBILITY_THRESHOLD = 0.5
STATE_STABLE_SECONDS = 1.5  # Must stay in state for 1.5s before switching

# Colors (BGR)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED = (0, 0, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)

STATE_COLORS = {
    "neutral": COLOR_GREEN,
    "slouch": COLOR_YELLOW,
    "lean": COLOR_RED,
    "away": COLOR_GRAY,
}

STATE_MESSAGES = {
    "neutral": "Great posture!",
    "slouch": "Sit up straight!",
    "lean": "Move back from screen!",
    "away": "Away from desk",
}


def angle_to_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    ang = abs(np.degrees(np.arctan2(dx, -dy)))
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
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


def extract_features(landmarks, mp_pose, w: int, h: int) -> Optional[dict]:
    """Extract posture features from landmarks."""
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
    
    shoulder_width = distance(ls_xy, rs_xy)
    if shoulder_width < 10:
        return None
    
    roll_deg = float(np.degrees(np.arctan2(rs_xy[1] - ls_xy[1], rs_xy[0] - ls_xy[0])))
    
    nose_xy = (int(nose[0]), int(nose[1])) if nose is not None else None
    le_xy = (int(le[0]), int(le[1])) if le is not None else None
    re_xy = (int(re[0]), int(re[1])) if re is not None else None
    
    ls_n = rotate_point(ls_xy, mid_shoulder, -roll_deg)
    rs_n = rotate_point(rs_xy, mid_shoulder, -roll_deg)
    nose_n = rotate_point(nose_xy, mid_shoulder, -roll_deg) if nose_xy else None
    le_n = rotate_point(le_xy, mid_shoulder, -roll_deg) if le_xy else None
    re_n = rotate_point(re_xy, mid_shoulder, -roll_deg) if re_xy else None
    
    features = {
        "present": True,
        "raw_points": {
            "ls": ls_xy, "rs": rs_xy, "mid_shoulder": mid_shoulder,
            "nose": nose_xy, "le": le_xy, "re": re_xy,
        }
    }
    
    if nose_n is not None:
        features["neck_angle"] = angle_to_vertical(mid_shoulder, nose_n)
        nose_dist = distance(mid_shoulder, nose_n)
        features["nose_dist_norm"] = nose_dist / shoulder_width
        features["nose_forward_norm"] = (nose_n[1] - mid_shoulder[1]) / shoulder_width
    
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
    
    if le_n is not None and re_n is not None:
        features["head_tilt"] = (le_n[1] - re_n[1]) / shoulder_width
    
    features["shoulder_y_norm"] = mid_shoulder[1] / h
    features["roll_deg"] = roll_deg
    
    return features


def draw_status_pill(img: np.ndarray, state: str, confidence: float) -> None:
    """Draw posture status pill in top-right corner."""
    h, w = img.shape[:2]
    color = STATE_COLORS.get(state, COLOR_GRAY)
    
    label = f"{state.upper()}"
    if confidence > 0:
        label += f" {confidence:.0%}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    
    pad_x, pad_y = 20, 12
    pill_w = text_w + 2 * pad_x
    pill_h = text_h + 2 * pad_y
    
    x2 = w - 10
    y1 = 10
    x1 = x2 - pill_w
    y2 = y1 + pill_h
    
    # Draw pill background
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # Draw text
    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_h - 4
    cv2.putText(img, label, (text_x, text_y), font, font_scale, COLOR_WHITE, thickness)


def draw_posture_alert(img: np.ndarray, message: str, duration: float) -> None:
    """Draw alert banner when posture is bad for too long."""
    h, w = img.shape[:2]
    
    # Pulsing effect
    alpha = 0.5 + 0.3 * np.sin(duration * 3)
    
    # Red banner at bottom
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), COLOR_RED, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Alert text
    cv2.putText(img, f"⚠ {message}", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
    cv2.putText(img, f"Bad posture for {duration:.0f}s - please adjust!", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)


def main() -> int:
    # Load model
    model_path = os.path.join("backend", "models", "posture_model_v2.joblib")
    if not os.path.exists(model_path):
        # Try old model as fallback
        model_path_old = os.path.join("backend", "models", "posture_model.joblib")
        if os.path.exists(model_path_old):
            print("WARNING: Using old model. Run train_posture_model_v2.py for better results.")
            model_data = joblib.load(model_path_old)
            # Old model format
            POSTURE_MODEL = model_data
            FEATURE_COLS = ["neck_deg", "ear_deg", "shoulder_y"]
            use_old_model = True
        else:
            print(f"ERROR: No model found at {model_path}", file=sys.stderr)
            print("Run train_posture_model_v2.py first.", file=sys.stderr)
            return 1
    else:
        model_data = joblib.load(model_path)
        POSTURE_MODEL = model_data["model"]
        FEATURE_COLS = model_data["feature_cols"]
        use_old_model = False
    
    print(f"Loaded model with features: {FEATURE_COLS}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.", file=sys.stderr)
        return 1
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Webcam opened")
    print("Controls: q=quit, r=reset timer")
    
    # Smoothing windows
    windows = {k: deque(maxlen=ROLLING_WINDOW_FRAMES) for k in [
        "neck_angle", "ear_angle", "nose_dist_norm", "ear_dist_norm",
        "nose_forward_norm", "shoulder_y_norm", "head_tilt", "roll_deg"
    ]}
    
    # Timers
    session_start = time.time()
    focused_seconds = 0.0
    last_time = time.time()
    
    # State smoothing
    current_state = "away"
    current_confidence = 0.0
    candidate_state = None
    candidate_duration = 0.0
    
    # Bad posture tracking (for alerts)
    bad_posture_start = None
    BAD_POSTURE_ALERT_THRESHOLD = 30.0  # Alert after 30s of bad posture
    
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
                break
            
            now = time.time()
            dt = now - last_time
            last_time = now
            
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
            
            smoothed = {k: median_ignore_none(windows[k]) for k in windows}
            present = features is not None
            
            # Model prediction
            raw_state = "away"
            raw_confidence = 0.0
            
            if present:
                if use_old_model:
                    # Old model uses different features
                    neck = smoothed.get("neck_angle")
                    ear = smoothed.get("ear_angle")
                    sh_y = smoothed.get("shoulder_y_norm")
                    if neck is not None and ear is not None and sh_y is not None:
                        feat = [[neck, ear, sh_y * 720]]  # Denormalize for old model
                        probs = POSTURE_MODEL.predict_proba(feat)[0]
                        idx = int(np.argmax(probs))
                        raw_state = str(POSTURE_MODEL.classes_[idx])
                        raw_confidence = float(probs[idx])
                else:
                    # New model
                    feat_values = []
                    valid = True
                    for col in FEATURE_COLS:
                        # Map feature names
                        key = col
                        val = smoothed.get(key)
                        if val is None:
                            valid = False
                            break
                        feat_values.append(val)
                    
                    if valid:
                        feat = [feat_values]
                        probs = POSTURE_MODEL.predict_proba(feat)[0]
                        idx = int(np.argmax(probs))
                        raw_state = str(POSTURE_MODEL.classes_[idx])
                        raw_confidence = float(probs[idx])
            
            # State smoothing (debounce)
            if raw_state == current_state:
                candidate_state = None
                candidate_duration = 0.0
            else:
                if candidate_state == raw_state:
                    candidate_duration += dt
                    if candidate_duration >= STATE_STABLE_SECONDS:
                        current_state = raw_state
                        current_confidence = raw_confidence
                        candidate_state = None
                        candidate_duration = 0.0
                else:
                    candidate_state = raw_state
                    candidate_duration = dt
            
            # Update confidence even if state unchanged
            if raw_state == current_state:
                current_confidence = 0.8 * current_confidence + 0.2 * raw_confidence
            
            # Focus time tracking
            if current_state in ("neutral", "slouch"):
                focused_seconds += dt
            
            # Bad posture tracking
            if current_state in ("slouch", "lean"):
                if bad_posture_start is None:
                    bad_posture_start = now
            else:
                bad_posture_start = None
            
            bad_posture_duration = 0.0
            if bad_posture_start is not None:
                bad_posture_duration = now - bad_posture_start
            
            # Draw UI
            session_secs = int(now - session_start)
            focus_secs = int(focused_seconds)
            
            y0, dy = 30, 28
            
            # Session time
            cv2.putText(frame, f"Session: {session_secs // 60}:{session_secs % 60:02d}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
            
            # Focus time
            cv2.putText(frame, f"Focused: {focus_secs // 60}:{focus_secs % 60:02d}",
                        (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
            
            # State message
            msg = STATE_MESSAGES.get(current_state, "")
            msg_color = STATE_COLORS.get(current_state, COLOR_WHITE)
            cv2.putText(frame, msg, (10, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)
            
            # Debug info (smaller)
            if features:
                neck = smoothed.get("neck_angle")
                ear = smoothed.get("ear_angle")
                debug = f"neck: {neck:.1f}  ear: {ear:.1f}" if neck and ear else ""
                cv2.putText(frame, debug, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1)
            
            # Draw keypoints
            if features and "raw_points" in features:
                pts = features["raw_points"]
                for key in ["ls", "rs", "nose", "le", "re"]:
                    if pts.get(key):
                        cv2.circle(frame, pts[key], 5, STATE_COLORS.get(current_state, COLOR_GREEN), -1)
                if pts.get("mid_shoulder") and pts.get("nose"):
                    cv2.line(frame, pts["mid_shoulder"], pts["nose"], COLOR_CYAN, 2)
            
            # Status pill
            draw_status_pill(frame, current_state, current_confidence)
            
            # Bad posture alert
            if bad_posture_duration >= BAD_POSTURE_ALERT_THRESHOLD:
                draw_posture_alert(frame, STATE_MESSAGES.get(current_state, "Fix your posture!"), bad_posture_duration)
            
            cv2.imshow("StudyWise Posture Monitor v2", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                focused_seconds = 0.0
                print("Focus timer reset")
    
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        
        # Print session summary
        total = int(time.time() - session_start)
        focus = int(focused_seconds)
        print(f"\n=== SESSION SUMMARY ===")
        print(f"Total time: {total // 60}:{total % 60:02d}")
        print(f"Focused time: {focus // 60}:{focus % 60:02d}")
        if total > 0:
            print(f"Focus rate: {100 * focus / total:.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

