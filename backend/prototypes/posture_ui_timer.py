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

NEUTRAL_NECK_MAX = 10.0
NEUTRAL_EARSHOULDER_MAX = 30.0
SEVERE_NECK_MIN = 20.0
SEVERE_EARSHOULDER_MIN = 25.0

COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED = (0, 0, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_WHITE = (255, 255, 255)

VISIBILITY_THRESHOLD = 0.5



def angle_to_vertical(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Return acute angle (0..90°) between vector p1->p2 and the vertical axis.

    Image coords: origin (0,0) top-left; x right is +; y down is +.
    Treat vertical 'up' as vector (0, -1). For vector v = (dx, dy), angle to vertical is
    atan2(dx, -dy). We then take absolute degrees and fold obtuse to acute.

    Expected upright: ≈ 0–8° neck tilt, ≈ 0–12° ear–shoulder.
    """
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])  # positive is down
    ang = abs(np.degrees(np.arctan2(dx, -dy)))  # 0 when perfectly vertical up
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)


def rotate_point(p: Tuple[int, int], center: Tuple[int, int], deg: float) -> Tuple[int, int]:
    """Rotate point p around center by deg degrees (image coords)."""
    th = np.radians(deg)
    cos, sin = float(np.cos(th)), float(np.sin(th))
    x, y = float(p[0] - center[0]), float(p[1] - center[1])
    xr = x * cos - y * sin
    yr = x * sin + y * cos
    return (int(xr + center[0]), int(yr + center[1]))


def get_landmark_xy(landmarks, idx: int, width: int, height: int) -> Optional[Tuple[int, int, float]]:
    """Return (x_px, y_px, visibility) for a landmark if confidently visible, else None.

    px = int(lm.x * frame_width); py = int(lm.y * frame_height). y increases downward.
    Only accept landmarks with visibility >= 0.5.
    """
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


def classify_posture(present: bool, neck_deg: Optional[float], ear_deg: Optional[float]) -> str:
    if not present:
        return "away"
    # missing values handling; neck primary, ear supporting
    n = neck_deg if neck_deg is not None else 999.0
    e = ear_deg if ear_deg is not None else 999.0
    if n <= NEUTRAL_NECK_MAX and e <= (NEUTRAL_EARSHOULDER_MAX + 5.0):
        return "neutral"
    if n >= SEVERE_NECK_MIN or e >= SEVERE_EARSHOULDER_MIN:
        return "lean"
    return "slouch"


def draw_status_pill(img: np.ndarray, label: str, color: Tuple[int, int, int]) -> None:
    """Draw a rounded status pill with text in the top-right corner."""
    h, w = img.shape[:2]
    margin_x, margin_y = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad_x, pad_y = 16, 10
    pill_w = text_w + 2 * pad_x
    pill_h = text_h + 2 * pad_y

    x2 = w - margin_x
    y1 = margin_y
    x1 = x2 - pill_w
    y2 = y1 + pill_h
    radius = pill_h // 2

    # Central rectangles
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    # Circles at corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

    # Text
    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_h
    cv2.putText(img, label, (text_x, text_y), font, font_scale, COLOR_WHITE, thickness, cv2.LINE_AA)


def main() -> int:
    # Open default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.", file=sys.stderr)
        print("Please check that:", file=sys.stderr)
        print("  - A webcam is connected", file=sys.stderr)
        print("  - No other application is using it", file=sys.stderr)
        print("  - Camera permissions are granted", file=sys.stderr)
        return 1

    # Target 1280x720 if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened at {actual_width}x{actual_height}")
    print("Press 'q' to quit, 'r' to reset focus timer")

    # Rolling windows for smoothing (store per-frame values; ignore None in median)
    neck_window: Deque[Optional[float]] = deque(maxlen=ROLLING_WINDOW_FRAMES)
    ear_window: Deque[Optional[float]] = deque(maxlen=ROLLING_WINDOW_FRAMES)

    # Focus timer state
    focus_seconds = 0.0
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
                print("ERROR: Failed to read frame from webcam.", file=sys.stderr)
                break

            # Time delta for focus timer
            now = time.time()
            dt = now - last_time
            last_time = now

            # Process pose
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            frame = frame_bgr
            h, w = frame.shape[:2]

            present = False
            neck_deg_frame: Optional[float] = None
            ear_deg_frame: Optional[float] = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                PL = mp_pose.PoseLandmark

                # Shoulders determine presence
                ls = get_landmark_xy(lms, PL.LEFT_SHOULDER.value, w, h)
                rs = get_landmark_xy(lms, PL.RIGHT_SHOULDER.value, w, h)
                nose = get_landmark_xy(lms, PL.NOSE.value, w, h)

                # Ears with fallback to eye-outer if ear occluded
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

                    # Roll compensation: rotate points around mid-shoulder so shoulder line becomes horizontal
                    # roll_deg: angle of L->R shoulder vs. horizontal
                    roll_deg = float(np.degrees(np.arctan2(rs_xy[1] - ls_xy[1], rs_xy[0] - ls_xy[0])))

                    nose_xy = (int(nose[0]), int(nose[1])) if nose is not None else None
                    le_xy = (int(le[0]), int(le[1])) if le is not None else None
                    re_xy = (int(re[0]), int(re[1])) if re is not None else None

                    ls_n = rotate_point(ls_xy, mid_shoulder, -roll_deg)
                    rs_n = rotate_point(rs_xy, mid_shoulder, -roll_deg)
                    nose_n = rotate_point(nose_xy, mid_shoulder, -roll_deg) if nose_xy else None
                    le_n = rotate_point(le_xy, mid_shoulder, -roll_deg) if le_xy else None
                    re_n = rotate_point(re_xy, mid_shoulder, -roll_deg) if re_xy else None

                    # Neck tilt (angles are roll-normalized)
                    if nose_n is not None:
                        neck_deg_frame = angle_to_vertical(mid_shoulder, nose_n)
                        # Draw vectors on original frame for visual sanity
                        cv2.line(frame, (mid_shoulder[0], mid_shoulder[1]), (nose_xy[0], nose_xy[1]), (0, 255, 255), 2)

                    # Ear–shoulder per side (roll-normalized angles)
                    side_vals = []
                    if le_n is not None:
                        side_vals.append(angle_to_vertical(ls_n, le_n))
                        cv2.line(frame, (ls_xy[0], ls_xy[1]), (le_xy[0], le_xy[1]), (255, 255, 0), 2)
                    if re_n is not None:
                        side_vals.append(angle_to_vertical(rs_n, re_n))
                        cv2.line(frame, (rs_xy[0], rs_xy[1]), (re_xy[0], re_xy[1]), (255, 255, 0), 2)
                    if side_vals:
                        ear_deg_frame = float(np.mean(side_vals))

                    # Draw keypoints (original coords)
                    for pt in [nose, le, re, ls, rs]:
                        if pt is not None:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            # Update smoothing windows
            neck_window.append(neck_deg_frame)
            ear_window.append(ear_deg_frame)
            neck_deg_med = median_ignore_none(neck_window)
            ear_deg_med = median_ignore_none(ear_window)

            # Update focus timer (counts only while present)
            if present:
                focus_seconds += dt

            # Classification
            state = classify_posture(present, neck_deg_med, ear_deg_med)
            color = {
                "neutral": COLOR_GREEN,
                "slouch": COLOR_YELLOW,
                "lean": COLOR_RED,
                "away": COLOR_GRAY,
            }[state]

            # Top-left overlay text
            y0 = 30
            dy = 28
            txt_present = f"present: {'yes' if present else 'no'}"
            txt_neck = f"neck_tilt_deg: {neck_deg_med:.1f}" if neck_deg_med is not None else "neck_tilt_deg: --"
            txt_ear = f"ear_shoulder_deg: {ear_deg_med:.1f}" if ear_deg_med is not None else "ear_shoulder_deg: --"
            total_sec = int(focus_seconds)
            mm = total_sec // 60
            ss = total_sec % 60
            txt_focus = f"Focus: {mm}:{ss:02d}"

            for i, text in enumerate([txt_present, txt_neck, txt_ear, txt_focus]):
                cv2.putText(frame, text, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Status pill (top-right)
            draw_status_pill(frame, state, color)

            # Show frame
            cv2.imshow("Posture UI + Focus Timer", frame)

            # Calibration (optional): press 'c' to capture 3s of medians while present and adjust neutral thresholds
            # We accumulate only while present; then set NEUTRAL_* based on captured medians and flash a message.

            # Show calibration status/message overlays if active
            # (draw after main overlays)
            # Setup mutable state containers on first loop iteration
            if 'calibrating' not in locals():
                calibrating = False
                calib_present_elapsed = 0.0
                calib_neck_vals: Deque[float] = deque()
                calib_ear_vals: Deque[float] = deque()
                calibrate_msg_until = 0.0
                calibrate_msg_text = ""

            # Update calibration capture
            if calibrating:
                if present and neck_deg_med is not None:
                    calib_neck_vals.append(float(neck_deg_med))
                if present and ear_deg_med is not None:
                    calib_ear_vals.append(float(ear_deg_med))
                if present:
                    calib_present_elapsed += dt
                cv2.putText(frame, "Calibrating...", (10, y0 + 4 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if calib_present_elapsed >= 3.0:
                    # Compute medians from captured values (if any)
                    neck_cal_med = float(np.median(list(calib_neck_vals))) if len(calib_neck_vals) > 0 else None
                    ear_cal_med = float(np.median(list(calib_ear_vals))) if len(calib_ear_vals) > 0 else None

                    # Update thresholds (use globals)
                    global NEUTRAL_NECK_MAX, NEUTRAL_EARSHOULDER_MAX
                    if neck_cal_med is not None:
                        NEUTRAL_NECK_MAX = max(8.0, neck_cal_med + 4.0)
                    if ear_cal_med is not None:
                        NEUTRAL_EARSHOULDER_MAX = max(12.0, ear_cal_med + 6.0)

                    calibrate_msg_text = f"Calibrated: neck<={NEUTRAL_NECK_MAX:.1f}°, ear<={NEUTRAL_EARSHOULDER_MAX:.1f}°"
                    calibrate_msg_until = time.time() + 2.0
                    calibrating = False
                    calib_present_elapsed = 0.0
                    calib_neck_vals.clear()
                    calib_ear_vals.clear()

            # Flash post-calibration message
            if time.time() < calibrate_msg_until and calibrate_msg_text:
                cv2.putText(frame, calibrate_msg_text, (10, y0 + 5 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2, cv2.LINE_AA)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                focus_seconds = 0.0
                print("Focus timer reset")
            elif key == ord('c') and not calibrating:
                calibrating = True
                calib_present_elapsed = 0.0
                calib_neck_vals.clear()
                calib_ear_vals.clear()
                print("Calibration started: need ~3s while present")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            pose.close()
        except Exception:
            pass
        print("Webcam released and windows closed")

    return 0


if __name__ == "__main__":
    sys.exit(main())


