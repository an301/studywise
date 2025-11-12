#!/usr/bin/env python3
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: Failed to import mediapipe. Install with: pip install mediapipe",
        file=sys.stderr,
    )
    sys.exit(1)


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


def get_landmark_xy(
    landmarks, idx: int, width: int, height: int
) -> Optional[Tuple[int, int, float]]:
    """Return (x_px, y_px, visibility) for a landmark if confidently visible, else None.

    px = int(lm.x * frame_width); py = int(lm.y * frame_height). y increases downward.
    Only accept landmarks with visibility >= 0.5.
    """
    lm = landmarks[idx]
    if lm.visibility is None or lm.visibility < VISIBILITY_THRESHOLD:
        return None
    x_px = int(lm.x * width)
    y_px = int(lm.y * height)
    # Basic bounds check
    if x_px < 0 or y_px < 0 or x_px > width or y_px > height:
        return None
    return (x_px, y_px, lm.visibility)


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

    # Try to set resolution to 1280x720; fallback if not supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened at {actual_width}x{actual_height}")
    print("Press 'q' to quit")

    # FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = time.time()

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

            # MediaPipe expects RGB input
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Prepare for drawing/overlay
            frame = frame_bgr
            h, w = frame.shape[:2]

            present = False
            neck_tilt_deg = None
            ear_shoulder_deg = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                # Landmark indices
                PL = mp_pose.PoseLandmark

                nose = get_landmark_xy(lms, PL.NOSE.value, w, h)
                # Shoulders (required for presence)
                ls = get_landmark_xy(lms, PL.LEFT_SHOULDER.value, w, h)
                rs = get_landmark_xy(lms, PL.RIGHT_SHOULDER.value, w, h)

                # Ears with optional eye-outer fallback if ear missing
                le = get_landmark_xy(lms, PL.LEFT_EAR.value, w, h)
                if le is None:
                    le = get_landmark_xy(lms, PL.LEFT_EYE_OUTER.value, w, h)
                re = get_landmark_xy(lms, PL.RIGHT_EAR.value, w, h)
                if re is None:
                    re = get_landmark_xy(lms, PL.RIGHT_EYE_OUTER.value, w, h)

                # present only if both shoulders are confidently detected
                if ls is not None and rs is not None:
                    present = True

                    ls_xy = (int(ls[0]), int(ls[1]))
                    rs_xy = (int(rs[0]), int(rs[1]))
                    mid_shoulder = (
                        int(0.5 * (ls_xy[0] + rs_xy[0])),
                        int(0.5 * (ls_xy[1] + rs_xy[1])),
                    )

                    # Neck tilt: angle between mid-shoulder→nose and vertical
                    if nose is not None:
                        nose_xy = (int(nose[0]), int(nose[1]))
                        neck_tilt_deg = angle_to_vertical(mid_shoulder, nose_xy)
                        # Draw mid-shoulder to nose vector
                        cv2.line(
                            frame,
                            (int(mid_shoulder[0]), int(mid_shoulder[1])),
                            (int(nose_xy[0]), int(nose_xy[1])),
                            (0, 255, 255),  # yellow
                            2,
                        )

                    # Ear–shoulder angles per side; average across available sides
                    side_angles = []
                    if le is not None:
                        le_xy = (int(le[0]), int(le[1]))
                        angle_left = angle_to_vertical(ls_xy, le_xy)
                        side_angles.append(abs(angle_left))
                        cv2.line(
                            frame,
                            (int(ls_xy[0]), int(ls_xy[1])),
                            (int(le_xy[0]), int(le_xy[1])),
                            (255, 255, 0),  # cyan
                            2,
                        )
                    if re is not None:
                        re_xy = (int(re[0]), int(re[1]))
                        angle_right = angle_to_vertical(rs_xy, re_xy)
                        side_angles.append(abs(angle_right))
                        cv2.line(
                            frame,
                            (int(rs_xy[0]), int(rs_xy[1])),
                            (int(re_xy[0]), int(re_xy[1])),
                            (255, 255, 0),  # cyan
                            2,
                        )
                    if side_angles:
                        ear_shoulder_deg = float(np.mean(side_angles))

                    # Draw keypoints
                    for pt in [nose, le, re, ls, rs]:
                        if pt is not None:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            # Update FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed

            # Compose overlay text
            y0 = 30
            dy = 28
            txt_present = f"present: {'yes' if present else 'no'}"
            txt_neck = (
                f"neck_tilt_deg: {neck_tilt_deg:.1f}" if neck_tilt_deg is not None else "neck_tilt_deg: --"
            )
            txt_ear = (
                f"ear_shoulder_deg: {ear_shoulder_deg:.1f}" if ear_shoulder_deg is not None else "ear_shoulder_deg: --"
            )
            txt_fps = f"FPS: {fps:.1f}"

            for i, text in enumerate([txt_present, txt_neck, txt_ear, txt_fps]):
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

            # Show frame
            cv2.imshow("Posture Angles Preview", frame)

            # Quit on 'q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("\nQuitting...")
                break

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


