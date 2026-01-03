"""
StudyWise API Server - Lightweight Version

Streams video + posture data via WebSocket.

Run:
    cd /Users/anam301/Documents/GitHub/studywise
    source .venv311/bin/activate
    uvicorn backend.api.server:app --reload --port 8000
"""

import asyncio
import base64
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed")
    sys.exit(1)

app = FastAPI(title="StudyWise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class PostureEngine:
    """Lightweight posture detection."""
    
    VISIBILITY_THRESHOLD = 0.5
    
    def __init__(self):
        self.model = None
        self.feature_cols = []
        self.cap = None
        self.pose = None
        self.mp_pose = None
        self.camera_ok = False
        self.model_loaded = False
        
        # Smoothing - smaller window for responsiveness
        self.state_history = deque(maxlen=5)
        
        self._load_model()
    
    def _load_model(self):
        model_path = DATA_DIR.parent / "backend" / "models" / "posture_model_v2.joblib"
        if model_path.exists():
            data = joblib.load(model_path)
            self.model = data["model"]
            self.feature_cols = data["feature_cols"]
            self.model_loaded = True
            print(f"Loaded model: {len(self.feature_cols)} features")
        else:
            print("No model found")
    
    def start_camera(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_ok = False
            return False
        
        # Lower resolution for speed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=0,  # Fastest
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.camera_ok = True
        return True
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.pose:
            self.pose.close()
            self.pose = None
        self.camera_ok = False
    
    def _get_landmark(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        if lm.visibility is None or lm.visibility < self.VISIBILITY_THRESHOLD:
            return None
        x, y = int(lm.x * w), int(lm.y * h)
        if x < 0 or y < 0 or x > w or y > h:
            return None
        return (x, y)
    
    def _distance(self, p1, p2):
        return float(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
    
    def _angle_to_vertical(self, p1, p2):
        dx, dy = float(p2[0]-p1[0]), float(p2[1]-p1[1])
        ang = abs(np.degrees(np.arctan2(dx, -dy)))
        return float(180.0 - ang if ang > 90 else ang)
    
    def _rotate_point(self, p, center, deg):
        th = np.radians(deg)
        cos, sin = float(np.cos(th)), float(np.sin(th))
        x, y = float(p[0]-center[0]), float(p[1]-center[1])
        return (int(x*cos - y*sin + center[0]), int(x*sin + y*cos + center[1]))
    
    def process_frame(self) -> tuple:
        """Process frame, return (jpeg_bytes, posture_data)."""
        if not self.cap or not self.camera_ok:
            return None, {"state": "away", "present": False, "camera_ok": False}
        
        ret, frame = self.cap.read()
        if not ret:
            self.camera_ok = False
            return None, {"state": "away", "present": False, "camera_ok": False}
        
        h, w = frame.shape[:2]
        
        # Process pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        state = "away"
        confidence = 0.0
        present = False
        features = {}
        
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            PL = self.mp_pose.PoseLandmark
            
            ls = self._get_landmark(lms, PL.LEFT_SHOULDER.value, w, h)
            rs = self._get_landmark(lms, PL.RIGHT_SHOULDER.value, w, h)
            nose = self._get_landmark(lms, PL.NOSE.value, w, h)
            
            le = self._get_landmark(lms, PL.LEFT_EAR.value, w, h)
            if not le:
                le = self._get_landmark(lms, PL.LEFT_EYE_OUTER.value, w, h)
            re = self._get_landmark(lms, PL.RIGHT_EAR.value, w, h)
            if not re:
                re = self._get_landmark(lms, PL.RIGHT_EYE_OUTER.value, w, h)
            
            if ls and rs and (nose or le or re):
                present = True
                mid = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
                shoulder_w = self._distance(ls, rs)
                
                if shoulder_w > 10:
                    roll = float(np.degrees(np.arctan2(rs[1]-ls[1], rs[0]-ls[0])))
                    
                    ls_n = self._rotate_point(ls, mid, -roll)
                    rs_n = self._rotate_point(rs, mid, -roll)
                    
                    if nose:
                        nose_n = self._rotate_point(nose, mid, -roll)
                        features["neck_angle"] = self._angle_to_vertical(mid, nose_n)
                        features["nose_dist_norm"] = self._distance(mid, nose_n) / shoulder_w
                        features["nose_forward_norm"] = (nose_n[1] - mid[1]) / shoulder_w
                    
                    ear_angles, ear_dists = [], []
                    if le:
                        le_n = self._rotate_point(le, mid, -roll)
                        ear_angles.append(self._angle_to_vertical(ls_n, le_n))
                        ear_dists.append(self._distance(ls_n, le_n) / shoulder_w)
                    if re:
                        re_n = self._rotate_point(re, mid, -roll)
                        ear_angles.append(self._angle_to_vertical(rs_n, re_n))
                        ear_dists.append(self._distance(rs_n, re_n) / shoulder_w)
                    
                    if ear_angles:
                        features["ear_angle"] = float(np.mean(ear_angles))
                        features["ear_dist_norm"] = float(np.mean(ear_dists))
                    
                    if le and re:
                        le_n = self._rotate_point(le, mid, -roll)
                        re_n = self._rotate_point(re, mid, -roll)
                        features["head_tilt"] = (le_n[1] - re_n[1]) / shoulder_w
                    
                    features["shoulder_y_norm"] = mid[1] / h
                    features["roll_deg"] = roll
                    
                    # Model prediction
                    if self.model_loaded and len(features) >= len(self.feature_cols):
                        feat_vals = [features.get(c, 0) for c in self.feature_cols]
                        probs = self.model.predict_proba([feat_vals])[0]
                        idx = int(np.argmax(probs))
                        state = str(self.model.classes_[idx])
                        confidence = float(probs[idx])
                    else:
                        # Fallback rules
                        neck = features.get("neck_angle", 0)
                        ear = features.get("ear_angle", 0)
                        if neck < 15 and ear < 35:
                            state, confidence = "neutral", 0.7
                        elif neck > 30 or ear > 50:
                            state, confidence = "lean", 0.6
                        else:
                            state, confidence = "slouch", 0.6
                
                # Draw on frame
                color_map = {
                    "neutral": (0, 200, 0),
                    "slouch": (0, 200, 255),
                    "lean": (0, 0, 255),
                    "away": (128, 128, 128)
                }
                color = color_map.get(state, (128, 128, 128))
                
                # Draw keypoints
                for pt in [ls, rs, nose, le, re]:
                    if pt:
                        cv2.circle(frame, pt, 6, color, -1)
                        cv2.circle(frame, pt, 8, (255, 255, 255), 2)
                
                # Draw lines
                if ls and rs:
                    cv2.line(frame, ls, rs, color, 2)
                if nose and mid:
                    cv2.line(frame, mid, nose, (255, 255, 0), 2)
                if le and ls:
                    cv2.line(frame, ls, le, color, 2)
                if re and rs:
                    cv2.line(frame, rs, re, color, 2)
        
        # Smooth state
        self.state_history.append(state)
        if len(self.state_history) >= 3:
            # Majority vote
            from collections import Counter
            state = Counter(self.state_history).most_common(1)[0][0]
        
        # Draw status on frame
        status_colors = {
            "neutral": (0, 200, 0),
            "slouch": (0, 200, 255),
            "lean": (0, 0, 255),
            "away": (128, 128, 128)
        }
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{state.upper()} ({confidence:.0%})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_colors.get(state, (255,255,255)), 2)
        
        # Encode frame to JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        return jpeg.tobytes(), {
            "state": state,
            "confidence": round(confidence, 2),
            "present": present,
            "camera_ok": True,
            "model_loaded": self.model_loaded,
        }


# Global engine
engine = None


def get_engine():
    global engine
    if engine is None:
        engine = PostureEngine()
    return engine


@app.get("/")
def root():
    return {"status": "ok", "service": "StudyWise API"}


@app.get("/api/status")
def status():
    e = get_engine()
    return {
        "camera_ok": e.camera_ok,
        "model_loaded": e.model_loaded,
    }


@app.websocket("/ws/stream")
async def stream_websocket(websocket: WebSocket):
    """WebSocket: streams video frames + posture data."""
    await websocket.accept()
    
    e = get_engine()
    if not e.start_camera():
        await websocket.send_json({"error": "Camera not available"})
        await websocket.close()
        return
    
    session_start = time.time()
    focused_time = 0.0
    away_time = 0.0
    last_time = time.time()
    
    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            
            jpeg_bytes, posture = e.process_frame()
            
            if jpeg_bytes is None:
                await websocket.send_json({"error": "Camera error"})
                break
            
            # Track time
            if posture["state"] in ("neutral", "slouch"):
                focused_time += dt
            elif posture["state"] == "away":
                away_time += dt
            
            # Send frame + data
            await websocket.send_json({
                "type": "frame",
                "image": base64.b64encode(jpeg_bytes).decode('utf-8'),
                "posture": posture,
                "session_seconds": int(now - session_start),
                "focused_seconds": int(focused_time),
                "away_seconds": int(away_time),
            })
            
            # Check for messages
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                if msg.get("action") == "reset":
                    session_start = time.time()
                    focused_time = 0.0
                    away_time = 0.0
                elif msg.get("action") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.066)  # ~15 FPS
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as ex:
        print(f"Error: {ex}")
    finally:
        e.stop_camera()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
