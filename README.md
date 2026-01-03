# StudyWise 📚

A webcam-based study companion that tracks your posture and focus time. Get real-time feedback on your posture (neutral, slouch, lean, away) and see how long you've been focused during study sessions.

![StudyWise Demo](https://img.shields.io/badge/Status-Active-green)

## Features

- **Real-time posture detection** - Uses MediaPipe + ML to classify your posture
- **Focus tracking** - Tracks total session time vs focused time
- **Live video feed** - See exactly what the model sees with pose overlay
- **Lightweight** - Runs at ~15 FPS with minimal CPU usage
- **Train your own model** - Collect data and train a personalized posture model

## Quick Start

### Prerequisites

- **Python 3.11** (required for MediaPipe compatibility)
- **Node.js 18+** and npm
- **Webcam**

### 1. Clone and Setup Python Environment

```bash
# Clone the repo
git clone https://github.com/yourusername/studywise.git
cd studywise

# Create Python 3.11 virtual environment
python3.11 -m venv .venv311

# Activate it
source .venv311/bin/activate  # Mac/Linux
# or
.venv311\Scripts\activate     # Windows

# Install Python dependencies
pip install --upgrade pip
pip install mediapipe==0.10.14 opencv-python numpy pandas scikit-learn joblib fastapi uvicorn websockets
```

### 2. Setup Frontend

```bash
cd frontend
npm install
cd ..
```

### 3. Run the App

You need **two terminal windows**:

**Terminal 1 - Backend API:**

```bash
cd studywise
source .venv311/bin/activate
uvicorn backend.api.server:app --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd studywise/frontend
npm run dev
```

Then open **http://localhost:3000** (or the port shown in terminal) in your browser.

---

## Training Your Own Model

The app comes with a pre-trained model, but you can train your own for better accuracy with your specific setup (camera angle, lighting, your body).

### Step 1: Collect Training Data

```bash
cd studywise
source .venv311/bin/activate
python backend/prototypes/posture_collect_data_v2.py
```

**Controls:**

- `c` - Calibrate (do this first! Sit upright for 3 seconds)
- `1` - Label as **neutral** (sitting upright)
- `2` - Label as **slouch** (rounded back, droopy shoulders)
- `3` - Label as **lean** (head pushed forward toward screen)
- `4` - Label as **away** (not at desk / not visible)
- `0` - Pause labeling
- `q` - Quit

**Tips for good data:**

- Collect **~2-3 minutes per posture** minimum
- Keep classes **balanced** (similar sample counts)
- **Exaggerate** each posture so they're distinct
- The screen shows sample counts - aim for ~3000+ per class

Data saves to `data/posture_data_v2.csv`.

### Step 2: Train the Model

```bash
python backend/prototypes/train_posture_model_v2.py
```

This will:

- Load your collected data
- Train a Random Forest classifier
- Show accuracy metrics and confusion matrix
- Save the model to `backend/models/posture_model_v2.joblib`

**Good accuracy targets:**

- Overall: >90%
- Per-class: >80% each

### Step 3: Test It

Run the standalone posture monitor (no web UI):

```bash
python backend/prototypes/posture_live_timer_v2.py
```

Or start the web app (see Quick Start above).

---

## Project Structure

```
studywise/
├── backend/
│   ├── api/
│   │   └── server.py          # FastAPI server with WebSocket streaming
│   ├── models/
│   │   └── posture_model_v2.joblib  # Trained ML model
│   └── prototypes/
│       ├── posture_collect_data_v2.py   # Data collection script
│       ├── train_posture_model_v2.py    # Model training script
│       └── posture_live_timer_v2.py     # Standalone posture monitor
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React app
│   │   └── index.css          # Styles
│   └── package.json
├── data/
│   └── posture_data_v2.csv    # Training data (generated)
└── README.md
```

---

## How It Works

### Posture Detection Pipeline

1. **Webcam capture** → OpenCV reads frames at ~15 FPS
2. **Pose estimation** → MediaPipe detects body landmarks (shoulders, nose, ears)
3. **Feature extraction** → Calculate angles and normalized distances:
   - `neck_angle` - Angle from shoulder midpoint to nose
   - `ear_angle` - Angle from shoulders to ears
   - `nose_dist_norm` - Normalized head distance
   - `shoulder_y_norm` - Vertical shoulder position
   - Plus: head tilt, roll compensation, forward lean
4. **Classification** → Random Forest model predicts: neutral, slouch, lean, or away
5. **Smoothing** → Majority vote over last 5 frames to prevent flicker

### Posture Classes

| Class       | Description                           | Visual Cue |
| ----------- | ------------------------------------- | ---------- |
| **Neutral** | Upright, head balanced over shoulders | Green      |
| **Slouch**  | Rounded back, shoulders forward/down  | Yellow     |
| **Lean**    | Head pushed forward toward screen     | Red        |
| **Away**    | Not visible / not at desk             | Gray       |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mediapipe'"

Make sure you're using Python 3.11 and have the virtual environment activated:

```bash
source .venv311/bin/activate
pip install mediapipe==0.10.14
```

### "Camera not available"

- Check no other app is using your webcam
- Try restarting the backend server
- On Mac, ensure camera permissions are granted

### "WebSocket connection failed"

- Make sure the backend is running on port 8000
- Check for errors in the backend terminal

### Model accuracy is low

- Collect more training data (aim for 3000+ samples per class)
- Make sure postures are exaggerated and distinct
- Check class balance in training output

### Posture detection is flickery

The app already smooths predictions. If still flickery:

- Improve lighting
- Sit more centered in frame
- Retrain with more data

---

## Dependencies

### Python

- `mediapipe==0.10.14` - Pose estimation
- `opencv-python` - Video capture and processing
- `numpy` - Numerical operations
- `pandas` - Data handling
- `scikit-learn` - ML model training
- `joblib` - Model serialization
- `fastapi` - API server
- `uvicorn` - ASGI server
- `websockets` - Real-time streaming

### Node.js

- `react` - UI framework
- `vite` - Build tool
- `tailwindcss` - Styling
- `recharts` - Charts (optional dashboard)
- `lucide-react` - Icons

---

## One-Line Install Commands

**All Python dependencies:**

```bash
pip install mediapipe==0.10.14 opencv-python numpy pandas scikit-learn joblib fastapi uvicorn websockets
```

**Frontend:**

```bash
cd frontend && npm install
```

---
