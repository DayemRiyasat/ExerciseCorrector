# FormCheck

FormCheck is a Flask-based exercise form analysis web app that uses computer vision and machine learning to evaluate movement quality from webcam, image, or video input. The project uses MediaPipe pose landmarks to extract body-position features, then applies exercise-specific TensorFlow models and rule-based feedback to return a form score, confidence level, rep timeline, body-area feedback, and practical coaching notes.

The current version supports squat, lunge, push-up, deadlift, and bicep curl analysis.

## Project Overview

Most exercise-tracking apps count reps, but they do not explain what went wrong during the movement. FormCheck is designed to go one step further by combining pose detection, model inference, and structured feedback into a clean web interface.

Users can:

- Select an exercise model
- Analyze movement through webcam, uploaded video, or uploaded image
- View a form score out of 100
- See model confidence and movement metrics
- Review body-area feedback such as knees, hips, back, shoulders, and range of motion
- Track rep-by-rep feedback in a timeline
- Load a demo session without uploading media
- View recent analysis sessions stored locally with SQLite

## Features

### Frontend

- Minimal black interface with bright accent styling
- Cinematic landing section and product-style layout
- Exercise selection cards
- Webcam, video, and image analysis modes
- Results dashboard with score, confidence, and feedback panels
- Rep timeline display
- Camera setup and readiness checklist
- Recent session history section
- Demo session mode
- Responsive layout for different screen sizes

### Backend

- Flask API backend
- Exercise registry for supported movements
- MediaPipe pose landmark detection
- TensorFlow model loading for exercise classification
- Joblib scaler and label encoder support
- Standardized JSON response structure
- SQLite session history storage
- Upload validation and basic error handling
- Logging for model loading, analysis flow, and backend events
- Health check endpoint

### Supported Exercises

| Exercise | Focus Areas |
|---|---|
| Squat | Depth, knee tracking, back position |
| Lunge | Stride length, torso control, knee position |
| Push-up | Body line, hand placement, hip position |
| Deadlift | Hip hinge, spine position, stance width |
| Bicep Curl | Elbow stability, torso control, range of motion |

## Tech Stack

- Python
- Flask
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Joblib
- SQLite
- HTML
- CSS
- JavaScript

## How It Works

```text
User webcam / image / video input
        ↓
Flask backend receives frame or file
        ↓
MediaPipe detects body landmarks
        ↓
Exercise-specific feature extraction
        ↓
TensorFlow model predicts form class
        ↓
Rule-based metrics and feedback are added
        ↓
Frontend displays score, confidence, metrics, and coaching notes
```

## Project Structure

```text
FormCheck/
├── app.py
├── base_predictor.py
├── video_processor.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── exercises/
│   ├── squat/
│   ├── lunge/
│   ├── pushup/
│   ├── deadlift/
│   └── bicep_curl/
├── models/
├── data/
├── uploads/
└── README.md
```

Some folders, such as `data/`, `uploads/`, and local testing media, should usually be ignored by Git because they are generated or environment-specific.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If MediaPipe causes an import issue, reinstall it with:

```bash
python -m pip uninstall -y mediapipe
python -m pip install --upgrade "mediapipe>=0.10.14,<0.11"
```

### 4. Run the app

```bash
python app.py
```

Then open the local Flask URL in your browser, usually:

```text
http://127.0.0.1:5000
```

## API Endpoints

| Endpoint | Method | Purpose |
|---|---:|---|
| `/` | GET | Main web app |
| `/api/exercises` | GET | Returns supported exercise metadata |
| `/api/load_model` | POST | Loads the selected exercise model |
| `/api/process_frame` | POST | Processes a webcam frame |
| `/api/process_image` | POST | Processes an uploaded image |
| `/api/start_video_processing` | POST | Starts video analysis |
| `/api/process_video_frame` | POST | Processes a video frame |
| `/api/get_video_result` | GET | Returns video analysis results |
| `/api/stop_video_processing` | POST | Stops video analysis |
| `/api/pause_video_processing` | POST | Pauses video analysis |
| `/api/resume_video_processing` | POST | Resumes video analysis |
| `/api/reset_counter` | POST | Resets rep counter state |
| `/api/demo_analysis` | POST | Loads a sample demo result |
| `/api/session_history` | GET | Returns recent saved sessions |
| `/api/save_session` | POST | Saves a session result |
| `/api/health` | GET | Returns app health and model state |

## Example Analysis Output

A typical analysis response includes:

```json
{
  "success": true,
  "exercise": "squat",
  "form_score": 86,
  "confidence": 0.92,
  "main_issue": "Depth slightly shallow",
  "metrics": {
    "knee_angle": 94,
    "hip_angle": 101,
    "back_angle": 83
  },
  "body_feedback": [
    {
      "area": "Knees",
      "status": "Good",
      "detail": "Knee tracking appears stable."
    }
  ],
  "rep_timeline": [
    {
      "rep": 1,
      "status": "Good",
      "note": "Controlled movement."
    }
  ]
}
```

## Design Direction

The interface is designed to feel like a modern fitness-tech product rather than a basic computer vision demo. The visual style focuses on:

- Black background
- Minimal navigation
- Large typography
- Rounded cards
- Clean buttons
- High-contrast cyan and green accents
- Product-preview inspired dashboard sections
- No emoji-based UI labels

## Current Limitations

FormCheck is still a prototype and has limitations:

- It depends on good lighting and full-body visibility
- Results can be affected by camera angle and video quality
- Models may not generalize perfectly to all users, body types, gym setups, or movement variations
- Analysis should not be treated as medical, physiotherapy, or professional coaching advice
- Some exercise feedback is based on model predictions and rule-based estimates, not a certified biomechanics assessment

## Roadmap

Planned improvements include:

- More accurate rep-by-rep scoring
- Best-frame and worst-frame review
- Before-and-after comparison mode
- More detailed angle charts over time
- User accounts and cloud session history
- Model version tracking in the UI
- Expanded exercise library
- More unit tests for feature extraction and rep counting
- Background video processing queue for larger uploads
- Deployment-ready configuration

## Development Notes

Recommended Git workflow:

```bash
git checkout -b feature/session-history
# make changes
git add .
git commit -m "Add session history dashboard"
git push origin feature/session-history
```

Useful commit examples:

```text
Add modern landing page layout
Add exercise cards and input mode selector
Standardize backend analysis responses
Add SQLite session history
Add demo analysis mode
Improve MediaPipe startup compatibility
Add rep timeline and body-area feedback
```

## Safety Disclaimer

This project is for educational and portfolio purposes. It is not a substitute for a certified trainer, medical professional, physiotherapist, or rehabilitation specialist. Users should stop any exercise that causes pain and seek professional guidance when needed.

## Author

Built as a portfolio project to explore computer vision, pose estimation, machine learning, and full-stack web development.
