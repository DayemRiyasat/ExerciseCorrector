from flask import Flask, render_template, request, jsonify
import base64
import cv2
import gc
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime

import numpy as np

from video_processor import VideoProcessor
from exercises.squat.SquatPredictor import SquatPredictor
from exercises.lunge.LungePredictor import LungePredictor
from exercises.pushup.PushupPredictor import PushupPredictor
from exercises.deadlift.DeadliftPredictor import DeadliftPredictor
from exercises.bicep_curl.BicepCurlPredictor import BicepCurlPredictor

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "formcheck_sessions.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("formcheck")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

current_predictor = None
video_processor = None

EXERCISE_REGISTRY = {
    "squat": {
        "label": "Squat",
        "short": "SQ",
        "difficulty": "Foundational",
        "focus": "Depth, knee tracking, back position",
        "recommended_view": "Side view with full body visible",
        "predictor": SquatPredictor,
    },
    "lunge": {
        "label": "Lunge",
        "short": "LU",
        "difficulty": "Intermediate",
        "focus": "Stride length, torso control, knee position",
        "recommended_view": "Side view or slight front angle",
        "predictor": LungePredictor,
    },
    "pushup": {
        "label": "Push-up",
        "short": "PU",
        "difficulty": "Foundational",
        "focus": "Body line, hand placement, hip position",
        "recommended_view": "Side view at floor level",
        "predictor": PushupPredictor,
    },
    "deadlift": {
        "label": "Deadlift",
        "short": "DL",
        "difficulty": "Advanced",
        "focus": "Hip hinge, spine position, stance width",
        "recommended_view": "Side view with bar and full body visible",
        "predictor": DeadliftPredictor,
    },
    "bicep_curl": {
        "label": "Bicep Curl",
        "short": "BC",
        "difficulty": "Foundational",
        "focus": "Elbow stability, torso control, range of motion",
        "recommended_view": "Front or slight side view",
        "predictor": BicepCurlPredictor,
    },
}

FEEDBACK_MESSAGES = {
    'squat': {
        'none': {
            'status': 'EXCELLENT FORM',
            'message': 'Perfect squat form!',
            'tips': ['Keep back straight', 'Chest up', 'Knees aligned with toes', 'Great work!'],
            'color': 'success'
        },
        'extreme_backward_lean': {
            'status': 'BACK ISSUE',
            'message': 'Don\'t lean backward',
            'tips': ['Engage core muscles', 'Maintain neutral spine', 'Keep weight centered'],
            'color': 'danger'
        },
        'extreme_forward_lean': {
            'status': 'FORWARD LEAN',
            'message': 'Don\'t lean too far forward',
            'tips': ['Keep chest up', 'Look forward', 'Sit back into the squat'],
            'color': 'warning'
        },
        'foots_too_close': {
            'status': 'STANCE TOO NARROW',
            'message': 'Widen your stance',
            'tips': ['Feet shoulder-width apart', 'Toes slightly pointed out'],
            'color': 'warning'
        },
        'foots_too_far': {
            'status': 'STANCE TOO WIDE',
            'message': 'Narrow your stance',
            'tips': ['Bring feet closer', 'Maintain control'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },
    'lunge': {   
        'none': {
            'status': 'EXCELLENT FORM',
            'message': 'Perfect lunge form!',
            'tips': ['Keep back straight', 'Front knee over ankle', 'Back knee bent', 'Great work!'],
            'color': 'success'
        },
        'extreme_backward_lean': {
            'status': 'BACK ISSUE',
            'message': 'Don\'t lean backward',
            'tips': ['Engage core muscles', 'Keep torso upright', 'Look forward'],
            'color': 'danger'
        },
        'extreme_forward_lean': {
            'status': 'FORWARD LEAN',
            'message': 'Don\'t lean too far forward',
            'tips': ['Keep chest up', 'Shoulders back', 'Stay vertical'],
            'color': 'warning'
        },
        'foots_too_close': {
            'status': 'STANCE TOO NARROW',
            'message': 'Step further forward',
            'tips': ['Increase stride length', 'Front foot should be forward'],
            'color': 'warning'
        },
        'foots_too_far': {
            'status': 'STANCE TOO WIDE',
            'message': 'Reduce stride length',
            'tips': ['Step closer', 'Maintain balance'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },  
    'pushup': {  
        'none': {
            'status': 'EXCELLENT FORM',
            'message': 'Perfect push-up form!',
            'tips': ['Elbows close to body', 'Straight back', 'Full range of motion', 'Great work!'],
            'color': 'success'
        },
        'hand_too_far_or_incorrect_position': {
            'status': 'HAND POSITION',
            'message': 'Adjust hand placement',
            'tips': ['Hands shoulder-width apart', 'Position under shoulders', 'Fingers forward'],
            'color': 'warning'
        },
        'hips_too_high': {
            'status': 'HIP POSITION',
            'message': 'Lower your hips',
            'tips': ['Maintain plank position', 'Keep core engaged', 'Straight line head to heels'],
            'color': 'warning'
        },
        'incorrect_leg_position': {
            'status': 'LEG ALIGNMENT',
            'message': 'Check leg position',
            'tips': ['Keep legs straight', 'Feet together', 'Toes on ground'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },
     'deadlift': {   
        'none': {
            'status': 'EXCELLENT FORM',
            'message': 'Perfect deadlift form!',
            'tips': ['Neutral spine', 'Chest up', 'Hips and shoulders rise together', 'Great work!'],
            'color': 'success'
        },
        'back_arch_posture': {
            'status': 'BACK ARCH CRITICAL',
            'message': 'Keep spine neutral!',
            'tips': ['Engage core', 'Chest up', 'Don\'t hyperextend back', 'Maintain neutral spine throughout'],
            'color': 'danger'
        },
        'hand_grip_width': {
            'status': 'GRIP WIDTH',
            'message': 'Adjust hand position',
            'tips': ['Hands shoulder-width or slightly wider', 'Arms straight', 'Grip outside knees'],
            'color': 'warning'
        },
        'leg_position_width': {
            'status': 'STANCE WIDTH',
            'message': 'Adjust foot position',
            'tips': ['Feet hip-width apart', 'Toes slightly out', 'Weight on mid-foot'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure full body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    },
    'bicep_curl': {   
        'none': {
            'status': 'EXCELLENT FORM',
            'message': 'Perfect bicep curl form!',
            'tips': ['Elbows stable', 'Controlled movement', 'No momentum', 'Great work!'],
            'color': 'success'
        },
        'back_too_backward_lean': {
            'status': 'BACKWARD LEAN',
            'message': 'Don\'t lean backward!',
            'tips': ['Engage core', 'Stand upright', 'No momentum', 'Control the weight'],
            'color': 'danger'
        },
        'back_too_forward_lean': {
            'status': 'FORWARD LEAN',
            'message': 'Don\'t lean forward!',
            'tips': ['Keep torso upright', 'Shoulders back', 'Engage core'],
            'color': 'danger'
        },
        'hand_position_too_close': {
            'status': 'HANDS TOO CLOSE',
            'message': 'Widen your grip',
            'tips': ['Hands shoulder-width apart', 'Natural grip width'],
            'color': 'warning'
        },
        'hand_position_too_wide': {
            'status': 'HANDS TOO WIDE',
            'message': 'Narrow your grip',
            'tips': ['Bring hands closer', 'Shoulder-width grip'],
            'color': 'warning'
        },
        'hand_above_near_head': {
            'status': 'OVER-CURLING',
            'message': 'Don\'t curl too high',
            'tips': ['Stop at shoulder level', 'Don\'t swing weights', 'Control the motion'],
            'color': 'warning'
        },
        'one_hand_up_other_down': {
            'status': 'ASYMMETRIC',
            'message': 'Keep both hands level',
            'tips': ['Curl both arms together', 'Maintain symmetry', 'Equal weight on both sides'],
            'color': 'warning'
        },
        'unknown': {
            'status': 'UNCLEAR POSITION',
            'message': 'Position yourself in frame',
            'tips': ['Ensure upper body visible', 'Stand in good lighting'],
            'color': 'secondary'
        }
    }

}


def init_database():
    """Create the local SQLite session history table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                exercise TEXT NOT NULL,
                mode TEXT NOT NULL,
                score INTEGER NOT NULL,
                confidence REAL NOT NULL,
                reps INTEGER DEFAULT 0,
                main_issue TEXT,
                summary TEXT,
                payload TEXT
            )
            """
        )
        conn.commit()


def get_predictor_for_exercise(exercise_type):
    """Return the predictor class configured for the selected exercise."""
    exercise = EXERCISE_REGISTRY.get(exercise_type)
    if not exercise:
        return None
    return exercise["predictor"]()


def clean_prediction_label(prediction):
    if not prediction or prediction == "none":
        return "Good form"
    return str(prediction).replace("_", " ").title()


def calculate_form_score(prediction, confidence, rep_info=None):
    """Convert model confidence and prediction class into a simple portfolio-friendly score."""
    confidence = max(0.0, min(float(confidence or 0.0), 1.0))
    if prediction == "none":
        score = 82 + int(confidence * 18)
    elif prediction == "unknown":
        score = 0
    else:
        score = 45 + int(confidence * 30)

    if rep_info and isinstance(rep_info, dict) and rep_info.get("form_quality") is not None:
        rep_quality = max(0.0, min(float(rep_info.get("form_quality", 0.0)), 1.0))
        score = int((score * 0.65) + (rep_quality * 100 * 0.35))

    return max(0, min(score, 100))


def status_from_score(score):
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Needs minor work"
    if score > 0:
        return "Needs review"
    return "Not detected"


def build_metrics(analysis_details):
    """Map each predictor's detailed values into a common frontend metrics structure."""
    if not isinstance(analysis_details, dict):
        return []

    metric_map = [
        ("knee_angle", "Knee angle", "°"),
        ("front_knee_angle", "Front knee", "°"),
        ("back_knee_angle", "Back knee", "°"),
        ("hip_angle", "Hip angle", "°"),
        ("front_hip_angle", "Front hip", "°"),
        ("elbow_angle", "Elbow angle", "°"),
        ("back_angle", "Back angle", "°"),
        ("shoulder_angle", "Shoulder angle", "°"),
        ("lean_amount", "Lean amount", "%"),
    ]

    metrics = []
    for key, label, suffix in metric_map:
        if key in analysis_details and analysis_details[key] is not None:
            value = analysis_details[key]
            metrics.append({"key": key, "label": label, "value": value, "suffix": suffix})

    for key in ["depth_achieved", "stance_width", "back_lean", "hand_position", "hip_position", "leg_position"]:
        if key in analysis_details and analysis_details[key] is not None:
            label = key.replace("_", " ").title()
            value = analysis_details[key]
            if isinstance(value, bool):
                value = "Good" if value else "Needs work"
            metrics.append({"key": key, "label": label, "value": value, "suffix": ""})

    return metrics[:8]


def build_body_feedback(exercise_type, prediction, analysis_details):
    """Create human-readable body-area feedback for the right panel."""
    details = analysis_details if isinstance(analysis_details, dict) else {}
    is_good = prediction == "none"

    if exercise_type == "squat":
        return [
            {"area": "Knees", "status": "Good" if is_good else "Check", "note": "Track knees over toes and avoid inward collapse."},
            {"area": "Hips", "status": "Good" if details.get("depth_achieved") else "Review", "note": "Aim for consistent depth while staying balanced."},
            {"area": "Back", "status": "Good" if details.get("back_lean") in ["Neutral", None] else "Check", "note": "Keep a neutral spine and steady chest position."},
            {"area": "Stance", "status": str(details.get("stance_width", "Check")), "note": "Use a stable shoulder-width stance."},
        ]
    if exercise_type == "lunge":
        return [
            {"area": "Front knee", "status": "Good" if is_good else "Check", "note": "Keep the front knee stacked over the ankle."},
            {"area": "Stride", "status": str(details.get("stance_width", "Check")), "note": "Use a stride long enough to stay balanced."},
            {"area": "Torso", "status": "Good" if details.get("back_lean") in ["Neutral", None] else "Check", "note": "Keep your torso tall through the movement."},
        ]
    if exercise_type == "pushup":
        return [
            {"area": "Hands", "status": "Good" if prediction != "hand_too_far_or_incorrect_position" else "Check", "note": "Place hands close to shoulder-width."},
            {"area": "Hips", "status": "Good" if prediction != "hips_too_high" else "Review", "note": "Maintain a straight line from shoulders to heels."},
            {"area": "Core", "status": "Good" if is_good else "Check", "note": "Brace the core to reduce sagging or piking."},
        ]
    if exercise_type == "deadlift":
        return [
            {"area": "Back", "status": "Good" if prediction != "back_arch_posture" else "Review", "note": "Keep the spine neutral through the hinge."},
            {"area": "Grip", "status": "Good" if prediction != "hand_grip_width" else "Check", "note": "Grip just outside the legs."},
            {"area": "Stance", "status": "Good" if prediction != "leg_position_width" else "Check", "note": "Keep feet around hip-width apart."},
        ]
    if exercise_type == "bicep_curl":
        return [
            {"area": "Elbows", "status": "Good" if is_good else "Check", "note": "Keep elbows stable beside the torso."},
            {"area": "Torso", "status": "Good" if "lean" not in str(prediction) else "Review", "note": "Avoid using body momentum to move the weight."},
            {"area": "Symmetry", "status": "Good" if prediction != "one_hand_up_other_down" else "Check", "note": "Keep both arms moving evenly."},
        ]
    return []


def build_rep_timeline(rep_info, prediction, confidence):
    """Return a small current-rep event that the frontend can add to the timeline."""
    if not isinstance(rep_info, dict):
        return None

    if rep_info.get("rep_counted"):
        rep_number = int(rep_info.get("rep_count", 0))
        quality = rep_info.get("form_quality", confidence or 0)
        return {
            "rep": rep_number,
            "status": "Good" if prediction == "none" else "Review",
            "score": int(max(0, min(float(quality or 0), 1)) * 100),
            "issue": clean_prediction_label(prediction),
        }
    return None


def camera_guidance_for_result(result):
    if not result.get("success"):
        return [
            {"label": "Full body visible", "status": "Check"},
            {"label": "Lighting", "status": "Check"},
            {"label": "Camera stability", "status": "Check"},
        ]
    confidence = float(result.get("confidence") or 0)
    return [
        {"label": "Full body visible", "status": "Good" if confidence >= 0.45 else "Check"},
        {"label": "Lighting", "status": "Good" if confidence >= 0.55 else "Improve"},
        {"label": "Camera stability", "status": "Good"},
    ]


def enrich_result(result, exercise_type, mode="live"):
    """Add portfolio-facing fields while preserving the original response shape."""
    if not isinstance(result, dict):
        result = {"success": False, "error": "Invalid model response"}

    if result.get("success"):
        prediction = result.get("prediction", "unknown")
        confidence = float(result.get("confidence") or 0)
        feedback = FEEDBACK_MESSAGES.get(exercise_type, {}).get(
            prediction,
            FEEDBACK_MESSAGES.get(exercise_type, {}).get("unknown", {
                "status": "UNCLEAR POSITION",
                "message": "Position yourself in frame",
                "tips": ["Ensure the full body is visible", "Use steady lighting"],
                "color": "secondary",
            }),
        ).copy()

        analysis_details = result.get("analysis_details") or {}
        score = calculate_form_score(prediction, confidence, result.get("rep_info"))

        result.update({
            "exercise": exercise_type,
            "mode": mode,
            "feedback": feedback,
            "form_score": score,
            "score_status": status_from_score(score),
            "main_issue": clean_prediction_label(prediction),
            "metrics": build_metrics(analysis_details),
            "body_feedback": build_body_feedback(exercise_type, prediction, analysis_details),
            "camera_guidance": camera_guidance_for_result(result),
            "explainability": build_explainability(prediction, confidence, score, feedback),
            "rep_event": build_rep_timeline(result.get("rep_info"), prediction, confidence),
            "model_version": "prototype-v1",
        })
    else:
        result.update({
            "exercise": exercise_type,
            "mode": mode,
            "form_score": 0,
            "score_status": "Not detected",
            "main_issue": "No pose detected",
            "metrics": [],
            "body_feedback": [],
            "camera_guidance": camera_guidance_for_result(result),
        })
    return result


def build_explainability(prediction, confidence, score, feedback):
    confidence_pct = int(max(0, min(float(confidence or 0), 1)) * 100)
    if prediction == "none":
        reason = "The model classified the frame as correct form and the movement stayed within the expected posture range."
    elif prediction == "unknown":
        reason = "The system could not identify a stable pose with enough confidence."
    else:
        reason = f"The main detected issue was {clean_prediction_label(prediction).lower()}, so the form score was reduced."
    return {
        "reason": reason,
        "confidence_note": f"Model confidence for this reading is {confidence_pct}%.",
        "score_note": f"The current form score is {score}/100.",
        "primary_tip": feedback.get("tips", ["Repeat the movement with steady camera framing."])[0],
    }


def decode_data_url(data_url, media_label="frame"):
    if not data_url or not isinstance(data_url, str) or "," not in data_url:
        raise ValueError(f"No valid {media_label} data provided")
    try:
        encoded = data_url.split(",", 1)[1]
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError(f"Failed to decode {media_label}") from exc
    if frame is None:
        raise ValueError(f"Failed to decode {media_label}")
    return frame


def resize_frame(frame, min_width=None, max_width=1280):
    height, width = frame.shape[:2]
    if max_width and width > max_width:
        scale = max_width / width
        return cv2.resize(frame, (max_width, int(height * scale)))
    if min_width and width < min_width:
        scale = min_width / width
        return cv2.resize(frame, (min_width, int(height * scale)))
    return frame


def save_session_record(exercise, mode, score, confidence, reps=0, main_issue="", summary="", payload=None):
    session_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, created_at, exercise, mode, score, confidence, reps, main_issue, summary, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                exercise,
                mode,
                int(score or 0),
                float(confidence or 0),
                int(reps or 0),
                main_issue or "",
                summary or "",
                json.dumps(payload or {}, default=str),
            ),
        )
        conn.commit()
    return session_id


@app.route("/")
def index():
    return render_template("index.html", exercises=EXERCISE_REGISTRY)


@app.route("/api/exercises", methods=["GET"])
def list_exercises():
    return jsonify({"success": True, "exercises": EXERCISE_REGISTRY})


@app.route("/api/load_model", methods=["POST"])
def load_model():
    global current_predictor, video_processor

    data = request.get_json(silent=True) or {}
    exercise_type = data.get("exercise_type")

    if exercise_type not in EXERCISE_REGISTRY:
        return jsonify({"success": False, "error": "Unsupported exercise selected"}), 400

    if current_predictor:
        current_predictor.cleanup()
        current_predictor = None
        gc.collect()

    if video_processor:
        video_processor.stop()
        video_processor = None

    current_predictor = get_predictor_for_exercise(exercise_type)
    if not current_predictor:
        return jsonify({"success": False, "error": f"Exercise type {exercise_type} is not implemented"}), 400

    start = time.perf_counter()
    success = current_predictor.load_model()
    load_ms = int((time.perf_counter() - start) * 1000)

    if success:
        logger.info("Loaded %s model in %sms", exercise_type, load_ms)
        return jsonify({
            "success": True,
            "message": f"{EXERCISE_REGISTRY[exercise_type]['label']} model loaded",
            "exercise": exercise_type,
            "exercise_meta": {k: v for k, v in EXERCISE_REGISTRY[exercise_type].items() if k != "predictor"},
            "model_version": "prototype-v1",
            "load_ms": load_ms,
        })

    current_predictor = None
    logger.warning("Failed to load %s model", exercise_type)
    return jsonify({"success": False, "error": f"Failed to load {exercise_type} model"}), 500


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    global current_predictor

    if not current_predictor:
        return jsonify({"success": False, "error": "No model loaded"}), 400

    data = request.get_json(silent=True) or {}
    frame_data = data.get("frame")

    try:
        frame = decode_data_url(frame_data, "frame")
        frame = resize_frame(frame, max_width=1280)
        start = time.perf_counter()
        result = current_predictor.process_frame(frame)
        result = enrich_result(result, current_predictor.exercise_type, mode="webcam")
        result["processing_ms"] = int((time.perf_counter() - start) * 1000)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Error processing frame")
        return jsonify({"success": False, "error": str(exc), "message": "Unable to process this frame"}), 400


@app.route("/api/process_image", methods=["POST"])
def process_image():
    global current_predictor

    if not current_predictor:
        return jsonify({"success": False, "error": "No model loaded"}), 400

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    try:
        frame = decode_data_url(image_data, "image")
        frame = resize_frame(frame, min_width=640, max_width=1280)
        start = time.perf_counter()
        result = current_predictor.process_image(frame)
        result = enrich_result(result, current_predictor.exercise_type, mode="image")
        result["processing_ms"] = int((time.perf_counter() - start) * 1000)

        if result.get("success"):
            result["session_id"] = save_session_record(
                exercise=current_predictor.exercise_type,
                mode="image",
                score=result.get("form_score", 0),
                confidence=result.get("confidence", 0),
                reps=0,
                main_issue=result.get("main_issue", ""),
                summary=result.get("explainability", {}).get("reason", ""),
                payload={
                    "metrics": result.get("metrics", []),
                    "body_feedback": result.get("body_feedback", []),
                    "model_version": result.get("model_version"),
                },
            )
        return jsonify(result)
    except Exception as exc:
        logger.exception("Error processing image")
        return jsonify({"success": False, "error": str(exc), "message": "Unable to analyze this image"}), 400


@app.route("/api/save_session", methods=["POST"])
def save_session():
    data = request.get_json(silent=True) or {}
    exercise = data.get("exercise") or (current_predictor.exercise_type if current_predictor else "unknown")
    mode = data.get("mode", "manual")

    if exercise not in EXERCISE_REGISTRY and exercise != "demo":
        return jsonify({"success": False, "error": "Unsupported exercise"}), 400

    session_id = save_session_record(
        exercise=exercise,
        mode=mode,
        score=data.get("score", 0),
        confidence=data.get("confidence", 0),
        reps=data.get("reps", 0),
        main_issue=data.get("main_issue", ""),
        summary=data.get("summary", ""),
        payload=data.get("payload", {}),
    )
    return jsonify({"success": True, "session_id": session_id})


@app.route("/api/session_history", methods=["GET"])
def session_history():
    limit = min(int(request.args.get("limit", 8)), 50)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, exercise, mode, score, confidence, reps, main_issue, summary
            FROM sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return jsonify({"success": True, "sessions": [dict(row) for row in rows]})


@app.route("/api/demo_analysis", methods=["POST"])
def demo_analysis():
    data = request.get_json(silent=True) or {}
    exercise = data.get("exercise_type") if data.get("exercise_type") in EXERCISE_REGISTRY else "squat"
    demo = {
        "success": True,
        "exercise": exercise,
        "mode": "demo",
        "prediction": "none",
        "confidence": 0.92,
        "form_score": 91,
        "score_status": "Strong",
        "main_issue": "Good form",
        "feedback": {
            "status": "DEMO ANALYSIS",
            "message": "Sample session loaded",
            "tips": [
                "Use this mode to show the product without uploading media",
                "Replace the sample result with a real saved session later",
                "Keep the camera side-on for most compound lifts",
            ],
            "color": "success",
        },
        "metrics": [
            {"key": "knee_angle", "label": "Knee angle", "value": 96, "suffix": "°"},
            {"key": "hip_angle", "label": "Hip angle", "value": 104, "suffix": "°"},
            {"key": "back_angle", "label": "Back angle", "value": 84, "suffix": "°"},
            {"key": "depth_achieved", "label": "Depth", "value": "Good", "suffix": ""},
        ],
        "body_feedback": [
            {"area": "Knees", "status": "Good", "note": "Knees stayed aligned through the rep."},
            {"area": "Hips", "status": "Good", "note": "Depth was consistent and controlled."},
            {"area": "Back", "status": "Good", "note": "Torso angle stayed stable."},
        ],
        "rep_timeline": [
            {"rep": 1, "status": "Good", "score": 88, "issue": "Good form"},
            {"rep": 2, "status": "Good", "score": 91, "issue": "Good form"},
            {"rep": 3, "status": "Review", "score": 74, "issue": "Slight depth inconsistency"},
            {"rep": 4, "status": "Good", "score": 93, "issue": "Good form"},
        ],
        "camera_guidance": [
            {"label": "Full body visible", "status": "Good"},
            {"label": "Lighting", "status": "Good"},
            {"label": "Camera stability", "status": "Good"},
        ],
        "explainability": {
            "reason": "This sample shows how a completed session summary will appear after analysis.",
            "confidence_note": "Model confidence for this demo reading is 92%.",
            "score_note": "The demo form score is 91/100.",
            "primary_tip": "Keep the movement controlled from start to finish.",
        },
        "model_version": "demo-v1",
    }
    demo["session_id"] = save_session_record(
        exercise=exercise,
        mode="demo",
        score=demo["form_score"],
        confidence=demo["confidence"],
        reps=len(demo["rep_timeline"]),
        main_issue=demo["main_issue"],
        summary=demo["explainability"]["reason"],
        payload={"metrics": demo["metrics"], "body_feedback": demo["body_feedback"]},
    )
    return jsonify(demo)


@app.route("/api/start_video_processing", methods=["POST"])
def start_video_processing():
    global current_predictor, video_processor

    if not current_predictor:
        return jsonify({"success": False, "error": "No model loaded"}), 400

    if video_processor:
        video_processor.stop()

    video_processor = VideoProcessor(current_predictor)
    video_processor.start()
    current_predictor.reset_counter()

    return jsonify({"success": True, "message": "Video processor started"})


@app.route("/api/process_video_frame", methods=["POST"])
def process_video_frame():
    global video_processor

    if not video_processor:
        return jsonify({"success": False, "error": "Video processor not initialized"}), 400

    data = request.get_json(silent=True) or {}
    frame_data = data.get("frame")
    frame_number = data.get("frame_number", 0)
    timestamp = data.get("timestamp", 0)

    if not frame_data:
        return jsonify({"success": False, "error": "No frame data provided"}), 400

    success = video_processor.add_frame(frame_data, frame_number, timestamp)
    return jsonify({"success": success, "message": "Frame queued" if success else "Queue full"})


@app.route("/api/get_video_result", methods=["GET"])
def get_video_result():
    global video_processor

    if not video_processor:
        return jsonify({"success": False, "error": "Video processor not initialized"}), 400

    result = video_processor.get_result(timeout=0.05)
    if result:
        result = enrich_result(result, current_predictor.exercise_type, mode="video")
        return jsonify(result)
    return jsonify({"success": False, "no_result": True})


@app.route("/api/stop_video_processing", methods=["POST"])
def stop_video_processing():
    global video_processor

    if video_processor:
        video_processor.stop()
        video_processor = None

    return jsonify({"success": True, "message": "Video processor stopped"})


@app.route("/api/pause_video_processing", methods=["POST"])
def pause_video_processing():
    global video_processor

    if not video_processor:
        return jsonify({"success": False, "error": "Video processor not initialized"}), 400
    video_processor.pause()
    return jsonify({"success": True, "message": "Video processor paused"})


@app.route("/api/resume_video_processing", methods=["POST"])
def resume_video_processing():
    global video_processor

    if not video_processor:
        return jsonify({"success": False, "error": "Video processor not initialized"}), 400
    video_processor.resume()
    return jsonify({"success": True, "message": "Video processor resumed"})


@app.route("/api/reset_counter", methods=["POST"])
def reset_counter():
    global current_predictor

    if not current_predictor:
        return jsonify({"success": False, "error": "No model loaded"}), 400

    current_predictor.reset_counter()
    return jsonify({"success": True, "message": "Rep counter reset", "rep_count": 0})


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": current_predictor is not None,
        "exercise_type": current_predictor.exercise_type if current_predictor else None,
        "rep_count": current_predictor.get_rep_count() if current_predictor else 0,
        "video_processor_active": video_processor is not None,
        "session_store": os.path.exists(DB_PATH),
        "available_exercises": list(EXERCISE_REGISTRY.keys()),
    })


init_database()

if __name__ == "__main__":
    logger.info("Starting FormCheck on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
