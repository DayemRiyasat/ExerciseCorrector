# base_predictor.py - Common base class for all exercises
# Location: project_root/base_predictor.py

import base64
import gc
import os
from importlib import import_module
from pathlib import Path

import cv2
import joblib
import numpy as np

# TensorFlow is deliberately NOT imported at module level. It is a ~1.2 GB
# install and is only needed to load a Keras .h5 file. On the server we load a
# converted .tflite model through a small runtime instead. TensorFlow is only
# imported lazily, as a local-development fallback, inside load_model().


class MediaPipeUnavailableError(RuntimeError):
    """Raised when the installed MediaPipe package does not expose pose utilities."""


class ModelBackendUnavailableError(RuntimeError):
    """Raised when no TFLite runtime and no TensorFlow install can load a model."""


def _get_tflite_interpreter_class():
    """Return a TFLite Interpreter class from whichever runtime is installed.

    Preference order, smallest install first:
      1. tflite-runtime      (a few MB, the right choice for deployment)
      2. ai-edge-litert      (Google's newer name for the same runtime)
      3. tensorflow.lite     (only if the full TensorFlow is already present)
    """
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except Exception:
        pass

    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except Exception:
        pass

    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except Exception:
        return None


class TFLiteModel:
    """Wraps a .tflite model so it exposes the same predict() call as Keras.

    This means predict() in ExercisePredictor does not need to know or care
    which backend loaded the model.
    """

    def __init__(self, model_path):
        interpreter_cls = _get_tflite_interpreter_class()
        if interpreter_cls is None:
            raise ModelBackendUnavailableError(
                "No TFLite runtime found. Install one with: "
                "python -m pip install tflite-runtime"
            )
        self.interpreter = interpreter_cls(model_path=str(model_path))
        self.interpreter.allocate_tensors()

    def predict(self, features, verbose=0):
        """Run inference. Signature matches keras.Model.predict for drop-in use."""
        batch = np.asarray(features, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        input_details = self.interpreter.get_input_details()[0]

        # The converted model has a fixed batch size of 1. Resize if a caller
        # ever passes a different batch shape.
        if tuple(input_details["shape"]) != batch.shape:
            self.interpreter.resize_tensor_input(input_details["index"], list(batch.shape))
            self.interpreter.allocate_tensors()
            input_details = self.interpreter.get_input_details()[0]

        self.interpreter.set_tensor(
            input_details["index"], batch.astype(input_details["dtype"])
        )
        self.interpreter.invoke()

        output_details = self.interpreter.get_output_details()[0]
        return np.array(self.interpreter.get_tensor(output_details["index"]))


_MEDIAPIPE_CACHE = {
    "loaded": False,
    "pose": None,
    "drawing": None,
    "error": None,
}

_DRAWING_SPEC_LANDMARK = None
_DRAWING_SPEC_CONNECTION = None
PROJECT_ROOT = Path(__file__).resolve().parent


def _try_load_mediapipe():
    """Load MediaPipe pose and drawing utilities without crashing app startup.

    The older MediaPipe Solutions API is required by this project. Some broken,
    partial, or incompatible installs let `import mediapipe` succeed but do not
    expose `mp.solutions`. This loader keeps the Flask app importable and gives
    a clear runtime error when the user tries to start analysis.
    """
    if _MEDIAPIPE_CACHE["loaded"]:
        return _MEDIAPIPE_CACHE["pose"], _MEDIAPIPE_CACHE["drawing"]
    if _MEDIAPIPE_CACHE["error"] is not None:
        raise MediaPipeUnavailableError(_MEDIAPIPE_CACHE["error"])

    errors = []

    # Preferred path for the classic Solutions API.
    try:
        pose_module = import_module("mediapipe.python.solutions.pose")
        drawing_module = import_module("mediapipe.python.solutions.drawing_utils")
        _MEDIAPIPE_CACHE.update({
            "loaded": True,
            "pose": pose_module,
            "drawing": drawing_module,
            "error": None,
        })
        return pose_module, drawing_module
    except Exception as exc:
        errors.append(f"mediapipe.python.solutions import failed: {exc}")

    # Fallback used by most MediaPipe examples: import mediapipe as mp.
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            version = getattr(mp, "__version__", "unknown")
            location = getattr(mp, "__file__", "unknown")
            raise AttributeError(
                f"mediapipe imported from {location}, version {version}, "
                "but it does not expose mp.solutions"
            )
        pose_module = mp.solutions.pose
        drawing_module = mp.solutions.drawing_utils
        _MEDIAPIPE_CACHE.update({
            "loaded": True,
            "pose": pose_module,
            "drawing": drawing_module,
            "error": None,
        })
        return pose_module, drawing_module
    except Exception as exc:
        errors.append(f"mp.solutions fallback failed: {exc}")

    message = (
        "MediaPipe pose utilities could not be loaded. This project requires the "
        "classic MediaPipe Solutions API. Reinstall MediaPipe in your active "
        "environment with: python -m pip uninstall -y mediapipe && "
        "python -m pip install --upgrade \"mediapipe>=0.10.14,<0.11\". "
        "Loader details: " + " | ".join(errors)
    )
    _MEDIAPIPE_CACHE["error"] = message
    raise MediaPipeUnavailableError(message)


def _get_drawing_specs():
    """Create drawing specs after MediaPipe drawing utilities are available."""
    global _DRAWING_SPEC_LANDMARK, _DRAWING_SPEC_CONNECTION

    _, drawing_module = _try_load_mediapipe()
    if _DRAWING_SPEC_LANDMARK is None:
        _DRAWING_SPEC_LANDMARK = drawing_module.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2
        )
    if _DRAWING_SPEC_CONNECTION is None:
        _DRAWING_SPEC_CONNECTION = drawing_module.DrawingSpec(
            color=(0, 0, 255), thickness=2
        )
    return _DRAWING_SPEC_LANDMARK, _DRAWING_SPEC_CONNECTION


class ExercisePredictor:
    """Base class for all exercise predictors."""

    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.mp_pose = None
        self.mp_drawing = None
        self.pose = None
        self.rep_counter = None

    def _ensure_mediapipe(self):
        """Attach MediaPipe modules to this predictor instance."""
        if self.mp_pose is None or self.mp_drawing is None:
            self.mp_pose, self.mp_drawing = _try_load_mediapipe()
        return self.mp_pose, self.mp_drawing

    def load_model(self):
        """Load ML model, scaler, label encoder, and MediaPipe pose tracker."""
        try:
            self._ensure_mediapipe()

            model_dir = PROJECT_ROOT / "exercises" / self.exercise_type / "models"
            tflite_path = model_dir / f"{self.exercise_type}_model.tflite"
            keras_path = model_dir / f"{self.exercise_type}_model.h5"
            scaler_path = model_dir / f"{self.exercise_type}_scaler.pkl"
            encoder_path = model_dir / f"{self.exercise_type}_label_encoder.pkl"

            missing_paths = [
                str(path) for path in (scaler_path, encoder_path) if not path.exists()
            ]
            if not tflite_path.exists() and not keras_path.exists():
                missing_paths.append(f"{tflite_path} (or {keras_path})")
            if missing_paths:
                print(f"Model files not found for {self.exercise_type}: {missing_paths}")
                return False

            # Prefer the .tflite model. It is what the server has, and it does
            # not drag TensorFlow into the process.
            if tflite_path.exists():
                self.model = TFLiteModel(tflite_path)
                print(f"Loaded TFLite model for {self.exercise_type}")
            else:
                # Local development fallback only. Never hit on the server,
                # because the .tflite files are committed alongside the .h5 files.
                import tensorflow as tf

                # These models are only used for inference. compile=False avoids
                # Keras/HDF5 deserialization issues from older training environments.
                self.model = tf.keras.models.load_model(str(keras_path), compile=False)
                print(f"Loaded Keras model for {self.exercise_type} (TensorFlow fallback)")

            self.scaler = joblib.load(str(scaler_path))
            self.label_encoder = joblib.load(str(encoder_path))

            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            print(f"Model loaded successfully: {self.exercise_type}")
            return True
        except MediaPipeUnavailableError as exc:
            print(str(exc))
            return False
        except ModelBackendUnavailableError as exc:
            print(str(exc))
            return False
        except Exception as exc:
            print(f"Error loading model: {exc}")
            return False

    def cleanup(self):
        """Release model and MediaPipe resources when switching exercises."""
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
        self.pose = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        gc.collect()

    def reset_counter(self):
        """Reset the exercise rep counter, if the exercise supports one."""
        if self.rep_counter is not None and hasattr(self.rep_counter, "reset"):
            self.rep_counter.reset()

    def get_rep_count(self):
        """Return the current rep count for health checks and UI state."""
        if self.rep_counter is not None and hasattr(self.rep_counter, "rep_count"):
            return int(self.rep_counter.rep_count)
        return 0

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points."""
        try:
            a = np.array(point1)
            b = np.array(point2)
            c = np.array(point3)

            ba = a - b
            bc = c - b

            denominator = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
            cosine_angle = np.dot(ba, bc) / denominator
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except Exception:
            return 0.0

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        try:
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        except Exception:
            return 0.0

    def extract_features(self, landmarks):
        """Extract features from landmarks. Implemented by each exercise subclass."""
        raise NotImplementedError("Subclass must implement extract_features method")

    def predict(self, landmarks):
        """Make prediction from landmarks."""
        try:
            features_dict = self.extract_features(landmarks)
            feature_array = np.array(list(features_dict.values())).reshape(1, -1)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

            features_scaled = self.scaler.transform(feature_array)
            prediction_proba = self.model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(prediction_proba)
            confidence = float(prediction_proba[0][predicted_class_idx])
            predicted_class = self.label_encoder.classes_[predicted_class_idx]

            return predicted_class, confidence
        except Exception as exc:
            print(f"Prediction error: {exc}")
            return "unknown", 0.0

    def process_frame(self, frame):
        """Process a single webcam/video frame and return analysis results."""
        try:
            self._ensure_mediapipe()
            if self.pose is None:
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "message": "Select an exercise before starting analysis.",
                }

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmark_spec, connection_spec = _get_drawing_specs()
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_spec,
                    connection_spec,
                )

                prediction, confidence = self.predict(results.pose_landmarks.landmark)

                rep_info = {}
                if self.rep_counter is not None:
                    rep_info = self.rep_counter.update(
                        results.pose_landmarks.landmark,
                        prediction,
                        confidence,
                    )

                try:
                    analysis_details = self.get_analysis_details(results.pose_landmarks.landmark)
                except Exception:
                    analysis_details = {}

                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                processed_frame = base64.b64encode(buffer).decode("utf-8")

                return {
                    "success": True,
                    "processed_frame": f"data:image/jpeg;base64,{processed_frame}",
                    "prediction": prediction,
                    "confidence": confidence,
                    "rep_info": rep_info,
                    "analysis_details": analysis_details,
                }

            return {
                "success": False,
                "error": "No pose detected",
                "message": "Position yourself so your full body is visible in frame.",
            }

        except MediaPipeUnavailableError as exc:
            return {"success": False, "error": "MediaPipe unavailable", "message": str(exc)}
        except Exception as exc:
            print(f"Error processing frame: {exc}")
            return {"success": False, "error": str(exc)}

    def process_image(self, frame):
        """Process a single image and return detailed analysis."""
        pose_static = None
        try:
            self._ensure_mediapipe()
            pose_static = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                smooth_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_static.process(rgb_frame)

            if results.pose_landmarks:
                landmark_spec, connection_spec = _get_drawing_specs()
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_spec,
                    connection_spec,
                )

                prediction, confidence = self.predict(results.pose_landmarks.landmark)
                analysis_details = self.get_analysis_details(results.pose_landmarks.landmark)

                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                processed_frame = base64.b64encode(buffer).decode("utf-8")

                return {
                    "success": True,
                    "processed_frame": f"data:image/jpeg;base64,{processed_frame}",
                    "prediction": prediction,
                    "confidence": confidence,
                    "analysis_details": analysis_details,
                }

            return {
                "success": False,
                "error": "No pose detected",
                "message": "Ensure your full body is visible in the image.",
            }

        except MediaPipeUnavailableError as exc:
            return {"success": False, "error": "MediaPipe unavailable", "message": str(exc)}
        except Exception as exc:
            print(f"Error processing image: {exc}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(exc)}
        finally:
            if pose_static is not None:
                pose_static.close()
            gc.collect()

    def get_analysis_details(self, landmarks):
        """Get detailed analysis. Implemented by each exercise subclass."""
        raise NotImplementedError("Subclass must implement get_analysis_details method")
