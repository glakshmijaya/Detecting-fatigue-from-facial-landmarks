# fatigue_dashboard.py
#main.py
"""
Streamlit dashboard for internship project:
Detecting Fatigue from Facial Landmarks — multi-page app.

Pages (sidebar - now reduced):
 - Ethical & Privacy Considerations
 - Real-time Webcam Detection (Continuous)
 - Driver Drowsy Alert Mechanism

Uses MediaPipe FaceMesh for landmark extraction and supports image upload + browser webcam (st.camera_input).
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import io
# import pygame
# import multiprocessing
from scipy.spatial import distance as dist
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import base64
from datetime import datetime 
import os
import joblib

# import threading


# --- Robust audio playback ---
try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    import pygame
except Exception:
    pygame = None


def safe_play_audio(path):
    """Play alarm sound using playsound or pygame fallback."""
    if not os.path.exists(path):
        print(f"⚠ Audio file missing: {path}")
        return

    def _threaded_play():
        try:
            if playsound:
                playsound(path, block=True)
            elif pygame:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
            else:
                print("❌ No supported audio library found (install playsound or pygame).")
        except Exception as e:
            print(f"Audio playback failed: {e}")

    threading.Thread(target=_threaded_play, daemon=True).start()


# Optional imports for CNN
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional audio
try:
    import simpleaudio as sa
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

st.set_page_config(page_title="Detecting Fatigue Dashboard", layout="wide")

# -----------------------------
# Utility: MediaPipe FaceMesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource
def get_face_mesh(max_num_faces=1):
    return mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces)

def get_landmarks_from_image(img_bgr):
    """Return list of landmarks (as mp Landmarks) or None"""
    face_mesh = get_face_mesh()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks  # iterable of landmarks

# -----------------------------
# Feature calculators
# -----------------------------
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
OUTER_MOUTH_IDX = list(range(61,68)) + list(range(291,298))
INNER_MOUTH_IDX = list(range(78,89)) + list(range(308,319))
POSE_IDX = [1, 199, 33, 263, 61, 291]

def eye_aspect_ratio_landmarks(landmarks, idxs, img_w, img_h):
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in idxs])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3]) + 1e-8
    return float((A + B) / (2.0 * C))

def mouth_aspect_ratio_landmarks(landmarks, outer_idx, inner_idx, img_w, img_h):
    pts_o = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in outer_idx])
    pts_i = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in inner_idx])
    # robust vertical measure
    try:
        vertical = (np.linalg.norm(pts_i[13] - pts_i[19]) +
                    np.linalg.norm(pts_i[14] - pts_i[18]) +
                    np.linalg.norm(pts_i[15] - pts_i[17])) / 3.0
    except Exception:
        vertical = np.linalg.norm(pts_o[2 % len(pts_o)] - pts_o[6 % len(pts_o)])
    width = np.linalg.norm(pts_o[0] - pts_o[6]) + 1e-8
    return float(vertical / width)

def estimate_head_pose(landmarks, img_shape):
    h, w = img_shape[:2]
    try:
        pts2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_IDX], dtype=np.float64)
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        focal_length = w
        center = (w/2, h/2)
        cam = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0,0,1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))
        ok, rvec, tvec = cv2.solvePnP(model_points, pts2d, cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0]*2 + rmat[1,0]*2)
        singular = sy < 1e-6
        if not singular:
            x = np.degrees(np.arctan2(rmat[2,1], rmat[2,2]))
            y = np.degrees(np.arctan2(-rmat[2,0], sy))
            z = np.degrees(np.arctan2(rmat[1,0], rmat[0,0]))
        else:
            x = np.degrees(np.arctan2(-rmat[1,2], rmat[1,1]))
            y = np.degrees(np.arctan2(-rmat[2,0], sy))
            z = 0
        return float(x), float(y), float(z)
    except Exception:
        return None, None, None

# -----------------------------
# Model loading helpers
# -----------------------------

# --- Define Model Paths (Using the user's absolute paths for the .pkl files) ---
BASE_DIR = os.path.dirname(__file__)

RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_pipeline.pkl")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")

EYE_BLINK_MODEL_PATH = os.path.join(BASE_DIR, "models", "eye_blink_model.pkl")
YAWN_MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.pkl")
# --- End Define Model Paths ---


@st.cache_resource
def load_rf_model(path=RF_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠ Failed to load RF model ({path}): {e}")
    else:
        st.warning(f"⚠ RF model not found at path: {path}")
    return None

@st.cache_resource
def load_cnn_model(path=CNN_MODEL_PATH):
    if TF_AVAILABLE and os.path.exists(path):
        try:
            return load_model(path)
        except Exception as e:
            st.warning(f"⚠ Failed to load CNN model ({path}): {e}")
    elif TF_AVAILABLE:
        st.warning(f"⚠ CNN model not found at path: {path}")
    return None

# --- Custom Model Loaders using Absolute Paths ---
@st.cache_resource
def load_eye_blink_model(path=EYE_BLINK_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠ Failed to load Eye Blink model ({path}): {e}")
    else:
        st.warning(f"⚠ Eye blink model not found at path: {path}")
    return None

@st.cache_resource
def load_yawn_model(path=YAWN_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠ Failed to load Yawn model ({path}): {e}")
    else:
        st.warning(f"⚠ Yawn model not found at path: {path}")
    return None
# -----------------------------
# End Model loading helpers
# -----------------------------

# -----------------------------
# Helpers: display images / plots
# -----------------------------
def image_to_bytes(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr)
    return buf.tobytes()

def plot_landmarks_scatter(landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    plt.figure(figsize=(4,5))
    plt.scatter(xs, [-y for y in ys], s=8)
    for i,(x,y) in enumerate(zip(xs,ys)):
        plt.text(x, -y, str(i), fontsize=6)
    plt.title("Facial Landmarks (indexed)")
    plt.axis('off'); st.pyplot(plt.gcf()); plt.clf()

# -----------------------------
# Page logic
# -----------------------------
st.sidebar.title("Navigation")
# Reduced pages list
pages = [
    #"Ethical & Privacy Considerations",    # Page 13
    "Real-time Webcam Detection (Continuous)", # Page 14 (Continuous Webcam)
    "Driver Drowsy Alert Mechanism",        # Page 15
    #"Driver monitoring system"              #page 16
]
choice = st.sidebar.radio("Go to", pages)

# common models
rf_model = load_rf_model()
cnn_model = load_cnn_model()
eye_blink_model = load_eye_blink_model()
yawn_model = load_yawn_model()


# ------------------- PAGE 13: Ethical & Privacy Considerations -------------------
# if choice == "Ethical & Privacy Considerations":
#     st.title("🧠 Ethical & Privacy Considerations")

#     st.markdown("""
#     This page demonstrates responsible AI principles applied to fatigue detection.
#     Each area below shows how user consent, privacy, and on-device processing are handled ethically.
#     """)

#     st.subheader("📜 Area 1: Consent-Based Recording")
#     consent = st.checkbox("I consent to enable video capture and analysis.", key="consent_check")
#     if consent:
#         st.success("✅ Consent granted. Camera functionality is enabled on the 'Real-time Detection' page.")
#         st.info("No video is saved or transmitted by this Streamlit app.")
#     else:
#         st.warning("⚠ Camera functionality remains restricted without consent.")

#     st.markdown("---")
#     st.subheader("🕶 Area 2: Data Anonymization Principles")
#     st.markdown("""
#     - *No Face Images Stored:* Raw video frames are processed for landmark features (EAR, MAR, Head Pose) only.
#     - *Feature Anonymity:* Fatigue scores are based on abstract ratios, not identifiable facial data.
#     - *Local Processing:* All detection and prediction runs locally (on-device) where possible, minimizing data transmission.
#     """)
#     st.code("""
#     # Pseudo-code for anonymization:
#     landmarks = get_landmarks(frame)  # Only coordinates used
#     EAR, MAR = calculate_features(landmarks)
#     store_data(timestamp, EAR, MAR, score) # Only store features/scores, not the frame
#     """)

#     st.markdown("---")
#     st.subheader("🔒 Area 3: Transparency and Control")
#     st.markdown("""
#     Users should always know:
#     1. *When* the system is running (e.g., clear indicator light).
#     2. *What* data is being analyzed (facial landmarks).
#     3. *How* the fatigue score is calculated (Model Training page).
#     4. *How to turn it OFF* (Stop button on Real-time Detection page).
#     """)

# ------------------- PAGE 14: Real-time Webcam Detection (Continuous) -------------------
if choice == "Real-time Webcam Detection (Continuous)":
    st.title("📹 Real-time Webcam Detection (Continuous)")
    st.markdown(
        "This demo runs a live webcam feed, processing each frame to detect fatigue using EAR (eye), MAR (mouth), and head tilt analysis."
    )
    st.info("Ensure eye_blink_model.pkl, yawn_model.pkl, and alarm.wav are correctly located in your working directory.")

    import threading, os, time, cv2, numpy as np
    from playsound import playsound

    # --- Alarm setup ---
    ALARM_ACTIVE_FLAG = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")
    EVENTS_FILE = os.path.join(BASE_DIR, "drowsy_events.csv")

    def play_alarm_sound(path):
        """Play alarm sound safely in a separate thread."""
        global ALARM_ACTIVE_FLAG
        if ALARM_ACTIVE_FLAG:
            return
        ALARM_ACTIVE_FLAG = True
        try:
            safe_path = os.path.normpath(path)
            playsound(safe_path)
        except Exception as e:
            print("Audio error:", e)
        ALARM_ACTIVE_FLAG = False

    def play_alarm_non_blocking(path):
        """Play alarm only if not already active."""
        global ALARM_ACTIVE_FLAG
        if not ALARM_ACTIVE_FLAG:
            threading.Thread(target=play_alarm_sound, args=(path,), daemon=True).start()

    # --- Event logging for dashboard ---
    def log_event(event_type, extra=None):
        import pandas as pd
        now = pd.Timestamp.now().isoformat()
        entry = {"timestamp": now, "event": event_type}
        if extra:
            entry.update(extra)
        df = pd.DataFrame([entry])
        file_exists = os.path.exists(EVENTS_FILE)
        df.to_csv(EVENTS_FILE, mode='a', header=not file_exists, index=False)
    #userbehaviour trends dashboard
    import pandas as pd
    if os.path.exists(EVENTS_FILE):
        st.subheader("User Behaviour Trends (Past Days/Months)")
        df_evt = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"], on_bad_lines="skip")
        df_evt['day'] = pd.to_datetime(df_evt["timestamp"]).dt.date
        daily = df_evt.groupby(['day', 'event']).size().unstack(fill_value=0)
        st.bar_chart(daily)


    # --- Streamlit control buttons ---
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam", key="start_rt_webcam"):
            st.session_state.run_webcam = True
    with col2:
        if st.button("Stop Webcam", key="stop_rt_webcam"):
            st.session_state.run_webcam = False

    # --- Webcam processing loop ---
    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
            st.session_state.run_webcam = False
        else:
            frame_placeholder = st.empty()
            FRAME_RATE = 30

            consecutive_eye_closed = 0
            consecutive_yawn = 0
            consecutive_tilt = 0

            try:
                while st.session_state.run_webcam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠ Frame capture failed.")
                        break

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    color_default = (255, 255, 255)
                    color_status = (0, 255, 0)
                    fatigue_pct_eye = 0
                    fatigue_pct_yawn = 0
                    status_text_eye = ""
                    status_text_yawn = ""

                    lms_all = get_landmarks_from_image(frame)
                    if lms_all:
                        lm = lms_all[0].landmark
                        ear = (
                            eye_aspect_ratio_landmarks(lm, LEFT_EYE_IDX, w, h) + 
                            eye_aspect_ratio_landmarks(lm, RIGHT_EYE_IDX, w, h)
                        ) / 2.0
                        mar = mouth_aspect_ratio_landmarks(lm, OUTER_MOUTH_IDX, INNER_MOUTH_IDX, w, h)
                        pitch, yaw, roll = estimate_head_pose(lm, frame.shape)

                        # --- Predictions ---
                        eye_pred = int(eye_blink_model.predict(np.array([ear]).reshape(1, -1))[0])
                        mar_pred = int(yawn_model.predict(np.array([mar]).reshape(1, -1))[0])
                        ear_label = "OPEN" if eye_pred == 1 else "CLOSED"
                        yawn_label = "YAWN" if mar_pred == 1 else "NO YAWN"

                        # --- Log events for dashboard ---
                        log_event("EYE_" + ear_label)  # log EYE_OPEN or EYE_CLOSED each frame

                        head_tilt = pitch is not None and pitch > 20
                        eye_closed = ear_label == "CLOSED"

                        if eye_closed:
                            consecutive_eye_closed += 1
                        else:
                            consecutive_eye_closed = 0

                        yawn_detected = (mar_pred == 1 and mar > 0.35)
                        if yawn_detected:
                            consecutive_yawn += 1
                            if consecutive_yawn / FRAME_RATE >= 3:
                                log_event("YAWN_ALARM")
                            else:
                                log_event("YAWN_WARNING")
                        else:
                            consecutive_yawn = 0

                        if head_tilt:
                            consecutive_tilt += 1
                        else:
                            consecutive_tilt = 0

                        eye_closed_sec = consecutive_eye_closed / FRAME_RATE
                        yawn_sec = consecutive_yawn / FRAME_RATE
                        tilt_sec = consecutive_tilt / FRAME_RATE

                        # --- Eye-blink fatigue logic ---
                        if eye_closed_sec >= 3:
                            fatigue_pct_eye = 100
                            status_text_eye = "🚨 FATIGUE DETECTED – TAKE A BREAK!"
                            color_status_eye = (0, 0, 255)
                            play_alarm_non_blocking(ALARM_PATH)
                        elif 0 < eye_closed_sec < 3:
                            fatigue_pct_eye = 50
                            status_text_eye = "STATUS: FATIGUE WARNING!!"
                            color_status_eye = (0, 165, 255)
                        else:
                            fatigue_pct_eye = 0
                            status_text_eye = "STATUS: NORMAL (ATTENTIVE)"
                            color_status_eye = (0, 255, 0)

                        # --- Yawn logic ---
                        if yawn_detected and yawn_sec >= 3:
                            fatigue_pct_yawn = 100
                            status_text_yawn = "⚠ YAWN ALERT — TAKE A BREAK!"
                            color_status_yawn = (0, 0, 255)
                            play_alarm_non_blocking(ALARM_PATH)
                        elif yawn_detected and 0 < yawn_sec < 3:
                            fatigue_pct_yawn = 50
                            status_text_yawn = "YAWN DETECTED! WARNING"
                            color_status_yawn = (0, 165, 255)
                        else:
                            fatigue_pct_yawn = 0
                            status_text_yawn = "NO YAWN"
                            color_status_yawn = (0, 255, 0)

                        # --- Display timers and overlays ---
                        cv2.putText(
                            frame,
                            f"Eye closed timer: {eye_closed_sec:.1f}s",
                            (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Yawn timer: {yawn_sec:.1f}s",
                            (10, 230),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )

                        # --- Fatigue overlays ---
                        cv2.putText(
                            frame,
                            f"STATUS EYEBLINK: {ear_label}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"EAR: {ear:.3f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"MAR: {mar:.3f} ({yawn_label})",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Fatigue (Eye): {fatigue_pct_eye}%",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255) if fatigue_pct_eye >= 60 else (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Fatigue (Yawn): {fatigue_pct_yawn}%",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255) if fatigue_pct_yawn >= 60 else (0, 255, 0),
                            2,
                        )

                        # --- Status overlays ---
                        cv2.putText(
                            frame,
                            status_text_eye,
                            (10, 270),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            color_status_eye,
                            2,
                        )
                        cv2.putText(
                            frame,
                            status_text_yawn,
                            (10, 300),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            color_status_yawn,
                            2,
                        )

                        # --- Red Overlay Watermark for 100% Fatigue ---
                        if fatigue_pct_eye == 100 or fatigue_pct_yawn == 100:
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 255), -1)
                            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                            cv2.putText(
                                frame,
                                "🚨 FATIGUE ALERT 🚨",
                                (w // 2 - 200, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                3,
                            )

                    # --- Stream frame to Streamlit ---
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    time.sleep(0.03)

            finally:
                cap.release()
                st.session_state.run_webcam = False
                st.info("✅ Webcam stopped.")



         
            
            model = joblib.load("models/eye_blink_model.pkl")
            test_ears = np.array([[0.15], [0.25], [0.35]])
            preds = model.predict(test_ears)
            for ear, pred in zip(test_ears.flatten(), preds):
                print(f"EAR={ear:.2f} -> Model Prediction={pred}")


# ------------------- PAGE 15: Driver Drowsy Alert Mechanism -------------------
elif choice == "Driver Drowsy Alert Mechanism":
    import streamlit as st
    import cv2, joblib, numpy as np, mediapipe as mp, os, time, threading
    import pandas as pd
    from datetime import datetime
    from playsound import playsound

    st.title("🚨 Driver Drowsy Alert Mechanism")
    st.markdown("""
    EAR & MAR based fatigue detection  
    - Real-time EAR / MAR metrics  
    - Head-pose readiness  
    - 3s sustained eye-closure / yawn alarm trigger  
    - Continuous visual warnings  
    """)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    ALARM_FILE = os.path.join(BASE_DIR, "alarm.wav")
    EVENTS_FILE = os.path.join(BASE_DIR, "drowsy_events.csv")

    EYE_BLINK_MODEL_PATH = os.path.join(MODELS_DIR, "eye_blink_model.pkl")
    YAWN_MODEL_PATH = os.path.join(MODELS_DIR, "yawn_model.pkl")
    ALARM_SOUND_PATH = ALARM_FILE

    for name, path in {
        "Eye Blink Model": EYE_BLINK_MODEL_PATH,
        "Yawn Model": YAWN_MODEL_PATH,
        "Alarm Sound": ALARM_SOUND_PATH,
    }.items():
        if not os.path.exists(path):
            st.warning(f"⚠ {name} not found: {path}")
        else:
            st.success(f"✅ {name} loaded: {os.path.basename(path)}")

    # --- Detection constants ---
    ALARM_COOLDOWN = 5.0
    EYE_OPEN_RECOVERY_FRAMES = 5
    CRITICAL_EYE_CLOSE_FRAMES = 6
    DANGER_DELAY = 3.0  # seconds of continuous closure before alarm
    YAWN_DELAY = 3.0    # seconds of continuous yawn before alarm

    # --- NEW thresholds ---
    YAWN_THRESHOLD = 0.7   # Example, tune as needed
    EYE_CLOSED_THRESHOLD = 0.21  # Tune as needed
    BRIGHTNESS_MIN = 50  # Minimum average gray value for reliable detection
    YAWN_FRAMES_REQUIRED = int(0.7 * 30) # ~0.7s at 30fps

    global DANGER_ACTIVE
    DANGER_ACTIVE = False
    ALARM_ACTIVE_FLAG = False

    # --- Helper functions ---
    def play_sound_thread(path):
        global ALARM_ACTIVE_FLAG
        ALARM_ACTIVE_FLAG = True
        try:
            playsound(path)
        except Exception as e:
            print("Audio error:", e)
        ALARM_ACTIVE_FLAG = False

    def play_alarm_non_blocking(path, last_time, cooldown):
        global ALARM_ACTIVE_FLAG
        t = time.time()
        if (t - last_time) > cooldown and not ALARM_ACTIVE_FLAG:
            if os.path.exists(path):
                threading.Thread(target=play_sound_thread, args=(path,), daemon=True).start()
                return t
        return last_time

    def log_event(event_type, extra=None):
        now = datetime.now().isoformat()
        entry = {"timestamp": now, "event": event_type}
        if extra: entry.update(extra)
        df = pd.DataFrame([entry])
        file_exists = os.path.exists(EVENTS_FILE)
        df.to_csv(EVENTS_FILE, mode='a', header=not file_exists, index=False)

    st.info("Click below to start detection.")

    # --- Dashboard: Show trends (after detection logic) --- 
    EVENTS_FILE = os.path.join(BASE_DIR, "drowsy_events.csv")
    if os.path.exists(EVENTS_FILE):
        st.subheader("User Behaviour Trends (Past Days/Months)")
        # add on_bad_lines="skip" to ignore malformed lines
        df_evt = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"], on_bad_lines="skip")
        df_evt['day'] = pd.to_datetime(df_evt["timestamp"]).dt.date
        daily = df_evt.groupby(['day', 'event']).size().unstack(fill_value=0)
        st.bar_chart(daily)

    start_btn = st.button("▶ Start Drowsy Detection")
    stop_btn = st.button("⏹ Stop Detection")

    if "running" not in st.session_state:
        st.session_state.running = False

    if start_btn:
        st.session_state.running = True
        st.success("✅ Detection started.")
    if stop_btn:
        st.session_state.running = False
        st.warning("🛑 Detection stopped.")

    FRAME_WINDOW = st.image([])

    if st.session_state.running:
        eye_model = joblib.load(EYE_BLINK_MODEL_PATH)
        yawn_model = joblib.load(YAWN_MODEL_PATH)
        mp_face = mp.solutions.face_mesh
        cap = cv2.VideoCapture(0)

        last_audio_alert_time = time.time() - ALARM_COOLDOWN
        consecutive_closed, open_recovery = 0, 0
        consecutive_yawn_frames = 0
        danger_start_time, yawn_start_time = None, None

        with mp_face.FaceMesh(
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        ) as face_mesh:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Camera not detected.")
                    break

                # --- Low light detection
                avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if avg_brightness < BRIGHTNESS_MIN:
                    st.warning("⚠ Low-light detected. Detection accuracy may be reduced.")

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark

                    def ratio(idxs):
                        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs])
                        A = np.linalg.norm(pts[1]-pts[5])
                        B = np.linalg.norm(pts[2]-pts[4])
                        C = np.linalg.norm(pts[0]-pts[3]) + 1e-8
                        return (A+B)/(2*C)

                    LEFT = [362,385,387,263,373,380]
                    RIGHT = [33,160,158,133,153,144]
                    MOUTH = [78,308,13,14,87,317]
                    ear = np.mean([ratio(LEFT), ratio(RIGHT)])
                    mar = ratio(MOUTH)

                    eye_state = int(eye_model.predict([[ear]])[0])
                    yawn_state = int(yawn_model.predict([[mar]])[0])

                    # --- Eye closure logic with timer --
                    if eye_state == 0 and ear < EYE_CLOSED_THRESHOLD:
                        consecutive_closed += 1
                        open_recovery = 0
                        if danger_start_time is None:
                            danger_start_time = time.time()
                        elapsed_danger = time.time() - danger_start_time
                    else:
                        consecutive_closed = 0
                        open_recovery += 1
                        danger_start_time = None
                        elapsed_danger = 0

                    # --- Yawn logic with robust frame window ---
                    if (yawn_state == 1) and (mar > YAWN_THRESHOLD):
                        consecutive_yawn_frames += 1
                        if yawn_start_time is None:
                            yawn_start_time = time.time()
                        elapsed_yawn = time.time() - yawn_start_time
                    else:
                        consecutive_yawn_frames = 0
                        yawn_start_time = None
                        elapsed_yawn = 0

                    yawn_detected = consecutive_yawn_frames >= YAWN_FRAMES_REQUIRED

                    # --- Status logic ---
                    is_danger = (elapsed_danger >= DANGER_DELAY)
                    status_text = ""
                    status_color = (0,255,0) # default green

                    if is_danger:
                        status_text = "STATUS: DANGER: IMMEDIATE ALERT"
                        status_color = (0, 0, 255)
                        cv2.putText(frame, "WAKE UP! PULL OVER!", (80, int(h / 2)),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 4)
                        log_event("EYE_CLOSED_ALARM")
                    elif consecutive_closed > 0 and consecutive_closed < CRITICAL_EYE_CLOSE_FRAMES:
                        status_text = "STATUS: STRAINING (LOW BLINKS)"
                        status_color = (0, 165, 255)
                    else:
                        status_text = "STATUS: NORMAL (ATTENTIVE)"
                        status_color = (0, 255, 0)

                    # --- Overlay Yawn Detection -- robust (avoid false positives)
                    cv2.putText(frame, f"YAWN: {'YES' if yawn_detected else 'NO'}", (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if yawn_detected else (0, 255, 0), 2)

                    if yawn_detected and elapsed_yawn >= YAWN_DELAY:
                        log_event("YAWN_ALARM")
                    elif yawn_detected:
                        log_event("YAWN_WARNING")

                    # --- Timer Overlays --
                    if elapsed_danger > 0:
                        cv2.putText(frame, f"Eyes closed: {elapsed_danger:.1f}s", (10, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if is_danger else (0, 165, 255), 2)

                    if elapsed_yawn > 0:
                        cv2.putText(frame, f"Yawn detected: {elapsed_yawn:.1f}s", (10, 220),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if elapsed_yawn >= YAWN_DELAY else (0,255,0), 2)

                    # --- Alarm condition: sustained events ---
                    if is_danger or (yawn_detected and elapsed_yawn >= YAWN_DELAY):
                        last_audio_alert_time = play_alarm_non_blocking(
                            ALARM_SOUND_PATH, last_audio_alert_time, ALARM_COOLDOWN
                        )
                    elif (consecutive_closed > 0 and consecutive_closed < CRITICAL_EYE_CLOSE_FRAMES) or yawn_detected:
                        cv2.putText(frame, "⚠ WARNING: POSSIBLE FATIGUE", (50, int(h * 0.90)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

                    # --- Overlay Info ---
                    cv2.putText(frame, status_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (200, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        st.success("🟢 Detection stopped successfully.")


# elif choice == "Driver Drowsy Alert Mechanism":
#     import streamlit as st
#     import cv2, joblib, numpy as np, mediapipe as mp, os, time, threading, math
#     from playsound import playsound

#     st.title("🚨 Driver Drowsy Alert Mechanism")
#     st.markdown("""
#     This page runs the improved real-time drowsiness detection system.  
#     Features:
#     - EAR & MAR based drowsiness metrics  
#     - Real-time visual alert frames  
#     - Head-pose (nod) readiness  
#     - Hysteresis alarm logic (no repeat sounds)  
#     - Uses playsound for stable sound triggering  
#     """)

#     # --- Auto-detect base path ---
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     MODELS_DIR = os.path.join(BASE_DIR, "models")
#     ALARM_FILE = os.path.join(BASE_DIR, "alarm.wav")

#     # --- Model paths ---
#     EYE_BLINK_MODEL_PATH = os.path.join(MODELS_DIR, "eye_blink_model.pkl")
#     YAWN_MODEL_PATH = os.path.join(MODELS_DIR, "yawn_model.pkl")
#     ALARM_SOUND_PATH = ALARM_FILE

#     # --- Verify paths exist ---
#     for name, path in {
#         "Eye Blink Model": EYE_BLINK_MODEL_PATH,
#         "Yawn Model": YAWN_MODEL_PATH,
#         "Alarm Sound": ALARM_SOUND_PATH,
#     }.items():
#         if not os.path.exists(path):
#             st.warning(f"⚠ {name} not found: {path}")
#         else:
#             st.success(f"✅ {name} loaded: {os.path.basename(path)}")

#     # --- Detection thresholds ---
#     ALARM_COOLDOWN = 3.0
#     EYE_OPEN_RECOVERY_FRAMES = 5
#     HEAD_DOWN_THRESHOLD = 20.0
#     CRITICAL_EYE_CLOSE_FRAMES = 6
#     FRAME_CLOSE_THRESHOLD = 3
#     CONSECUTIVE_NOD_FRAMES = 3

#     # --- State flags ---
#     #global DANGER_ACTIVE, ALARM_ACTIVE_FLAG
#     DANGER_ACTIVE = False
#     ALARM_ACTIVE_FLAG = False

#     # --- Helper functions ---
#     def play_sound_thread(path):
#         global ALARM_ACTIVE_FLAG
#         ALARM_ACTIVE_FLAG = True
#         try:
#             playsound(path)
#         except Exception as e:
#             print("Audio error:", e)
#         ALARM_ACTIVE_FLAG = False

#     def play_alarm_non_blocking(path, last_time, cooldown):
#         global ALARM_ACTIVE_FLAG
#         t = time.time()
#         if (t - last_time) > cooldown and not ALARM_ACTIVE_FLAG:
#             if os.path.exists(path):
#                 threading.Thread(target=play_sound_thread, args=(path,), daemon=True).start()
#                 return t
#         return last_time

#     st.info("Click below to start the improved real-time detection (camera feed will appear here).")

#     # --- Button controls ---
#     start_btn = st.button("▶ Start Drowsy Detection")
#     stop_btn = st.button("⏹ Stop Detection")

#     if "running" not in st.session_state:
#         st.session_state.running = False

#     if start_btn:
#         st.session_state.running = True
#         st.success("✅ Drowsy detection started.")
#     if stop_btn:
#         st.session_state.running = False
#         st.warning("🛑 Detection stopped by user.")

#     # --- Initialize display area for webcam feed ---
#     FRAME_WINDOW = st.image([])

#     if st.session_state.running:
#         eye_model = joblib.load(EYE_BLINK_MODEL_PATH)
#         yawn_model = joblib.load(YAWN_MODEL_PATH)
#         mp_face = mp.solutions.face_mesh
#         cap = cv2.VideoCapture(0)

#         last_audio_alert_time = time.time() - ALARM_COOLDOWN
#         consecutive_closed, open_recovery = 0, 0

#         with mp_face.FaceMesh(
#             refine_landmarks=True, max_num_faces=1,
#             min_detection_confidence=0.6, min_tracking_confidence=0.6
#         ) as face_mesh:
#             while st.session_state.running:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("❌ Camera not detected.")
#                     break

#                 frame = cv2.flip(frame, 1)
#                 h, w = frame.shape[:2]
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_mesh.process(rgb)

#                 if results.multi_face_landmarks:
#                     lm = results.multi_face_landmarks[0].landmark

#                     def ratio(idxs):
#                         pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs])
#                         A = np.linalg.norm(pts[1]-pts[5])
#                         B = np.linalg.norm(pts[2]-pts[4])
#                         C = np.linalg.norm(pts[0]-pts[3])+1e-8
#                         return (A+B)/(2*C)

#                     LEFT = [362,385,387,263,373,380]
#                     RIGHT = [33,160,158,133,153,144]
#                     MOUTH = [78,308,13,14,87,317]
#                     ear = np.mean([ratio(LEFT), ratio(RIGHT)])
#                     mar = ratio(MOUTH)

#                     # Predictions
#                     eye_state = int(eye_model.predict([[ear]])[0])
#                     yawn_state = int(yawn_model.predict([[mar]])[0])

#                     if eye_state == 0:
#                         consecutive_closed += 1
#                         open_recovery = 0
#                     else:
#                         consecutive_closed = 0
#                         open_recovery += 1

#                     is_danger = (consecutive_closed >= CRITICAL_EYE_CLOSE_FRAMES)
#                     if is_danger:
#                         if not DANGER_ACTIVE:
#                             DANGER_ACTIVE = True
#                             last_audio_alert_time = play_alarm_non_blocking(
#                                 ALARM_SOUND_PATH, last_audio_alert_time, ALARM_COOLDOWN
#                             )
#                     else:
#                         if DANGER_ACTIVE and open_recovery >= max(EYE_OPEN_RECOVERY_FRAMES*5, 15):
#                             DANGER_ACTIVE = False

#                     # ---------------------- VISUAL ALERTS ----------------------
#                     status_text = ""
#                     status_color = (0, 255, 0)  # Green

#                     if is_danger:
#                         status_text = "STATUS: DANGER: IMMEDIATE ALERT"
#                         status_color = (0, 0, 255)  # Red
#                         cv2.putText(frame, "WAKE UP! PULL OVER!", (80, int(h / 2)),
#                                     cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 4)
#                     elif consecutive_closed > 0 and consecutive_closed < CRITICAL_EYE_CLOSE_FRAMES:
#                         status_text = "STATUS: STRAINING (LOW BLINKS)"
#                         status_color = (0, 165, 255)  # Orange
#                     else:
#                         status_text = "STATUS: NORMAL (ATTENTIVE)"
#                         status_color = (0, 255, 0)

#                     cv2.putText(frame, status_text, (10, 60),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

#                     # EAR & MAR display
#                     cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#                     cv2.putText(frame, f"MAR: {mar:.2f}", (200, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#                     # Placeholder metrics for consistency with your example images
#                     cv2.putText(frame, "YAWS/60s: 0", (10, 140),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#                     cv2.putText(frame, "BLINKS/60s: 0", (10, 165),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#                     cv2.putText(frame, "PITCH: 177.3 deg (Nodding: False)", (10, 190),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#                     # ------------------------------------------------------------

#                 # Convert frame for Streamlit
#                 FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         cap.release()
#         st.success("🟢 Detection stopped successfully.")

# elif choice == "Driver Drowsy Alert Mechanism":
#     import streamlit as st
#     import cv2, joblib, numpy as np, mediapipe as mp, os, time, threading, subprocess, platform, time

#     st.title("🚨 Driver Drowsy Alert Mechanism")
#     st.markdown("""
#     This page runs the improved real-time drowsiness detection system.  
#     Features:
#     - EAR & MAR based drowsiness metrics  
#     - Real-time visual alert frames  
#     - Hysteresis alarm logic  
#     - Stable non-blocking alarm sound  
#     """)

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     MODELS_DIR = os.path.join(BASE_DIR, "models")
#     ALARM_FILE = os.path.join(BASE_DIR, "alarm.wav")

#     EYE_BLINK_MODEL_PATH = os.path.join(MODELS_DIR, "eye_blink_model.pkl")
#     YAWN_MODEL_PATH = os.path.join(MODELS_DIR, "yawn_model.pkl")

#     # --- Path verification ---
#     for name, path in {
#         "Eye Blink Model": EYE_BLINK_MODEL_PATH,
#         "Yawn Model": YAWN_MODEL_PATH,
#         "Alarm Sound": ALARM_FILE,
#     }.items():
#         if not os.path.exists(path):
#             st.warning(f"⚠ {name} not found: {path}")
#         else:
#             st.success(f"✅ {name} loaded: {os.path.basename(path)}")

#     # --- Thresholds ---
#     ALARM_COOLDOWN = 5.0
#     CRITICAL_EYE_CLOSE_SEC = 3.0
#     CRITICAL_YAWN_SEC = 3.0
#     FRAME_RATE = 30

#     #global DANGER_ACTIVE
#     DANGER_ACTIVE = False
#     ALARM_ACTIVE_FLAG = False

#     # --- Reliable alarm using subprocess ---
#     def play_alarm(path):
#         global ALARM_ACTIVE_FLAG
#         if not os.path.exists(path):
#             return
#         ALARM_ACTIVE_FLAG = True
#         try:
#             if platform.system() == "Windows":
#                 subprocess.Popen(["start", "", path], shell=True)
#             elif platform.system() == "Darwin":
#                 subprocess.Popen(["afplay", path])
#             else:
#                 subprocess.Popen(["xdg-open", path])
#         except Exception as e:
#             print("Audio playback error:", e)
#         time.sleep(1.5)
#         ALARM_ACTIVE_FLAG = False

#     def play_alarm_non_blocking(path, last_time, cooldown):
#         global ALARM_ACTIVE_FLAG
#         t = time.time()
#         if (t - last_time) > cooldown and not ALARM_ACTIVE_FLAG:
#             threading.Thread(target=play_alarm, args=(path,), daemon=True).start()
#             return t
#         return last_time

#     st.info("Click below to start the improved real-time detection (camera feed will appear here).")

#     # --- Control buttons ---
#     start_btn = st.button("▶ Start Drowsy Detection")
#     stop_btn = st.button("⏹ Stop Detection")

#     if "running" not in st.session_state:
#         st.session_state.running = False

#     if start_btn:
#         st.session_state.running = True
#         st.success("✅ Drowsy detection started.")
#     if stop_btn:
#         st.session_state.running = False
#         st.warning("🛑 Detection stopped by user.")

#     FRAME_WINDOW = st.image([])

#     if st.session_state.running:
#         eye_model = joblib.load(EYE_BLINK_MODEL_PATH)
#         yawn_model = joblib.load(YAWN_MODEL_PATH)
#         mp_face = mp.solutions.face_mesh
#         cap = cv2.VideoCapture(0)

#         last_audio_alert_time = time.time() - ALARM_COOLDOWN
#         consecutive_closed = 0
#         consecutive_yawn = 0
#         open_recovery = 0

#         with mp_face.FaceMesh(
#             refine_landmarks=True, max_num_faces=1,
#             min_detection_confidence=0.6, min_tracking_confidence=0.6
#         ) as face_mesh:
#             while st.session_state.running:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("❌ Camera not detected.")
#                     break

#                 frame = cv2.flip(frame, 1)
#                 h, w = frame.shape[:2]
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_mesh.process(rgb)

#                 if results.multi_face_landmarks:
#                     lm = results.multi_face_landmarks[0].landmark

#                     def ratio(idxs):
#                         pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs])
#                         A = np.linalg.norm(pts[1]-pts[5])
#                         B = np.linalg.norm(pts[2]-pts[4])
#                         C = np.linalg.norm(pts[0]-pts[3])+1e-8
#                         return (A+B)/(2*C)

#                     LEFT = [362,385,387,263,373,380]
#                     RIGHT = [33,160,158,133,153,144]
#                     MOUTH = [78,308,13,14,87,317]
#                     ear = np.mean([ratio(LEFT), ratio(RIGHT)])
#                     mar = ratio(MOUTH)

#                     eye_state = int(eye_model.predict([[ear]])[0])
#                     yawn_state = int(yawn_model.predict([[mar]])[0])

#                     # --- Eye Closure / Yawn ---
#                     if eye_state == 0:
#                         consecutive_closed += 1
#                         open_recovery = 0
#                     else:
#                         consecutive_closed = 0
#                         open_recovery += 1

#                     if yawn_state == 1:
#                         consecutive_yawn += 1
#                     else:
#                         consecutive_yawn = 0

#                     closed_sec = consecutive_closed / FRAME_RATE
#                     yawn_sec = consecutive_yawn / FRAME_RATE

#                     # --- Danger detection ---
#                     is_eye_danger = closed_sec >= CRITICAL_EYE_CLOSE_SEC
#                     is_yawn_danger = yawn_sec >= CRITICAL_YAWN_SEC
#                     is_danger = is_eye_danger or is_yawn_danger

#                     if is_danger:
#                         if not DANGER_ACTIVE:
#                             DANGER_ACTIVE = True
#                             last_audio_alert_time = play_alarm_non_blocking(
#                                 ALARM_FILE, last_audio_alert_time, ALARM_COOLDOWN
#                             )
#                     else:
#                         if DANGER_ACTIVE and open_recovery > 5:
#                             DANGER_ACTIVE = False

#                     # --- STATUS TEXT LOGIC ---
#                     if is_danger:
#                         status_text = "⚠ DANGER: STAY ALERT!"
#                         status_color = (0, 0, 255)
#                         cv2.putText(frame, "WAKE UP! PULL OVER!", (80, int(h / 2)),
#                                     cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
#                     elif consecutive_closed > 0 or yawn_state == 1:
#                         status_text = "STATUS: STRAINING (LOW BLINKS)"
#                         status_color = (0, 165, 255)
#                     else:
#                         status_text = "STATUS: NORMAL (ATTENTIVE)"
#                         status_color = (0, 255, 0)

#                     # --- Display inside frame ---
#                     cv2.putText(frame, status_text, (10, 60),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

#                     # EAR & MAR display
#                     cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
#                     cv2.putText(frame, f"MAR: {mar:.2f}", (200, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#                     cv2.putText(frame, f"Eye Closed: {closed_sec:.1f}s", (10, 140),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#                     cv2.putText(frame, f"Yawn: {yawn_sec:.1f}s", (10, 165),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

#                 FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         cap.release()
#         st.success("🟢 Detection stopped successfully.")




# ------------------- END OF FILE -------------------
