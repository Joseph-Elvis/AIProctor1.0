import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import urllib.request
import os
import requests
import threading
from datetime import datetime
import base64

print("DEBUG: Imports successful")

# ── Load YOLOv8 ───────────────────────────────────────────────────────────────
print("DEBUG: Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("DEBUG: YOLOv8 loaded successfully")

# ── Load MediaPipe Face Landmarker (new API) ──────────────────────────────────
print("DEBUG: Loading MediaPipe Face Mesh...")

MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("DEBUG: Downloading face landmarker model (~30MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("DEBUG: Model downloaded successfully")

BaseOptions          = mp.tasks.BaseOptions
FaceLandmarker       = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = FaceLandmarker.create_from_options(options)
print("DEBUG: MediaPipe loaded successfully")

# ── 3D Head Model Points ──────────────────────────────────────────────────────
FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,  -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# ── Suspicious Object Classes ─────────────────────────────────────────────────
SUSPICIOUS_CLASSES = {
    67: "cell phone",
    73: "book",
    63: "laptop",
    76: "scissors",
}
PERSON_CLASS_ID = 0

# ── Behaviour Thresholds ──────────────────────────────────────────────────────
YAW_THRESHOLD              = 20
NO_FACE_FRAMES_REQUIRED    = 30
SUSPICIOUS_FRAMES_REQUIRED = 30

# ── Gaze / Composite Deviation Scoring ────────────────────────────────────────
# The project spec calls for a composite score combining head-yaw magnitude
# with gaze eccentricity (iris position within the eye), not yaw alone.
# MediaPipe's face_landmarker.task model outputs 478 landmarks by default;
# 468 and 473 are the right/left iris centers used here.
RIGHT_IRIS_IDX   = 468
LEFT_IRIS_IDX    = 473
RIGHT_EYE_CORNERS = (33, 133)    # (outer, inner)
LEFT_EYE_CORNERS  = (362, 263)   # (inner, outer)

YAW_NORM_MAX         = 45.0   # yaw magnitude treated as "full" deviation
GAZE_NORM_MAX        = 0.5    # gaze eccentricity treated as "full" deviation
YAW_WEIGHT           = 0.6
GAZE_WEIGHT          = 0.4
DEVIATION_THRESHOLD  = 0.45   # composite score above this = suspicious

# ── Persistence Counters ──────────────────────────────────────────────────────
frame_counters = {
    "LOOKING LEFT":  0,
    "LOOKING RIGHT": 0,
    "NO FACE":       0,
}

# ── Alert Cooldown ────────────────────────────────────────────────────────────
last_alert_time        = {}
ALERT_COOLDOWN_SECONDS = 5

# ── API Configuration ─────────────────────────────────────────────────────────
API_URL      = "http://127.0.0.1:8000/alerts"
SESSION_ID   = "SESSION_001"
CANDIDATE_ID = "CANDIDATE_001"
STATION_ID   = "STATION_001"


# ═════════════════════════════════════════════════════════════════════════════
def get_head_pose(landmarks, frame_w, frame_h):
    """
    Computes head orientation from MediaPipe face landmarks.
    Returns (yaw, pitch, roll) in degrees or None if it fails.
    """
    img_points = []
    for idx in LANDMARK_IDS:
        lm = landmarks[idx]
        img_points.append([lm.x * frame_w, lm.y * frame_h])
    img_points = np.array(img_points, dtype=np.float64)

    focal_length = frame_w
    cam_matrix = np.array([
        [focal_length, 0,            frame_w / 2],
        [0,            focal_length, frame_h / 2],
        [0,            0,            1          ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_POINTS, img_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

    yaw   = angles[1]
    pitch = angles[0]
    roll  = angles[2]

    return yaw, pitch, roll


def get_gaze_eccentricity(landmarks):
    """
    Estimates horizontal gaze deviation from the iris landmarks that
    MediaPipe's 478-point Face Mesh provides (indices 468 and 473).
    Returns (eccentricity, direction):
      eccentricity: 0.0 (iris centered in the eye) up to ~1.0 (pressed
                    against one corner)
      direction:    negative = drifting toward image-left,
                    positive = drifting toward image-right
    Returns (0.0, 0.0) if landmarks are missing/degenerate.
    """
    def ratio(iris_idx, corner_a_idx, corner_b_idx):
        iris = landmarks[iris_idx]
        a    = landmarks[corner_a_idx]
        b    = landmarks[corner_b_idx]
        x_min, x_max = sorted([a.x, b.x])
        width = x_max - x_min
        if width < 1e-4:
            return 0.5
        return max(0.0, min(1.0, (iris.x - x_min) / width))

    try:
        right_ratio = ratio(RIGHT_IRIS_IDX, *RIGHT_EYE_CORNERS)
        left_ratio  = ratio(LEFT_IRIS_IDX,  *LEFT_EYE_CORNERS)
    except (IndexError, TypeError):
        # Landmark model didn't return iris points for this frame
        return 0.0, 0.0

    right_dev = right_ratio - 0.5
    left_dev  = left_ratio  - 0.5
    avg_dev   = (right_dev + left_dev) / 2
    avg_ecc   = (abs(right_dev) + abs(left_dev))  # already *2 equivalent

    return avg_ecc, avg_dev


# ═════════════════════════════════════════════════════════════════════════════
def classify_behaviour(yaw, pitch, gaze_ecc, gaze_dir, face_detected):
    """
    Classifies candidate behaviour using a composite deviation score
    (head yaw + gaze eccentricity), with persistence filtering.
    Only fires after suspicious behaviour is sustained long enough.
    """
    global frame_counters

    if not face_detected:
        raw = "NO FACE"
    else:
        yaw_norm  = min(abs(yaw) / YAW_NORM_MAX, 1.0)
        gaze_norm = min(gaze_ecc / GAZE_NORM_MAX, 1.0)
        composite = (YAW_WEIGHT * yaw_norm) + (GAZE_WEIGHT * gaze_norm)

        if composite >= DEVIATION_THRESHOLD:
            # Direction: prefer head yaw when it's meaningfully non-zero,
            # fall back to gaze direction for eyes-only deviation.
            direction_signal = yaw if abs(yaw) > 5 else (gaze_dir * 90)
            raw = "LOOKING LEFT" if direction_signal < 0 else "LOOKING RIGHT"
        else:
            raw = "FOCUSED"

    for key in frame_counters:
        if key == raw:
            frame_counters[key] += 1
        else:
            frame_counters[key] = 0

    required = {
        "NO FACE":       NO_FACE_FRAMES_REQUIRED,
        "LOOKING LEFT":  SUSPICIOUS_FRAMES_REQUIRED,
        "LOOKING RIGHT": SUSPICIOUS_FRAMES_REQUIRED,
    }

    if raw != "FOCUSED" and frame_counters[raw] >= required.get(raw, 15):
        return raw, True

    return raw, False


# ═════════════════════════════════════════════════════════════════════════════
def send_alert_to_api(violation_type, objects, frame=None):
    """
    Sends alert to FastAPI with optional evidence frame snapshot.
    """
    payload = {
        "session_id":     SESSION_ID,
        "candidate_id":   CANDIDATE_ID,
        "station_id":     STATION_ID,
        "violation_type": violation_type,
        "objects":        objects,
        "timestamp":      datetime.utcnow().isoformat()
    }

    # Capture evidence frame if provided
    if frame is not None:
        try:
            # Resize to reduce storage size
            small = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode(
                '.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            payload["evidence_frame"] = base64.b64encode(
                buffer
            ).decode("utf-8")
        except Exception as e:
            print(f"[FRAME CAPTURE ERROR] {e}")

    try:
        requests.post(API_URL, json=payload, timeout=2)
    except Exception as e:
        print(f"[API ERROR] {e}")


def send_alert_background(violation_type, objects, frame=None):
    """Runs API call in background so video stays smooth."""
    thread = threading.Thread(
        target=send_alert_to_api,
        args=(violation_type, objects, frame),
        daemon=True
    )
    thread.start()

# ═════════════════════════════════════════════════════════════════════════════
def draw_status_box(frame, label, alert, detections):
    """Draws semi-transparent status panel in top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    colour = (0, 0, 255) if alert else (0, 255, 0)
    status = "ALERT" if alert else "NORMAL"

    cv2.putText(frame, f"Behaviour:  {label}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Status:     {status}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
    cv2.putText(frame, f"Objects:    {', '.join(detections) if detections else 'None'}",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(frame, "AIProctor v1.0  |  Press Q to quit",
                (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("Step 1: Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Step 2: Camera opened. Starting main loop...")

    print("=" * 55)
    print("  AIProctor Vision Engine v1.0")
    print("  Monitoring started. Press Q to quit.")
    print("=" * 55)

    first_frame = True

    while True:
        ret, frame = cap.read()

        if first_frame:
            print("Step 3: First frame captured. Display initializing...")
            first_frame = False

        if not ret:
            print("ERROR: Failed to read frame.")
            break

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── 1. YOLOv8 Object & Person Detection ──────────────────────────────
        yolo_results     = model(frame, verbose=False)[0]
        detected_items   = []
        persons_detected = []

        for box in yolo_results.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence < 0.45:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls_id]

            if cls_id == PERSON_CLASS_ID:
                persons_detected.append((x1, y1, x2, y2, confidence))
                continue

            if cls_id in SUSPICIOUS_CLASSES:
                detected_items.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"WARNING: {label} {confidence:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # ── 2. Multiple Person Logic ──────────────────────────────────────────
        if len(persons_detected) == 1:
            x1, y1, x2, y2, conf = persons_detected[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Candidate {conf:.0%}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        elif len(persons_detected) > 1:
            for i, (x1, y1, x2, y2, conf) in enumerate(persons_detected):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                tag = "Candidate" if i == 0 else "WARNING: UNAUTHORIZED PERSON"
                cv2.putText(frame, tag,
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            detected_items.append("unauthorized person")

        # ── 3. MediaPipe Face Detection & Head Pose ───────────────────────────
        behaviour_label = "NO FACE"
        behaviour_alert = False
        face_detected   = False

        mp_image     = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        mesh_results = face_landmarker.detect(mp_image)

        if mesh_results.face_landmarks:
            face_detected = True
            landmarks     = mesh_results.face_landmarks[0]
            pose          = get_head_pose(landmarks, frame_w, frame_h)
            gaze_ecc, gaze_dir = get_gaze_eccentricity(landmarks)

            if pose:
                yaw, pitch, roll = pose
                behaviour_label, behaviour_alert = classify_behaviour(
                    yaw, pitch, gaze_ecc, gaze_dir, face_detected
                )
                cv2.putText(frame,
                            f"Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}  "
                            f"Roll:{roll:+.1f}  Gaze:{gaze_ecc:.2f}",
                            (10, frame_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        else:
            behaviour_label, behaviour_alert = classify_behaviour(
                0, 0, 0.0, 0.0, face_detected
            )

        # ── 4. Combine Alerts ─────────────────────────────────────────────────
        object_alert = len(detected_items) > 0
        final_alert  = behaviour_alert or object_alert

        # ── 5. Draw Status Panel ──────────────────────────────────────────────
        draw_status_box(frame, behaviour_label, final_alert, detected_items)

        # ── 6. Terminal Log + Send to API ─────────────────────────────────────
        # ── 6. Terminal Log + Send to API ─────────────────────────────────
        if final_alert:
            print(f"[ALERT] Behaviour={behaviour_label} | Objects={detected_items}")

            violation = (
                f"OBJECT_DETECTED: {', '.join(detected_items)}"
                if detected_items else behaviour_label
            )

            now       = datetime.utcnow().timestamp()
            last_sent = last_alert_time.get(violation, 0)

            if now - last_sent >= ALERT_COOLDOWN_SECONDS:
                last_alert_time[violation] = now
                # Pass current frame as evidence snapshot
                send_alert_background(violation, detected_items, frame.copy())
                print(f"[SENT TO DB] {violation}")

        # ── 7. Display ────────────────────────────────────────────────────────
        cv2.imshow("AIProctor - Vision Engine v1.0", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    print("\nAIProctor session ended.")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()