import cv2
import time
import requests
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
import os
import urllib.request

# ══════════════════════════════════════════════════════════════════════════
# AIProctor System Evaluation Script
# Generates performance metrics for Chapter 4
# ══════════════════════════════════════════════════════════════════════════

API_URL    = "http://localhost:8000"
MODEL_PATH = "face_landmarker.task"

print("=" * 60)
print("  AIProctor — System Evaluation")
print(f"  Date: {datetime.now().strftime('%d %B %Y, %H:%M')}")
print("=" * 60)


# ── Load Models ────────────────────────────────────────────────────────────
print("\n[1] Loading models...")
yolo_model = YOLO("yolov8n.pt")

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

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
print("  Models loaded successfully")


# ══════════════════════════════════════════════════════════════════════════
# TEST 1 — Processing Latency
# Measures time from frame capture to detection completion
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] TEST 1: Processing Latency")
print("    Capturing 100 frames and measuring detection time...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

yolo_times    = []
mediapipe_times = []
total_times   = []

for i in range(100):
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO latency
    t0 = time.perf_counter()
    yolo_model(frame, verbose=False)
    t1 = time.perf_counter()
    yolo_times.append((t1 - t0) * 1000)

    # MediaPipe latency
    t2 = time.perf_counter()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    face_landmarker.detect(mp_img)
    t3 = time.perf_counter()
    mediapipe_times.append((t3 - t2) * 1000)

    total_times.append((t1 - t0) * 1000 + (t3 - t2) * 1000)

    if (i + 1) % 20 == 0:
        print(f"    {i+1}/100 frames processed...")

cap.release()

yolo_avg    = np.mean(yolo_times)
yolo_max    = np.max(yolo_times)
mp_avg      = np.mean(mediapipe_times)
mp_max      = np.max(mediapipe_times)
total_avg   = np.mean(total_times)
total_max   = np.max(total_times)
fps_avg     = 1000 / total_avg

print(f"\n  YOLOv8 Detection:")
print(f"    Average latency : {yolo_avg:.1f} ms")
print(f"    Max latency     : {yolo_max:.1f} ms")
print(f"\n  MediaPipe Pose Estimation:")
print(f"    Average latency : {mp_avg:.1f} ms")
print(f"    Max latency     : {mp_max:.1f} ms")
print(f"\n  Combined Pipeline:")
print(f"    Average latency : {total_avg:.1f} ms")
print(f"    Max latency     : {total_max:.1f} ms")
print(f"    Effective FPS   : {fps_avg:.1f} frames/second")

latency_result = "PASS" if fps_avg >= 10 else "FAIL"
print(f"    Result          : {latency_result} "
      f"(threshold: 10 FPS minimum)")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2 — Object Detection Accuracy
# Tests detection of phone and book with known ground truth
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] TEST 2: Object Detection Accuracy")
print("    This test requires manual cooperation.")
print("    Follow the instructions carefully.\n")

SUSPICIOUS = {67: "cell phone", 73: "book", 63: "laptop"}

def run_detection_test(scenario_name, should_detect, instruction, frames=30):
    """
    Runs a detection test scenario.
    Returns (true_positives, false_positives, false_negatives)
    """
    print(f"    SCENARIO: {scenario_name}")
    print(f"    Instruction: {instruction}")
    input("    Press ENTER when ready...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detections = []
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret:
            continue
        results = yolo_model(frame, verbose=False)[0]
        found   = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf >= 0.45 and cls_id in SUSPICIOUS:
                found = True
                break
        detections.append(found)
    cap.release()

    detected_count = sum(detections)
    detection_rate = detected_count / frames * 100

    if should_detect:
        tp = detected_count
        fn = frames - detected_count
        fp = 0
        print(f"    Detected in {detected_count}/{frames} frames "
              f"({detection_rate:.0f}%)")
        print(f"    True Positives: {tp} | False Negatives: {fn}")
    else:
        tp = 0
        fn = 0
        fp = detected_count
        print(f"    False alerts in {detected_count}/{frames} frames "
              f"({detection_rate:.0f}%)")
        print(f"    False Positives: {fp}")

    return tp, fp, fn

# Run scenarios
print("  Running 4 detection scenarios (30 frames each)...\n")

tp1, fp1, fn1 = run_detection_test(
    "Phone Present",
    should_detect=True,
    instruction="Hold your mobile phone clearly visible to the camera"
)
print()

tp2, fp2, fn2 = run_detection_test(
    "Book Present",
    should_detect=True,
    instruction="Hold a book or notebook clearly visible to the camera"
)
print()

tp3, fp3, fn3 = run_detection_test(
    "No Object (Clean)",
    should_detect=False,
    instruction="Sit normally — no prohibited objects, just your face"
)
print()

tp4, fp4, fn4 = run_detection_test(
    "Phone Hidden/Partial",
    should_detect=True,
    instruction="Hold your phone partially hidden — partly under the desk"
)
print()

# Calculate metrics
total_tp = tp1 + tp2 + tp4
total_fp = fp1 + fp2 + fp3 + fp4
total_fn = fn1 + fn2 + fn4

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0)

print("  OBJECT DETECTION SUMMARY:")
print(f"    True Positives  : {total_tp}")
print(f"    False Positives : {total_fp}")
print(f"    False Negatives : {total_fn}")
print(f"    Precision       : {precision:.2%}")
print(f"    Recall          : {recall:.2%}")
print(f"    F1 Score        : {f1:.2%}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 3 — Behaviour Detection Accuracy
# Tests head pose and NO FACE detection
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] TEST 3: Behaviour Detection Accuracy")
print("    Running 5 behaviour scenarios...\n")

FACE_3D = np.array([
    [0.0,    0.0,    0.0  ],
    [0.0,  -330.0, -65.0  ],
    [-225.0, 170.0, -135.0],
    [225.0,  170.0, -135.0],
    [-150.0,-150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)
LANDMARK_IDS = [1, 152, 263, 33, 287, 57]
YAW_THRESH   = 20

def detect_behaviour(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    mesh   = face_landmarker.detect(mp_img)

    if not mesh.face_landmarks:
        return "NO FACE"

    lms    = mesh.face_landmarks[0]
    fh, fw = frame.shape[:2]
    pts    = np.array([
        [lms[i].x * fw, lms[i].y * fh]
        for i in LANDMARK_IDS
    ], dtype=np.float64)
    cam    = np.array([
        [fw, 0,  fw/2],
        [0,  fw, fh/2],
        [0,  0,  1   ]
    ], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D, pts, cam,
        np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return "FOCUSED"
    rmat, _    = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    yaw        = angles[1]
    if yaw < -YAW_THRESH: return "LOOKING LEFT"
    if yaw >  YAW_THRESH: return "LOOKING RIGHT"
    return "FOCUSED"

behaviour_scenarios = [
    ("Focused",       "FOCUSED",       "Look directly at the screen/camera"),
    ("Looking Left",  "LOOKING LEFT",  "Turn your head clearly to the LEFT"),
    ("Looking Right", "LOOKING RIGHT", "Turn your head clearly to the RIGHT"),
    ("No Face",       "NO FACE",       "Cover the camera or move out of frame"),
    ("Return Focus",  "FOCUSED",       "Look directly at the screen again"),
]

beh_tp = 0
beh_total = 0

for name, expected, instruction in behaviour_scenarios:
    print(f"    SCENARIO: {name}")
    print(f"    Instruction: {instruction}")
    input("    Press ENTER when ready...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    results_list = []
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            continue
        results_list.append(detect_behaviour(frame))
    cap.release()

    correct = sum(1 for r in results_list if r == expected)
    rate    = correct / len(results_list) * 100
    beh_tp    += correct
    beh_total += len(results_list)
    print(f"    Correct detections: {correct}/{len(results_list)} "
          f"({rate:.0f}%)\n")

beh_accuracy = beh_tp / beh_total * 100 if beh_total > 0 else 0
print(f"  BEHAVIOUR DETECTION ACCURACY: {beh_accuracy:.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# TEST 4 — API Response Time
# Measures how fast alerts reach the database
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] TEST 4: API Response Time")
print("    Sending 20 test alerts and measuring response time...")

api_times = []
for i in range(20):
    payload = {
        "session_id":     "EVAL_SESSION",
        "candidate_id":   "EVAL_CANDIDATE",
        "station_id":     "EVAL_STATION",
        "violation_type": "EVALUATION_TEST",
        "objects":        [],
        "timestamp":      datetime.utcnow().isoformat()
    }
    t0 = time.perf_counter()
    try:
        res = requests.post(
            f"{API_URL}/alerts", json=payload, timeout=5
        )
        t1  = time.perf_counter()
        api_times.append((t1 - t0) * 1000)
    except Exception as e:
        print(f"    ERROR: {e}")

if api_times:
    api_avg = np.mean(api_times)
    api_max = np.max(api_times)
    print(f"    Average response time : {api_avg:.1f} ms")
    print(f"    Max response time     : {api_max:.1f} ms")
    api_result = "PASS" if api_avg < 500 else "FAIL"
    print(f"    Result                : {api_result} "
          f"(threshold: 500ms)")

    # Clean up test alerts
    try:
        requests.delete(f"{API_URL}/alerts")
        print("    Test alerts cleared from database")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  EVALUATION SUMMARY")
print("=" * 60)
print(f"\n  Processing Performance:")
print(f"    Pipeline latency (avg) : {total_avg:.1f} ms")
print(f"    Effective frame rate   : {fps_avg:.1f} FPS")
print(f"    Latency test           : {latency_result}")

print(f"\n  Object Detection:")
print(f"    Precision              : {precision:.2%}")
print(f"    Recall                 : {recall:.2%}")
print(f"    F1 Score               : {f1:.2%}")

print(f"\n  Behaviour Detection:")
print(f"    Overall accuracy       : {beh_accuracy:.1f}%")

if api_times:
    print(f"\n  API Performance:")
    print(f"    Avg alert response     : {api_avg:.1f} ms")
    print(f"    API test               : {api_result}")

print(f"\n  Evaluation completed: "
      f"{datetime.now().strftime('%d %B %Y, %H:%M')}")
print("=" * 60)
print("\n  Save these results — they go directly into Chapter 4.")
print("  Run this script again if you need to repeat any test.\n")

face_landmarker.close()