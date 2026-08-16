import sys
import os
import base64
import threading
import requests
import cv2
import numpy as np
import face_recognition
from datetime import datetime, timezone
import queue
import logging
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QProgressBar, QInputDialog, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QUrl, QObject
)
from PyQt6.QtGui import (
    QPixmap, QImage, QKeySequence
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

# ── Crash / debug logging ───────────────────────────────────────────────────
# If this app runs windowed (no visible console), print() output and
# uncaught exceptions vanish with no trace. This writes everything to a
# log file next to the script so a crash can actually be diagnosed.
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aiproctor_debug.log"
)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AIProctor")


def _log_uncaught_exception(exc_type, exc_value, exc_tb):
    logger.critical(
        "UNCAUGHT EXCEPTION — app is about to crash:\n" +
        "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    )
    # Still print to console if one exists.
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = _log_uncaught_exception

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL         = "http://127.0.0.1:8000"
MATCH_TOLERANCE = 0.5
EXAM_DURATION   = 60 * 60   # 1 hour in seconds
ADMIN_PASSWORD  = "admin123"
ADMIN_SHORTCUT  = "Ctrl+Shift+Q"

# ── Gaze / Composite Deviation Scoring ────────────────────────────────────────
# Head-yaw alone misses "eyes only" cheating (head straight, eyes to the
# side). MediaPipe's face_landmarker.task model outputs 478 landmarks by
# default; 468/473 are the right/left iris centers, used here alongside
# yaw to build a single composite deviation score.
RIGHT_IRIS_IDX    = 468
LEFT_IRIS_IDX     = 473
RIGHT_EYE_CORNERS = (33, 133)    # (outer, inner)
LEFT_EYE_CORNERS  = (362, 263)   # (inner, outer)

YAW_NORM_MAX        = 45.0   # yaw magnitude treated as "full" deviation
GAZE_NORM_MAX        = 0.5    # gaze eccentricity treated as "full" deviation
YAW_WEIGHT           = 0.6
GAZE_WEIGHT          = 0.4
DEVIATION_THRESHOLD  = 0.45   # composite score above this = suspicious

# ── Persistence Thresholds ─────────────────────────────────────────────────
# Time-based (wall-clock seconds), not frame-count-based — CPU-only
# inference has variable/low FPS, so a frame-count threshold made the
# actual real-world delay unpredictable and unreliable. Shared between
# VisionEngineThread (which tracks elapsed time) and ExamWindow (which
# turns duration into an escalating violation weight).
GAZE_PERSIST_SECONDS    = 5.0   # looking-away/no-face/multiple-faces
BLOCKED_PERSIST_SECONDS = 3.0   # camera blocked — deliberate, flag faster


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
        return 0.0, 0.0

    right_dev = right_ratio - 0.5
    left_dev  = left_ratio  - 0.5
    avg_dev   = (right_dev + left_dev) / 2
    avg_ecc   = (abs(right_dev) + abs(left_dev))  # already *2 equivalent

    return avg_ecc, avg_dev


# ═════════════════════════════════════════════════════════════════════════════
# SHARED CAMERA MANAGER
# Single camera instance shared between overlay display
# and vision engine monitoring — prevents conflict
# ═════════════════════════════════════════════════════════════════════════════
class CameraManager(QObject):
    """
    Owns the single camera capture instance.
    Distributes frames to all subscribers via callbacks.
    """
    def __init__(self):
        super().__init__()
        self.cap         = None
        self.running     = False
        self.subscribers = []   # list of callables that receive frames
        self.thread      = None
        self.lock        = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.thread  = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self.thread.start()
        print("[CAMERA] Shared camera manager started")

    def subscribe(self, callback):
        """Add a frame receiver callback."""
        with self.lock:
            self.subscribers.append(callback)

    def unsubscribe(self, callback):
        """Remove a frame receiver callback."""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def _capture_loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        for cb in self.subscribers:
                            try:
                                cb(frame.copy())
                            except Exception:
                                pass

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[CAMERA] Shared camera manager stopped")


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL EVENT FILTER
# Installed at application level — intercepts ALL keyboard events
# before any widget (including QWebEngineView) sees them
# ═════════════════════════════════════════════════════════════════════════════
class GlobalKeyFilter(QObject):
    """
    Application-level event filter.
    Blocks student shortcuts and catches admin unlock combination.
    """
    admin_shortcut_pressed = pyqtSignal()

    def __init__(self, exam_window_ref):
        super().__init__()
        self.exam_window_ref = exam_window_ref

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui  import QKeyEvent

        if event.type() != QEvent.Type.KeyPress:
            return False

        key  = event.key()
        mods = event.modifiers()

        CTRL  = Qt.KeyboardModifier.ControlModifier
        SHIFT = Qt.KeyboardModifier.ShiftModifier
        ALT   = Qt.KeyboardModifier.AltModifier

        # ── Admin unlock: Ctrl+Shift+Q ────────────────────────────────────
        if (key == Qt.Key.Key_Q and
                mods == (CTRL | SHIFT)):
            self.admin_shortcut_pressed.emit()
            return True   # consume — do not pass to browser

        # ── Block Alt+F4, Alt+Tab etc ─────────────────────────────────────
        if mods & ALT:
            return True

        # ── Block specific keys ───────────────────────────────────────────
        blocked_keys = [
            Qt.Key.Key_Escape,
            Qt.Key.Key_F4,
            Qt.Key.Key_F11,
            Qt.Key.Key_F12,
            Qt.Key.Key_Super_L,
            Qt.Key.Key_Super_R,
        ]
        if key in blocked_keys:
            return True

        # ── Block Ctrl+W, Ctrl+T, Ctrl+N, Ctrl+Q etc ─────────────────────
        if mods == CTRL:
            blocked_ctrl = [
                Qt.Key.Key_W, Qt.Key.Key_T, Qt.Key.Key_N,
                Qt.Key.Key_Q, Qt.Key.Key_H, Qt.Key.Key_D,
            ]
            if key in blocked_ctrl:
                return True

        return False  # pass everything else through


# ═════════════════════════════════════════════════════════════════════════════
# VISION ENGINE THREAD
# Uses shared camera frames via callback subscription
# ═════════════════════════════════════════════════════════════════════════════
class VisionEngineThread(QThread):
    # duration_cs = how many CENTISECONDS (1/100s) this exact behaviour
    # has now continuously persisted, measured in wall-clock time (0 for
    # one-off detections like objects/persons that aren't tracked this
    # way). This is what lets the UI escalate severity based on actual
    # sustained duration, instead of how many times the (cooldown-gated)
    # alert happened to re-fire.
    alert_signal  = pyqtSignal(str, list, int)
    ready_signal  = pyqtSignal()
    # Fires every frame with the current raw classification (FOCUSED,
    # LOOKING LEFT/RIGHT, NO FACE, MULTIPLE FACES) — independent of the
    # persistence/cooldown gating that alert_signal uses, so the UI can
    # show live gaze movement instead of only delayed violations.
    status_signal = pyqtSignal(str)

    def __init__(self, candidate_id, session_id, station_id):
        super().__init__()
        self.candidate_id      = candidate_id
        self.session_id        = session_id
        self.station_id        = station_id
        self.running           = False
        self.models_ready      = False
        self.last_alert_time   = {}
        self.COOLDOWN          = 5
        self.model             = None
        self.face_landmarker   = None
        self.frame_queue       = queue.Queue(maxsize=2)

        # Dedicated looking-away tracker (separate from the generic
        # per-key dict used for NO FACE/MULTIPLE FACES/CAMERA BLOCKED)
        # because gaze classification is the noisiest signal — it sits
        # right near a threshold and can flicker frame-to-frame between
        # LOOKING LEFT/RIGHT/FOCUSED even during a genuine sustained
        # look-away. A brief single-frame blip back to FOCUSED (or a
        # flip between LEFT/RIGHT) should NOT reset the whole timer.
        self._gaze_away_start    = None   # when the current away-run began
        self._gaze_away_last_seen = None  # last frame classified as away
        self._gaze_away_evidence = None
        self._GAZE_GLITCH_GRACE  = 1.0    # seconds of tolerance

    def receive_frame(self, frame):
        """Called by CameraManager with each new frame."""
        if self.running:
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def preload_models(self):
        from ultralytics import YOLO
        import mediapipe as mp
        import os
        import urllib.request

        print("[MONITOR] Preloading models...")
        self.model = YOLO("yolov8n.pt")
        print("[MONITOR] YOLOv8 ready")

        MODEL_PATH = "face_landmarker.task"
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
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)
        self.models_ready    = True
        print("[MONITOR] All models ready")
        self.ready_signal.emit()

    def send_alert(self, violation_type, objects, frame=None,
                   duration_cs=0):
        now       = datetime.now(timezone.utc).timestamp()
        last_sent = self.last_alert_time.get(violation_type, 0)
        if now - last_sent < self.COOLDOWN:
            return
        self.last_alert_time[violation_type] = now

        payload = {
            "session_id":     self.session_id,
            "candidate_id":   self.candidate_id,
            "station_id":     self.station_id,
            "violation_type": violation_type,
            "objects":        objects,
            "timestamp":      datetime.now(timezone.utc).isoformat()
        }

        # Encode evidence frame as base64 JPEG
        if frame is not None:
            try:
                # Annotate frame with violation label
                annotated = frame.copy()
                cv2.putText(
                    annotated,
                    f"VIOLATION: {violation_type}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )
                cv2.putText(
                    annotated,
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )
                small = cv2.resize(annotated, (320, 240))
                _, buf = cv2.imencode(
                    '.jpg', small,
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                payload["evidence_frame"] = base64.b64encode(
                    buf
                ).decode("utf-8")
                print(f"[EVIDENCE] Frame captured for {violation_type} "
                      f"({len(payload['evidence_frame'])} chars)")
            except Exception as e:
                print(f"[FRAME ERROR] {e}")
        else:
            print(f"[EVIDENCE] No frame for {violation_type}")

        try:
            res = requests.post(
                f"{API_URL}/alerts", json=payload, timeout=3
            )
            if res.status_code == 200:
                print(f"[ALERT SENT] {violation_type} | "
                      f"Evidence={'YES' if frame is not None else 'NO'}")
        except Exception as e:
            print(f"[API ERROR] {e}")

        self.alert_signal.emit(violation_type, objects, duration_cs)

    def run(self):
        import mediapipe as mp
        import time

        while not self.models_ready:
            time.sleep(0.1)

        self.running = True
        print(f"[MONITOR] Started for {self.candidate_id}")

        FACE_3D = np.array([
            [0.0,    0.0,    0.0  ],
            [0.0,  -330.0, -65.0  ],
            [-225.0, 170.0, -135.0],
            [225.0,  170.0, -135.0],
            [-150.0,-150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float64)

        LANDMARK_IDS   = [1, 152, 263, 33, 287, 57]
        SUSPICIOUS     = {
            67: "cell phone",
            73: "book",
            63: "laptop",
            66: "keyboard",
            64: "mouse",
            65: "remote",
        }
        PERSON_ID      = 0
        YAW_THRESH     = 20
        # Time-based, not frame-count-based: CPU-only inference has
        # variable/low FPS, so counting frames made the actual delay
        # unpredictable. Tracking wall-clock seconds means "5 seconds"
        # means 5 seconds regardless of how fast frames arrive.
        REQUIRED_SECONDS         = GAZE_PERSIST_SECONDS
        REQUIRED_SECONDS_BLOCKED = BLOCKED_PERSIST_SECONDS

        # When each behaviour was first seen (None = not currently active)
        state_start_time = {
            "NO FACE":         None,
            "MULTIPLE FACES":  None,
            "CAMERA BLOCKED":  None,
        }
        # Whether we've already fired the "sustained" alert for the
        # CURRENT continuous run of this behaviour (so we alert once
        # per continuous occurrence, not once per cooldown window).
        state_fired = {k: False for k in state_start_time}
        # Frame captured at the moment each behaviour started
        trigger_frames = {k: None for k in state_start_time}

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except Exception:
                continue

            try:
                self._process_frame(
                    frame, FACE_3D, LANDMARK_IDS, SUSPICIOUS, PERSON_ID,
                    YAW_THRESH, REQUIRED_SECONDS, REQUIRED_SECONDS_BLOCKED,
                    state_start_time, state_fired, trigger_frames, mp, time
                )
            except Exception as e:
                # A single bad frame (landmark edge case, solvePnP
                # failure, etc.) must never kill the whole monitoring
                # thread — log it and keep watching the next frame.
                logger.error(
                    f"[VISION FRAME ERROR] {type(e).__name__}: {e}\n"
                    + traceback.format_exc()
                )
                print(f"[VISION FRAME ERROR] {type(e).__name__}: {e}")

        self.face_landmarker.close()
        print(f"[MONITOR] Stopped for {self.candidate_id}")

    def _process_frame(
        self, frame, FACE_3D, LANDMARK_IDS, SUSPICIOUS, PERSON_ID,
        YAW_THRESH, REQUIRED_SECONDS, REQUIRED_SECONDS_BLOCKED,
        state_start_time, state_fired, trigger_frames, mp, time_module
    ):
        # Make a clean copy for evidence capture
        evidence_frame = frame.copy()
        now = time_module.time()

        fh, fw = frame.shape[:2]

        # ── Camera-blocked check (cheap, runs before any heavy model) ──
        # A covered/obstructed lens produces a near-uniform, often dark
        # frame. Checking this first also saves YOLO/MediaPipe work on
        # a frame that has nothing meaningful to detect anyway.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 15 or sharpness < 8:
            raw = "CAMERA BLOCKED"
            self.status_signal.emit(raw)
            self._handle_gaze_state(None, now, evidence_frame)
            self._update_state_and_maybe_alert(
                raw, now, REQUIRED_SECONDS_BLOCKED, evidence_frame,
                state_start_time, state_fired, trigger_frames, []
            )
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── YOLO detection ────────────────────────────────────────────
        results  = self.model(frame, verbose=False)[0]
        detected = []
        persons  = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if conf < 0.45:
                continue
            if cls_id == PERSON_ID:
                persons.append(cls_id)
            elif cls_id in SUSPICIOUS:
                detected.append(SUSPICIOUS[cls_id])

        # Send object alerts immediately with current frame
        if detected:
            self.send_alert(
                f"OBJECT_DETECTED: {', '.join(detected)}",
                detected,
                evidence_frame
            )
        if len(persons) > 1:
            self.send_alert(
                "UNAUTHORIZED PERSON IN FRAME",
                ["unauthorized person"],
                evidence_frame
            )

        # ── MediaPipe: face count + behaviour detection ────────────────
        # One model call gives us face count (for MULTIPLE FACES / NO
        # FACE) and, for a single face, the landmarks used for gaze.
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )
        mesh = self.face_landmarker.detect(mp_img)
        n_faces = len(mesh.face_landmarks) if mesh.face_landmarks else 0

        if n_faces > 1:
            raw = "MULTIPLE FACES"

        elif n_faces == 0:
            raw = "NO FACE"

        else:
            lms     = mesh.face_landmarks[0]
            img_pts = np.array([
                [lms[i].x * fw, lms[i].y * fh]
                for i in LANDMARK_IDS
            ], dtype=np.float64)
            cam_mat = np.array([
                [fw, 0,  fw/2],
                [0,  fw, fh/2],
                [0,  0,  1   ]
            ], dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(
                FACE_3D, img_pts, cam_mat,
                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
            )
            gaze_ecc, gaze_dir = get_gaze_eccentricity(lms)

            if ok:
                rmat, _    = cv2.Rodrigues(rvec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                yaw        = angles[1]

                yaw_norm  = min(abs(yaw) / YAW_NORM_MAX, 1.0)
                gaze_norm = min(gaze_ecc / GAZE_NORM_MAX, 1.0)
                composite = (
                    (YAW_WEIGHT * yaw_norm) + (GAZE_WEIGHT * gaze_norm)
                )

                if composite >= DEVIATION_THRESHOLD:
                    # Direction: prefer head yaw when meaningfully
                    # non-zero, fall back to gaze-only deviation
                    # (head straight, eyes to the side).
                    direction_signal = (
                        yaw if abs(yaw) > 5 else (gaze_dir * 90)
                    )
                    raw = (
                        "LOOKING LEFT" if direction_signal < 0
                        else "LOOKING RIGHT"
                    )
                else:
                    raw = "FOCUSED"
            else:
                raw = "FOCUSED"

        # ── Live status: emitted every frame, regardless of persistence
        # or alert cooldown, so the UI can reflect gaze movement in
        # real time rather than only once a violation is confirmed.
        self.status_signal.emit(raw)

        if raw in ("LOOKING LEFT", "LOOKING RIGHT"):
            # Gaze is the noisiest signal (right near a threshold, can
            # flicker frame-to-frame) — uses the dedicated tracker with
            # glitch tolerance instead of the generic one.
            self._handle_gaze_state(raw, now, evidence_frame)
            # Still clears the generic NO FACE/MULTIPLE FACES/CAMERA
            # BLOCKED timers, since we're clearly in none of those.
            self._update_state_and_maybe_alert(
                "__NONE__", now, REQUIRED_SECONDS, evidence_frame,
                state_start_time, state_fired, trigger_frames, []
            )
        else:
            # FOCUSED, NO FACE, or MULTIPLE FACES this frame — also
            # clears any in-progress gaze-away run once the glitch
            # grace window has genuinely elapsed.
            self._handle_gaze_state(None, now, evidence_frame)
            self._update_state_and_maybe_alert(
                raw, now, REQUIRED_SECONDS, evidence_frame,
                state_start_time, state_fired, trigger_frames, []
            )

    def _update_state_and_maybe_alert(
        self, raw, now, required_seconds, evidence_frame,
        state_start_time, state_fired, trigger_frames, objects
    ):
        """
        Wall-clock version of the persistence check: a behaviour must
        continuously hold for `required_seconds` before it's flagged as
        a violation — measured in real time, not frame count, so it's
        accurate regardless of the pipeline's actual FPS.
        """
        for key in state_start_time:
            if key != raw:
                # Behaviour interrupted — reset its timer entirely.
                state_start_time[key] = None
                state_fired[key]      = False
                trigger_frames[key]   = None

        if raw == "FOCUSED" or raw not in state_start_time:
            return

        if state_start_time[raw] is None:
            # Just started — start the clock and grab evidence now.
            state_start_time[raw] = now
            trigger_frames[raw]   = evidence_frame.copy()
            state_fired[raw]      = False

        elapsed = now - state_start_time[raw]

        if elapsed >= required_seconds and not state_fired[raw]:
            # Sustained long enough — fire exactly once for this
            # continuous occurrence (further escalation, if it keeps
            # going, is handled by re-fires past the cooldown window).
            state_fired[raw] = True
            _tf = trigger_frames.get(raw)
            stored_frame = _tf if _tf is not None else evidence_frame
            self.send_alert(
                raw, objects, stored_frame,
                duration_cs=int(elapsed * 100)  # encode as centiseconds
            )
        elif elapsed >= required_seconds and state_fired[raw]:
            # Already fired once for this occurrence — later re-fires
            # (gated by send_alert's own cooldown) still carry the
            # growing elapsed time so escalation reflects reality.
            _tf = trigger_frames.get(raw)
            stored_frame = _tf if _tf is not None else evidence_frame
            self.send_alert(
                raw, objects, stored_frame,
                duration_cs=int(elapsed * 100)
            )

    def _handle_gaze_state(self, direction, now, evidence_frame):
        """
        Dedicated persistence tracker for LOOKING LEFT/RIGHT, with
        tolerance for brief classification glitches. Unlike the
        generic per-key tracker, this treats LEFT/RIGHT as one
        continuous "looking away" run — flipping direction, or a
        single noisy frame dropping back to FOCUSED, does not reset
        progress unless the interruption lasts longer than the grace
        window.

        direction: "LOOKING LEFT" / "LOOKING RIGHT" if away this
                   frame, otherwise None (i.e. FOCUSED this frame).
        """
        grace = self._GAZE_GLITCH_GRACE

        if direction is not None:
            gap_too_long = (
                self._gaze_away_last_seen is not None
                and (now - self._gaze_away_last_seen) > grace
            )
            if self._gaze_away_start is None or gap_too_long:
                # Either a genuinely new away-run, or the previous one
                # had a real (non-glitch) gap — start fresh.
                self._gaze_away_start    = now
                self._gaze_away_evidence = evidence_frame.copy()

            self._gaze_away_last_seen = now
            elapsed = now - self._gaze_away_start

            if elapsed >= GAZE_PERSIST_SECONDS:
                stored = (
                    self._gaze_away_evidence
                    if self._gaze_away_evidence is not None
                    else evidence_frame
                )
                self.send_alert(
                    direction, [], stored,
                    duration_cs=int(elapsed * 100)
                )
        else:
            # FOCUSED this frame — only treat as a real interruption
            # (and reset progress) if we've been focused longer than
            # the glitch-tolerance window.
            if (
                self._gaze_away_last_seen is not None
                and (now - self._gaze_away_last_seen) > grace
            ):
                self._gaze_away_start     = None
                self._gaze_away_last_seen = None
                self._gaze_away_evidence  = None

    def stop(self):
        self.running = False


# ═════════════════════════════════════════════════════════════════════════════
# FACE VERIFY THREAD
# ═════════════════════════════════════════════════════════════════════════════
class FaceVerifyThread(QThread):
    result_signal = pyqtSignal(bool, str)

    def __init__(self, frame, passport_b64):
        super().__init__()
        self.frame        = frame
        self.passport_b64 = passport_b64

    def run(self):
        try:
            passport_bytes = base64.b64decode(self.passport_b64)
            passport_arr   = np.frombuffer(passport_bytes, dtype=np.uint8)
            passport_img   = cv2.imdecode(passport_arr, cv2.IMREAD_COLOR)
            passport_rgb   = cv2.cvtColor(passport_img, cv2.COLOR_BGR2RGB)

            passport_locs = face_recognition.face_locations(passport_rgb)
            if not passport_locs:
                self.result_signal.emit(
                    False, "Could not detect face in passport photo."
                )
                return
            passport_enc = face_recognition.face_encodings(
                passport_rgb, passport_locs
            )[0]

            live_rgb  = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            live_locs = face_recognition.face_locations(live_rgb)
            if not live_locs:
                self.result_signal.emit(
                    False, "No face detected. Please look at the camera."
                )
                return
            live_enc = face_recognition.face_encodings(
                live_rgb, live_locs
            )[0]

            distance = face_recognition.face_distance(
                [passport_enc], live_enc
            )[0]
            match = distance <= MATCH_TOLERANCE

            if match:
                confidence = round((1 - distance) * 100, 1)
                self.result_signal.emit(
                    True, f"Identity verified ({confidence}% match)"
                )
            else:
                self.result_signal.emit(
                    False,
                    "Face does not match passport photo. Access denied."
                )
        except Exception as e:
            self.result_signal.emit(False, f"Verification error: {str(e)}")


# ═════════════════════════════════════════════════════════════════════════════
# LOADING SCREEN
# ═════════════════════════════════════════════════════════════════════════════
class LoadingScreen(QWidget):
    ready = pyqtSignal()

    def __init__(self, vision_thread, student_data, camera_manager):
        super().__init__()
        self.vision_thread  = vision_thread
        self.student_data   = student_data
        self.camera_manager = camera_manager
        self.models_ready   = False
        self.camera_ready   = False
        self.setup_ui()

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(200)
        self.progress_val = 0

        if self.vision_thread.models_ready:
            self.models_ready = True
            self.status_label.setText("Warming up camera...")
        else:
            self.vision_thread.ready_signal.connect(self.on_models_ready)

        # Check camera every second
        self.camera_check_timer = QTimer()
        self.camera_check_timer.timeout.connect(self.check_camera)
        self.camera_check_timer.start(1000)

    def setup_ui(self):
        self.setWindowTitle("AIProctor — Starting")
        self.setFixedSize(520, 320)
        self.setStyleSheet("background: #0f172a;")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo = QLabel("AIProctor")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet(
            "color: #38bdf8; font-size: 32px; font-weight: 800;"
        )
        layout.addWidget(logo)

        name = QLabel(f"Welcome, {self.student_data['full_name']}")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet("color: #e2e8f0; font-size: 16px;")
        layout.addWidget(name)

        self.status_label = QLabel("Starting monitoring system...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #64748b; font-size: 13px;")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet("""
            QProgressBar {
                background: #1e293b;
                border-radius: 3px;
                border: none;
            }
            QProgressBar::chunk {
                background: #38bdf8;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress)

        self.steps_label = QLabel("Loading object detection model...")
        self.steps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.steps_label.setStyleSheet("color: #475569; font-size: 11px;")
        layout.addWidget(self.steps_label)

        layout.addStretch()

        footer = QLabel("Please do not close this window")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #334155; font-size: 11px;")
        layout.addWidget(footer)

    def update_progress(self):
        if self.progress_val < 90:
            self.progress_val += 2
            self.progress.setValue(self.progress_val)
        if self.progress_val < 30:
            self.steps_label.setText("Loading object detection model...")
        elif self.progress_val < 60:
            self.steps_label.setText("Loading face analysis model...")
        elif self.progress_val < 90:
            self.steps_label.setText("Warming up camera...")

    def on_models_ready(self):
        self.models_ready = True
        self.status_label.setText("Warming up camera...")
        self.check_ready()

    def check_camera(self):
        """Checks shared camera manager is delivering frames."""
        if self.camera_manager.cap and self.camera_manager.cap.isOpened():
            self.camera_ready = True
            self.camera_check_timer.stop()
            self.steps_label.setText("Camera confirmed ✓")
            self.check_ready()

    def check_ready(self):
        if self.models_ready and self.camera_ready:
            self.progress_timer.stop()
            self.progress.setValue(100)
            self.status_label.setText(
                "All systems ready. Starting exam..."
            )
            self.status_label.setStyleSheet(
                "color: #22c55e; font-size: 13px; font-weight: 600;"
            )
            self.steps_label.setText(
                "Models ✓  Camera ✓  Monitoring ✓"
            )
            QTimer.singleShot(1500, self.ready.emit)


# ═════════════════════════════════════════════════════════════════════════════
# EXAM WINDOW
# ═════════════════════════════════════════════════════════════════════════════
class ExamWindow(QMainWindow):
    exam_ended = pyqtSignal()

    # Number of flagged violations that automatically ends the exam.
    MAX_VIOLATIONS     = 15
    # Start warning the candidate once this many violations remain.
    WARNING_REMAINING  = 3

    # Base severity per violation type — added to alert_count instead
    # of a flat 1, so more serious violations reach MAX_VIOLATIONS
    # faster. Anything not listed (e.g. OBJECT_DETECTED: ...) uses 1.
    VIOLATION_WEIGHT = {
        "CAMERA BLOCKED":              3,
        "MULTIPLE FACES":              2,
        "UNAUTHORIZED PERSON IN FRAME": 2,
        "NO FACE":                     1,
        "LOOKING LEFT":                1,
        "LOOKING RIGHT":               1,
    }
    # How much extra weight each consecutive repeat of the SAME
    # violation type adds on top of its base weight, capped — this is
    # what makes prolonged looking-away (or a prolonged blocked camera,
    # etc.) escalate instead of costing the same as a brief one-off.
    ESCALATION_STEP = 1
    ESCALATION_CAP  = 3

    def __init__(self, student_data, vision_thread, camera_manager):
        super().__init__()
        self.student_data   = student_data
        self.vision_thread  = vision_thread
        self.camera_manager = camera_manager
        self.alert_count    = 0
        self.time_left      = EXAM_DURATION
        self.exam_active    = True
        self.key_filter     = None
        # Tracks consecutive repeats of the same violation type, so
        # a prolonged behaviour (e.g. looking away for a long stretch)
        # escalates in weight instead of costing the same each time.
        self._last_violation_type = None
        self._violation_streak    = 0
        self.setup_ui()
        self.setup_lockdown()
        self.start_monitoring()
        self.start_timer()

    def setup_ui(self):
        self.setWindowTitle("AIProctor — Examination")
        self.showFullScreen()
        self.setStyleSheet("background: #0f172a;")

        # Remove all window controls
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(
            "background: #1e293b; border-bottom: 1px solid #334155;"
        )
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(20, 0, 20, 0)

        title = QLabel("AIProctor — Secure Examination")
        title.setStyleSheet(
            "color: #38bdf8; font-size: 13px; font-weight: 700;"
        )
        tb.addWidget(title)
        tb.addStretch()

        student_info = QLabel(
            f"{self.student_data['full_name']}  ·  "
            f"{self.student_data['registration_number']}"
        )
        student_info.setStyleSheet("color: #94a3b8; font-size: 12px;")
        tb.addWidget(student_info)

        tb.addSpacing(24)

        self.alert_label = QLabel("Alerts: 0")
        self.alert_label.setStyleSheet("color: #64748b; font-size: 12px;")
        tb.addWidget(self.alert_label)

        tb.addSpacing(24)

        self.timer_label = QLabel(self.format_time(self.time_left))
        self.timer_label.setStyleSheet(
            "color: #22c55e; font-size: 14px; font-weight: 700;"
        )
        tb.addWidget(self.timer_label)

        tb.addSpacing(24)

        self.monitor_label = QLabel("● Monitoring Active")
        self.monitor_label.setStyleSheet("color: #22c55e; font-size: 12px;")
        tb.addWidget(self.monitor_label)

        tb.addSpacing(16)

        self.live_status_label = QLabel("👁 FOCUSED")
        self.live_status_label.setStyleSheet(
            "color: #38bdf8; font-size: 12px; font-weight: 700;"
        )
        tb.addWidget(self.live_status_label)

        tb.addSpacing(24)

        self.submit_btn = QPushButton("Submit Exam")
        self.submit_btn.setFixedHeight(30)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background: #22c55e;
                color: #06210f;
                border: none;
                border-radius: 6px;
                padding: 0 16px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #16a34a;
            }
            QPushButton:disabled {
                background: #334155;
                color: #64748b;
            }
        """)
        self.submit_btn.clicked.connect(self.confirm_submit)
        tb.addWidget(self.submit_btn)

        layout.addWidget(topbar)

        # ── Violation flag banner (hidden until a violation fires) ────────
        self.flag_banner = QLabel("")
        self.flag_banner.setFixedHeight(36)
        self.flag_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flag_banner.setStyleSheet("""
            background: #7f1d1d;
            color: #fecaca;
            font-size: 13px;
            font-weight: 700;
            border-bottom: 2px solid #ef4444;
        """)
        self.flag_banner.setVisible(False)
        layout.addWidget(self.flag_banner)

        self.flag_banner_timer = QTimer()
        self.flag_banner_timer.setSingleShot(True)
        self.flag_banner_timer.timeout.connect(self._hide_flag_banner)

        # ── Exam browser ──────────────────────────────────────────────────
        exam_url = self.student_data.get("exam_url")
        if exam_url:
            self.browser = QWebEngineView()
            self.browser.setUrl(QUrl(exam_url))

            # Disable right-click context menu
            self.browser.setContextMenuPolicy(
                Qt.ContextMenuPolicy.NoContextMenu
            )
            layout.addWidget(self.browser)
        else:
            no_url = QLabel(
                "No exam URL assigned.\n"
                "Please inform your supervisor."
            )
            no_url.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_url.setStyleSheet("color: #64748b; font-size: 16px;")
            layout.addWidget(no_url)

        # ── Camera overlay (bottom right) ─────────────────────────────────
        self.cam_overlay = QLabel(self)
        self.cam_overlay.setFixedSize(200, 150)
        self.cam_overlay.setStyleSheet("""
            border: 2px solid #38bdf8;
            border-radius: 8px;
            background: #0f172a;
        """)
        self.cam_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_overlay.setText("Camera...")
        self.cam_overlay.raise_()

        # Subscribe to shared camera for overlay updates
        self.camera_manager.subscribe(self.on_camera_frame)

        # Timer to update overlay on UI thread
        self.overlay_frame  = None
        self.overlay_timer  = QTimer()
        self.overlay_timer.timeout.connect(self.update_cam_overlay)
        self.overlay_timer.start(100)

    def setup_lockdown(self):
        """
        Installs global keyboard filter at application level.
        This catches keys even when QWebEngineView has focus.
        """
        self.key_filter = GlobalKeyFilter(self)
        self.key_filter.admin_shortcut_pressed.connect(
            self.prompt_admin_unlock
        )
        QApplication.instance().installEventFilter(self.key_filter)
        print("[LOCKDOWN] Global key filter installed")

    def on_camera_frame(self, frame):
        """Receives frame from CameraManager for overlay display."""
        self.overlay_frame = frame

    def update_cam_overlay(self):
        """Updates camera overlay on UI thread safely."""
        if self.overlay_frame is not None:
            frame = self.overlay_frame
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w  = rgb.shape[:2]
            img   = QImage(
                rgb.data, w, h,
                w * 3, QImage.Format.Format_RGB888
            )
            pix = QPixmap.fromImage(img).scaled(
                200, 150,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.cam_overlay.setPixmap(pix)
            self.cam_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_cam_overlay()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(200, self.position_cam_overlay)

    def position_cam_overlay(self):
        if hasattr(self, 'cam_overlay'):
            margin = 16
            x = self.width()  - self.cam_overlay.width()  - margin
            y = self.height() - self.cam_overlay.height() - margin
            self.cam_overlay.move(x, y)
            self.cam_overlay.raise_()

    def start_monitoring(self):
        """
        Subscribes vision engine to camera frames and starts thread.
        Uses shared camera — no conflict with overlay.
        """
        self.camera_manager.subscribe(self.vision_thread.receive_frame)
        self.vision_thread.alert_signal.connect(self.on_alert)
        self.vision_thread.status_signal.connect(self.on_live_status)
        self.vision_thread.start()
        print(
            f"[EXAM] Monitoring active for "
            f"{self.student_data['registration_number']}"
        )

    def start_timer(self):
        self.exam_timer = QTimer()
        self.exam_timer.timeout.connect(self.tick)
        self.exam_timer.start(1000)

    def tick(self):
        if not self.exam_active:
            return
        self.time_left -= 1
        self.timer_label.setText(self.format_time(self.time_left))

        if self.time_left <= 300:
            self.timer_label.setStyleSheet(
                "color: #f97316; font-size: 14px; font-weight: 700;"
            )
        if self.time_left <= 60:
            self.timer_label.setStyleSheet(
                "color: #ef4444; font-size: 14px; font-weight: 700;"
            )
        if self.time_left <= 0:
            self.end_exam("TIME_UP")

    def format_time(self, seconds):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def on_live_status(self, raw):
        """
        Fires every frame with the current raw detection state, so the
        candidate/proctor can see gaze movement, no-face, or multiple-
        face conditions live — independent of whether they've persisted
        long enough to become a full alert.
        """
        try:
            display = {
                "FOCUSED":        ("👁 FOCUSED",             "#38bdf8"),
                "LOOKING LEFT":   ("👀 LOOKING LEFT",         "#f59e0b"),
                "LOOKING RIGHT":  ("👀 LOOKING RIGHT",        "#f59e0b"),
                "NO FACE":        ("🚫 NO FACE DETECTED",     "#f97316"),
                "MULTIPLE FACES": ("⚠ MULTIPLE FACES",        "#ef4444"),
                "CAMERA BLOCKED": ("⛔ CAMERA BLOCKED",        "#dc2626"),
            }.get(raw, (f"👁 {raw}", "#94a3b8"))

            text, colour = display
            self.live_status_label.setText(text)
            self.live_status_label.setStyleSheet(
                f"color: {colour}; font-size: 12px; font-weight: 700;"
            )
        except Exception as e:
            logger.error(f"[LIVE STATUS ERROR] {type(e).__name__}: {e}")

    def on_alert(self, violation_type, objects, duration_cs=0):
        try:
            self._handle_alert(violation_type, objects, duration_cs)
        except Exception as e:
            # A slot exception here can otherwise take the whole app
            # down silently. Log it (to file too, in case this is
            # running windowed with no visible console) and keep the
            # exam running instead.
            logger.error(
                f"[ON_ALERT ERROR] {type(e).__name__}: {e}\n"
                + traceback.format_exc()
            )
            print(f"[ON_ALERT ERROR] {type(e).__name__}: {e}")

    def _handle_alert(self, violation_type, objects, duration_cs=0):
        # Escalation is based on how long the behaviour has actually
        # been sustained in real time (duration_cs, in centiseconds,
        # tracked by the vision thread), NOT on how many times the
        # alert happened to re-fire — re-firing is gated by a 5s
        # cooldown, so a shorter sustained violation might only ever
        # fire once and would never look escalated if we counted
        # re-fires instead.
        if duration_cs > 0:
            elapsed_seconds = duration_cs / 100.0
            required_seconds = (
                BLOCKED_PERSIST_SECONDS if violation_type == "CAMERA BLOCKED"
                else GAZE_PERSIST_SECONDS
            )
            # How many multiples of the required duration this has now
            # run for: 1x = just crossed it (no escalation yet),
            # 2x = held twice as long, etc.
            multiples  = elapsed_seconds / required_seconds
            escalation = min(
                int(max(0, multiples - 1)) * self.ESCALATION_STEP,
                self.ESCALATION_CAP - 1
            )
        else:
            # One-off detections (objects, unauthorized person) don't
            # track continuous duration — fall back to counting
            # consecutive re-fires of the same type instead.
            if violation_type == self._last_violation_type:
                self._violation_streak += 1
            else:
                self._violation_streak    = 1
                self._last_violation_type = violation_type
            escalation = min(
                (self._violation_streak - 1) * self.ESCALATION_STEP,
                self.ESCALATION_CAP - 1
            )

        base_weight = self.VIOLATION_WEIGHT.get(violation_type, 1)
        weight      = base_weight + escalation

        self.alert_count += weight
        self.alert_label.setText(f"Alerts: {self.alert_count}")
        self.alert_label.setStyleSheet(
            "color: #ef4444; font-size: 12px; font-weight: 700;"
        )

        remaining  = self.MAX_VIOLATIONS - self.alert_count
        detail     = f" ({', '.join(objects)})" if objects else ""
        weight_tag = f"  [+{weight}]" if weight > 1 else ""

        if 0 < remaining <= self.WARNING_REMAINING:
            # Getting close to auto-termination — make it unmissable
            # and keep it on screen longer than a routine flag.
            self.flag_banner.setText(
                f"⚠ FLAGGED — {violation_type}{detail}{weight_tag}   ·   "
                f"WARNING: {remaining} more violation"
                f"{'s' if remaining != 1 else ''} will END your exam"
            )
            self.flag_banner.setStyleSheet("""
                background: #7c2d12;
                color: #fed7aa;
                font-size: 13px;
                font-weight: 700;
                border-bottom: 2px solid #f97316;
            """)
            self.flag_banner.setVisible(True)
            self.flag_banner_timer.start(6000)
        else:
            self.flag_banner.setText(
                f"⚠ FLAGGED — {violation_type}{detail}{weight_tag}"
            )
            self.flag_banner.setStyleSheet("""
                background: #7f1d1d;
                color: #fecaca;
                font-size: 13px;
                font-weight: 700;
                border-bottom: 2px solid #ef4444;
            """)
            self.flag_banner.setVisible(True)
            self.flag_banner_timer.start(4000)

        print(f"[ALERT] #{self.alert_count}/{self.MAX_VIOLATIONS} "
              f"{violation_type} (weight={weight}, "
              f"duration_cs={duration_cs})")

        if self.alert_count >= self.MAX_VIOLATIONS and self.exam_active:
            print(
                f"[EXAM] Violation threshold reached "
                f"({self.alert_count}/{self.MAX_VIOLATIONS}) — "
                f"auto-ending exam"
            )
            self.end_exam("INTEGRITY_VIOLATION")

    def prompt_admin_unlock(self):
        """Called when Ctrl+Shift+Q is detected by global filter."""
        password, ok = QInputDialog.getText(
            self,
            "Admin Unlock",
            "Enter admin password to exit exam:",
            QLineEdit.EchoMode.Password
        )
        if ok and password == ADMIN_PASSWORD:
            self.end_exam("ADMIN_EXIT")
        elif ok:
            QMessageBox.warning(
                self, "Access Denied",
                "Incorrect password. This attempt has been logged."
            )
            # Log failed unlock attempt
            try:
                requests.post(f"{API_URL}/alerts", json={
                    "session_id":     self.student_data.get(
                        "exam_id", "UNKNOWN"
                    ),
                    "candidate_id":   self.student_data[
                        "registration_number"
                    ],
                    "station_id":     "STATION_001",
                    "violation_type": "UNAUTHORIZED EXIT ATTEMPT",
                    "objects":        [],
                    "timestamp":      datetime.now(timezone.utc).isoformat()
                }, timeout=2)
            except Exception:
                pass

    def confirm_submit(self):
        """
        Triggered by the candidate clicking 'Submit Exam'.
        Confirms intent, then ends the exam the same way as
        TIME_UP / ADMIN_EXIT so all cleanup and status updates
        stay consistent.
        """
        reply = QMessageBox.question(
            self,
            "Submit Exam",
            "Are you sure you want to submit your exam?\n\n"
            "This cannot be undone and monitoring will stop.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.submit_btn.setEnabled(False)
            self.submit_btn.setText("Submitting...")
            self.end_exam("SUBMITTED")

    def end_exam(self, reason="COMPLETED"):
        if not self.exam_active:
            return
        self.exam_active = False
        self.exam_timer.stop()
        self.overlay_timer.stop()

        # Remove global key filter
        if self.key_filter:
            QApplication.instance().removeEventFilter(self.key_filter)

        # Unsubscribe from camera
        self.camera_manager.unsubscribe(self.on_camera_frame)

        # Stop vision engine
        if self.vision_thread:
            self.camera_manager.unsubscribe(
                self.vision_thread.receive_frame
            )
            self.vision_thread.stop()

        # Send session end record
        try:
            requests.post(f"{API_URL}/alerts", json={
                "session_id":     self.student_data.get(
                    "exam_id", "UNKNOWN"
                ),
                "candidate_id":   self.student_data["registration_number"],
                "station_id":     "STATION_001",
                "violation_type": f"EXAM_ENDED: {reason}",
                "objects":        [],
                "timestamp":      datetime.now(timezone.utc).isoformat()
            }, timeout=2)
        except Exception:
            pass

        # Update student status
        try:
            final_status = (
                "flagged" if reason == "INTEGRITY_VIOLATION" else "completed"
            )
            requests.patch(
                f"{API_URL}/students/"
                f"{self.student_data['registration_number']}/status",
                json={"status": final_status},
                timeout=2
            )
        except Exception:
            pass

        print(f"[EXAM] Ended — Reason={reason} | "
              f"Alerts={self.alert_count}")

        self.show_end_screen(reason)

    def show_end_screen(self, reason):
        # Restore normal window flags
        self.setWindowFlags(Qt.WindowType.Window)
        self.showNormal()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        if reason == "TIME_UP":
            icon    = "⏱"
            heading = "Exam Time Expired"
            colour  = "#f97316"
            message = "Your exam session has ended. Please remain seated."
        elif reason == "ADMIN_EXIT":
            icon    = "🔓"
            heading = "Exam Ended by Supervisor"
            colour  = "#38bdf8"
            message = "This exam session was ended by the administrator."
        elif reason == "INTEGRITY_VIOLATION":
            icon    = "🚫"
            heading = "Exam Terminated — Integrity Violation"
            colour  = "#ef4444"
            message = (
                f"Your exam was automatically ended after "
                f"{self.alert_count} flagged violations "
                f"(limit: {self.MAX_VIOLATIONS}), such as looking away, "
                f"no face detected, a blocked camera, or unauthorized "
                f"items/persons in view. This session has been marked "
                f"for admin review — please speak to your supervisor."
            )
        else:
            icon    = "✓"
            heading = "Exam Submitted"
            colour  = "#22c55e"
            message = "Your exam has been submitted successfully."

        for widget, style in [
            (QLabel(icon),    f"font-size: 64px;"),
            (QLabel(heading), f"color: {colour}; font-size: 28px; font-weight: 800;"),
            (QLabel(message), "color: #94a3b8; font-size: 15px;"),
            (QLabel(self.student_data["full_name"]),
             "color: #e2e8f0; font-size: 18px; font-weight: 700;"),
            (QLabel(self.student_data["registration_number"]),
             "color: #64748b; font-size: 14px;"),
            (QLabel(f"Total violations detected: {self.alert_count}"),
             "color: #64748b; font-size: 13px;"),
            (QLabel("Please inform your supervisor that you have finished."),
             "color: #334155; font-size: 12px; margin-top: 20px;"),
        ]:
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            widget.setStyleSheet(style)
            widget.setWordWrap(True)
            widget.setFixedWidth(560)
            layout.addWidget(widget)

        countdown_label = QLabel("")
        countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        countdown_label.setStyleSheet(
            "color: #475569; font-size: 12px; margin-top: 10px;"
        )
        layout.addWidget(countdown_label)

        return_btn = QPushButton("Return to Login  ·  Next Candidate")
        return_btn.setFixedHeight(42)
        return_btn.setFixedWidth(300)
        return_btn.setStyleSheet("""
            QPushButton {
                background: #38bdf8;
                color: #0f172a;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
                margin-top: 16px;
            }
            QPushButton:hover {
                background: #0ea5e9;
            }
        """)
        return_btn.clicked.connect(self.return_to_login)
        layout.addWidget(return_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # ── Auto-return countdown ───────────────────────────────────────
        # Regardless of why the exam ended (submitted, timed out, admin
        # exit, or flagged), the station should free itself for the next
        # candidate without requiring a manual click.
        self._auto_return_seconds = 10

        def tick_countdown():
            if self._auto_return_seconds <= 0:
                self.auto_return_timer.stop()
                self.return_to_login()
                return
            countdown_label.setText(
                f"Automatically returning to login in "
                f"{self._auto_return_seconds}s "
                f"(or click the button above)"
            )
            self._auto_return_seconds -= 1

        tick_countdown()
        self.auto_return_timer = QTimer()
        self.auto_return_timer.timeout.connect(tick_countdown)
        self.auto_return_timer.start(1000)

    def _hide_flag_banner(self):
        """
        Called by flag_banner_timer. Guarded because this single-shot
        timer can still be pending in Qt's event queue at the exact
        moment the window closes (return_to_login stops it, but a
        tiny race is possible) — without this guard it can try to
        touch an already-deleted QLabel and crash with
        'wrapped C/C++ object of type QLabel has been deleted'.
        """
        try:
            self.flag_banner.setVisible(False)
        except RuntimeError:
            pass

    def return_to_login(self):
        """
        Closes this candidate's exam window and tells the app
        to bring the login screen back up so the next candidate
        can use the same station. Safe to call more than once
        (e.g. countdown firing right as the button is clicked).
        """
        if getattr(self, "_returned_to_login", False):
            return
        self._returned_to_login = True

        if hasattr(self, "auto_return_timer"):
            self.auto_return_timer.stop()
        if hasattr(self, "flag_banner_timer"):
            # Prevents a pending single-shot timer from firing after
            # this window (and its flag_banner QLabel) is destroyed —
            # that caused a "wrapped C/C++ object has been deleted"
            # crash when the exam ended while a banner was showing.
            self.flag_banner_timer.stop()

        self.exam_ended.emit()
        self.close()

    def closeEvent(self, event):
        if self.exam_active:
            event.ignore()  # Block close during active exam
            return
        if self.vision_thread:
            self.vision_thread.wait(3000)
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# LOGIN WINDOW
# ═════════════════════════════════════════════════════════════════════════════
class LoginWindow(QWidget):
    login_success = pyqtSignal(dict)

    def __init__(self, camera_manager):
        super().__init__()
        self.camera_manager = camera_manager
        self.current_frame  = None
        self.student_data   = None
        self.verify_thread  = None
        self.setup_ui()

        # Subscribe to shared camera for live preview
        self.camera_manager.subscribe(self.on_camera_frame)

        # Update UI from camera frames
        self.cam_ui_timer = QTimer()
        self.cam_ui_timer.timeout.connect(self.update_camera_ui)
        self.cam_ui_timer.start(33)

    def on_camera_frame(self, frame):
        """Receives frames from shared camera manager."""
        self.current_frame = frame

    def update_camera_ui(self):
        """Updates camera preview label on UI thread."""
        if self.current_frame is not None:
            frame = self.current_frame
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w  = rgb.shape[:2]
            img   = QImage(
                rgb.data, w, h,
                w * 3, QImage.Format.Format_RGB888
            )
            pix = QPixmap.fromImage(img).scaled(
                400, 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.camera_label.setPixmap(pix)

    def setup_ui(self):
        self.setWindowTitle("AIProctor — Exam Login")
        self.setFixedSize(940, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: 'Segoe UI';
            }
            QLineEdit {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                color: #e2e8f0;
            }
            QLineEdit:focus { border: 1px solid #38bdf8; }
            QPushButton {
                background: #38bdf8;
                color: #0f172a;
                border: none;
                border-radius: 8px;
                padding: 11px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover    { background: #7dd3fc; }
            QPushButton:disabled {
                background: #1e3a5f;
                color: #334155;
            }
        """)

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ── Left — camera ─────────────────────────────────────────────────
        left = QFrame()
        left.setFixedWidth(440)
        left.setFixedHeight(600)
        left.setStyleSheet("background: #0a1628;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(20, 20, 20, 20)
        ll.setSpacing(10)

        cam_title = QLabel("Live Camera Verification")
        cam_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_title.setFixedHeight(20)
        cam_title.setStyleSheet(
            "color: #64748b; font-size: 12px; font-weight: 600;"
        )
        ll.addWidget(cam_title)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(400, 300)
        self.camera_label.setStyleSheet("""
            border: 2px solid #334155;
            border-radius: 12px;
            background: #1e293b;
        """)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Camera loading...")
        ll.addWidget(self.camera_label, 0, Qt.AlignmentFlag.AlignCenter)

        self.face_status = QLabel("Please log in to begin verification")
        self.face_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_status.setFixedHeight(44)
        self.face_status.setWordWrap(True)
        self.face_status.setStyleSheet("color: #64748b; font-size: 12px;")
        ll.addWidget(self.face_status)

        self.verify_btn = QPushButton("Verify Face & Start Exam")
        self.verify_btn.setFixedWidth(360)
        self.verify_btn.setFixedHeight(48)
        self.verify_btn.setEnabled(False)
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background: #38bdf8; color: #0f172a;
                border-radius: 8px; font-size: 14px;
                font-weight: bold; margin-top: 18px;
            }
            QPushButton:hover    { background: #7dd3fc; }
            QPushButton:disabled { background: #334155; color: #64748b; }
        """)
        self.verify_btn.clicked.connect(self.verify_face)
        ll.addWidget(self.verify_btn, 0, Qt.AlignmentFlag.AlignCenter)

        ll.addStretch()
        main.addWidget(left)

        # ── Right — form ──────────────────────────────────────────────────
        right = QFrame()
        right.setStyleSheet("background: #1e293b;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(40, 40, 40, 40)
        rl.setSpacing(0)
        rl.setAlignment(Qt.AlignmentFlag.AlignTop)

        logo = QLabel("AIProctor")
        logo.setStyleSheet(
            "color: #38bdf8; font-size: 28px; font-weight: 800;"
        )
        rl.addWidget(logo)

        sub = QLabel("Examination Access System")
        sub.setStyleSheet(
            "color: #64748b; font-size: 13px; margin-bottom: 32px;"
        )
        rl.addWidget(sub)

        for label_text, attr, ph, is_pwd in [
            ("REGISTRATION NUMBER", "reg_input",
             "e.g. CSC/2021/001",   False),
            ("PASSWORD",            "pwd_input",
             "Enter your password", True),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("""
                color: #94a3b8; font-size: 11px; font-weight: 600;
                letter-spacing: 1px; margin-bottom: 6px; margin-top: 14px;
            """)
            rl.addWidget(lbl)
            inp = QLineEdit()
            inp.setPlaceholderText(ph)
            if is_pwd:
                inp.setEchoMode(QLineEdit.EchoMode.Password)
                inp.returnPressed.connect(self.login)
            setattr(self, attr, inp)
            rl.addWidget(inp)

        self.login_btn = QPushButton("Login")
        self.login_btn.setFixedHeight(44)
        self.login_btn.clicked.connect(self.login)
        self.login_btn.setStyleSheet("""
            QPushButton {
                background: #38bdf8; color: #0f172a;
                border-radius: 8px; font-size: 14px;
                font-weight: bold; margin-top: 8px;
            }
            QPushButton:hover    { background: #7dd3fc; }
            QPushButton:disabled { background: #334155; color: #64748b; }
        """)
        rl.addWidget(self.login_btn)

        self.msg_label = QLabel("")
        self.msg_label.setWordWrap(True)
        self.msg_label.setFixedHeight(44)
        self.msg_label.setStyleSheet("margin-top: 12px; font-size: 12px;")
        rl.addWidget(self.msg_label)

        self.passport_frame = QFrame()
        self.passport_frame.setVisible(False)
        self.passport_frame.setStyleSheet("""
            background: #0f172a;
            border-radius: 12px;
            padding: 12px;
            margin-top: 16px;
        """)
        pf = QHBoxLayout(self.passport_frame)

        self.passport_img = QLabel()
        self.passport_img.setFixedSize(60, 60)
        self.passport_img.setStyleSheet(
            "border-radius: 30px; border: 2px solid #38bdf8;"
        )
        pf.addWidget(self.passport_img)

        pi = QVBoxLayout()
        self.name_lbl = QLabel()
        self.name_lbl.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #e2e8f0;"
        )
        self.dept_lbl = QLabel()
        self.dept_lbl.setStyleSheet("font-size: 12px; color: #64748b;")
        self.exam_lbl = QLabel()
        self.exam_lbl.setStyleSheet("font-size: 12px; color: #38bdf8;")
        pi.addWidget(self.name_lbl)
        pi.addWidget(self.dept_lbl)
        pi.addWidget(self.exam_lbl)
        pf.addLayout(pi)
        rl.addWidget(self.passport_frame)

        rl.addStretch()

        footer = QLabel("Look directly at the camera after logging in")
        footer.setStyleSheet(
            "color: #334155; font-size: 11px; margin-top: 8px;"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(footer)

        main.addWidget(right)

    def login(self):
        reg_num  = self.reg_input.text().strip()
        password = self.pwd_input.text().strip()

        if not reg_num or not password:
            self.show_msg(
                "Please enter your registration number and password.",
                "error"
            )
            return

        self.login_btn.setEnabled(False)
        self.login_btn.setText("Logging in...")

        try:
            res = requests.post(
                f"{API_URL}/students/login",
                json={
                    "registration_number": reg_num,
                    "password":            password
                },
                timeout=5
            )
            if res.status_code == 200:
                self.student_data = res.json()
                self.on_login_success()
            else:
                self.show_msg(
                    res.json().get("detail", "Login failed."), "error"
                )
                self.login_btn.setEnabled(True)
                self.login_btn.setText("Login")
        except requests.exceptions.ConnectionError:
            self.show_msg(
                "Cannot connect to server. Contact your supervisor.",
                "error"
            )
            self.login_btn.setEnabled(True)
            self.login_btn.setText("Login")

    def on_login_success(self):
        s = self.student_data
        self.name_lbl.setText(s["full_name"])
        self.dept_lbl.setText(s["department"])
        self.exam_lbl.setText(
            f"Exam: {s.get('exam_id') or 'Pending'}"
        )

        if s.get("passport_photo"):
            photo_bytes = base64.b64decode(s["passport_photo"])
            photo_arr   = np.frombuffer(photo_bytes, dtype=np.uint8)
            photo_img   = cv2.imdecode(photo_arr, cv2.IMREAD_COLOR)
            photo_rgb   = cv2.cvtColor(photo_img, cv2.COLOR_BGR2RGB)
            h, w        = photo_rgb.shape[:2]
            qimg        = QImage(
                photo_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
            )
            pix = QPixmap.fromImage(qimg).scaled(
                60, 60,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.passport_img.setPixmap(pix)

        self.passport_frame.setVisible(True)
        self.login_btn.setEnabled(False)
        self.reg_input.setEnabled(False)
        self.pwd_input.setEnabled(False)
        self.verify_btn.setEnabled(True)
        self.show_msg(
            f"Welcome {s['full_name']}. Look at camera and click Verify.",
            "success"
        )
        self.face_status.setText(
            "Look directly at the camera then click Verify Face"
        )
        self.face_status.setStyleSheet("color: #38bdf8; font-size: 12px;")

    def verify_face(self):
        if self.current_frame is None:
            self.show_msg("Camera not ready.", "error")
            return
        if not self.student_data.get("passport_photo"):
            self.show_msg(
                "No passport photo on file. Contact supervisor.", "error"
            )
            return

        self.verify_btn.setEnabled(False)
        self.verify_btn.setText("Verifying...")
        self.face_status.setText("Comparing face to passport photo...")
        self.face_status.setStyleSheet(
            "color: #eab308; font-size: 12px;"
        )

        self.verify_thread = FaceVerifyThread(
            self.current_frame.copy(),
            self.student_data["passport_photo"]
        )
        self.verify_thread.result_signal.connect(self.on_verify_result)
        self.verify_thread.start()

    def on_verify_result(self, success, message):
        if success:
            self.face_status.setText(f"✓ {message}")
            self.face_status.setStyleSheet(
                "color: #22c55e; font-size: 12px;"
            )
            self.show_msg(
                "Identity confirmed. Starting monitoring...", "success"
            )
            self.login_success.emit(self.student_data)
        else:
            self.face_status.setText(f"✗ {message}")
            self.face_status.setStyleSheet(
                "color: #ef4444; font-size: 12px;"
            )
            self.show_msg(message, "error")
            self.verify_btn.setEnabled(True)
            self.verify_btn.setText("Verify Face & Start Exam")

            try:
                requests.post(f"{API_URL}/alerts", json={
                    "session_id":     self.student_data.get(
                        "exam_id", "UNKNOWN"
                    ),
                    "candidate_id":   self.student_data.get(
                        "registration_number", "UNKNOWN"
                    ),
                    "station_id":     "STATION_001",
                    "violation_type": "IMPERSONATION ATTEMPT AT LOGIN",
                    "objects":        [],
                    "timestamp":      datetime.now(timezone.utc).isoformat()
                }, timeout=2)
            except Exception:
                pass

    def show_msg(self, text, msg_type):
        colours = {
            "error":   "#fca5a5",
            "success": "#6ee7b7",
            "info":    "#7dd3fc"
        }
        self.msg_label.setText(text)
        self.msg_label.setStyleSheet(
            f"color: {colours.get(msg_type, '#e2e8f0')}; font-size: 12px;"
        )

    def closeEvent(self, event):
        self.cam_ui_timer.stop()
        self.camera_manager.unsubscribe(self.on_camera_frame)
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class AIProctorApp:

    def __init__(self):
        self.app            = QApplication(sys.argv)
        self.app.setApplicationName("AIProctor")
        self.camera_manager = CameraManager()
        self.login_window   = None
        self.loading_screen = None
        self.exam_window    = None
        self.vision_thread  = None

    def run(self):
        # Start shared camera immediately
        self.camera_manager.start()

        # Start preloading vision models in background
        self.vision_thread = VisionEngineThread(
            candidate_id="PRELOAD",
            session_id="PRELOAD",
            station_id="PRELOAD"
        )
        threading.Thread(
            target=self.vision_thread.preload_models,
            daemon=True
        ).start()

        # Show login window
        self.login_window = LoginWindow(self.camera_manager)
        self.login_window.login_success.connect(self.on_login_success)
        self.login_window.show()

        sys.exit(self.app.exec())

    def on_login_success(self, student_data):
        try:
            self._handle_login_success(student_data)
        except Exception as e:
            logger.error(
                f"[ON_LOGIN_SUCCESS ERROR] {type(e).__name__}: {e}\n"
                + traceback.format_exc()
            )
            print(f"[ON_LOGIN_SUCCESS ERROR] {type(e).__name__}: {e}")

    def _handle_login_success(self, student_data):
        # Close login window
        self.login_window.cam_ui_timer.stop()
        self.camera_manager.unsubscribe(
            self.login_window.on_camera_frame
        )
        self.login_window.close()

        # Update vision thread with real student credentials
        self.vision_thread.candidate_id = student_data["registration_number"]
        self.vision_thread.session_id   = (
            student_data.get("exam_id") or "SESSION_UNKNOWN"
        )
        self.vision_thread.station_id   = "STATION_001"

        # Show loading screen
        self.loading_screen = LoadingScreen(
            self.vision_thread, student_data, self.camera_manager
        )
        self.loading_screen.ready.connect(
            lambda: self.launch_exam(student_data)
        )
        self.loading_screen.show()

    def launch_exam(self, student_data):
        try:
            self.loading_screen.close()
            self.exam_window = ExamWindow(
                student_data, self.vision_thread, self.camera_manager
            )
            self.exam_window.exam_ended.connect(self.on_exam_ended)
            self.exam_window.show()
            print(
                f"[APP] Exam launched for "
                f"{student_data['registration_number']}"
            )
        except Exception as e:
            logger.error(
                f"[LAUNCH_EXAM ERROR] {type(e).__name__}: {e}\n"
                + traceback.format_exc()
            )
            print(f"[LAUNCH_EXAM ERROR] {type(e).__name__}: {e}")

    def on_exam_ended(self):
        """
        Fired when the exam window's 'Return to Login' button is
        clicked (or its auto-return countdown finishes) after a
        submit/time-up/admin-exit/flagged end. Frees the station for
        a new candidate without restarting the app: preloads a fresh
        vision engine, then shows the login screen.
        """
        try:
            print(
                "[APP] Station freed — returning to login "
                "for next candidate"
            )

            self.exam_window   = None
            self.vision_thread = VisionEngineThread(
                candidate_id="PRELOAD",
                session_id="PRELOAD",
                station_id="PRELOAD"
            )
            threading.Thread(
                target=self.vision_thread.preload_models,
                daemon=True
            ).start()

            self.login_window = LoginWindow(self.camera_manager)
            self.login_window.login_success.connect(self.on_login_success)
            self.login_window.show()
        except Exception as e:
            logger.error(
                f"[ON_EXAM_ENDED ERROR] {type(e).__name__}: {e}\n"
                + traceback.format_exc()
            )
            print(f"[ON_EXAM_ENDED ERROR] {type(e).__name__}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "PyQt6-WebEngine"
        ])

    app = AIProctorApp()
    app.run()