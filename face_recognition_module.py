import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
REGISTERED_FACES_DIR = "registered_faces"
ENCODINGS_FILE       = "face_encodings.pkl"
MATCH_TOLERANCE      = 0.6   # lower = stricter matching
IMPERSONATION_FRAMES = 20    # frames before impersonation alert fires


# ═════════════════════════════════════════════════════════════════════════════
class FaceRecognitionModule:
    """
    Handles candidate face registration and real-time identity verification.

    Workflow:
      1. Register candidate face before exam starts
      2. During exam, verify face every frame
      3. Fire impersonation alert if face does not match or disappears
    """

    def __init__(self):
        self.known_encodings  = {}   # {candidate_id: face_encoding}
        self.no_match_counter = 0    # persistence counter for impersonation
        self.verified_label   = "NOT REGISTERED"
        self.is_registered    = False

        # Create directory for storing registered face images
        os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

        # Load any previously saved encodings
        self._load_encodings()
        print("DEBUG: FaceRecognitionModule initialized")


    # ── Registration ──────────────────────────────────────────────────────────
    def register_candidate(self, frame, candidate_id):
        """
        Registers a candidate's face from a captured frame.
        Saves the face encoding to disk for persistence.

        Returns: (success, message)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations in frame
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            return False, "No face detected. Please look directly at the camera."

        if len(face_locations) > 1:
            return False, "Multiple faces detected. Only one person should be in frame."

        # Extract face encoding
        encoding = face_recognition.face_encodings(
            rgb_frame, face_locations
        )[0]

        # Save encoding in memory
        self.known_encodings[candidate_id] = encoding
        self.is_registered = True
        self.verified_label = "VERIFIED"

        # Save face image to disk
        face_image_path = os.path.join(
            REGISTERED_FACES_DIR, f"{candidate_id}.jpg"
        )
        cv2.imwrite(face_image_path, frame)

        # Save all encodings to disk
        self._save_encodings()

        print(f"[REGISTERED] Candidate {candidate_id} registered successfully")
        return True, f"Candidate {candidate_id} registered successfully."


    # ── Verification ──────────────────────────────────────────────────────────
    def verify_face(self, frame, candidate_id):
        """
        Verifies the face in the current frame against the registered candidate.

        Returns:
          status  : 'VERIFIED' | 'IMPERSONATION' | 'NO FACE' | 'NOT REGISTERED'
          alert   : True if violation should be flagged
          message : Human readable description
        """
        if not self.is_registered or candidate_id not in self.known_encodings:
            return "NOT REGISTERED", False, "Candidate not registered"

        rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # No face in frame
        if len(face_locations) == 0:
            self.no_match_counter += 1
            if self.no_match_counter >= IMPERSONATION_FRAMES:
                return "NO FACE", True, "Candidate face not visible"
            return "NO FACE", False, "Face temporarily not visible"

        # Get encoding of detected face
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations
        )

        if not face_encodings:
            return "NO FACE", False, "Could not encode face"

        detected_encoding = face_encodings[0]

        # Compare with registered candidate
        known_encoding = self.known_encodings[candidate_id]
        distance       = face_recognition.face_distance(
            [known_encoding], detected_encoding
        )[0]
        match          = distance <= MATCH_TOLERANCE

        if match:
            # Reset counter on successful match
            self.no_match_counter = 0
            confidence = round((1 - distance) * 100, 1)
            return "VERIFIED", False, f"Identity confirmed ({confidence}%)"
        else:
            # Increment counter for persistent mismatch
            self.no_match_counter += 1
            if self.no_match_counter >= IMPERSONATION_FRAMES:
                return "IMPERSONATION", True, "Face does not match registered candidate"
            return "IMPERSONATION", False, "Face mismatch detected"


    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw_verification_box(self, frame, status, message):
        """
        Draws the identity verification status on the frame.
        Green = verified, Red = impersonation/no face
        """
        colours = {
            "VERIFIED":       (0, 255, 0),
            "IMPERSONATION":  (0, 0, 255),
            "NO FACE":        (0, 165, 255),
            "NOT REGISTERED": (128, 128, 128),
        }
        colour = colours.get(status, (255, 255, 255))

        frame_h, frame_w = frame.shape[:2]

        # Draw status bar at bottom right
        cv2.rectangle(frame,
                      (frame_w - 320, frame_h - 60),
                      (frame_w, frame_h),
                      (0, 0, 0), -1)
        cv2.putText(frame, f"ID: {status}",
                    (frame_w - 310, frame_h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        cv2.putText(frame, message,
                    (frame_w - 310, frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


    # ── Persistence ───────────────────────────────────────────────────────────
    def _save_encodings(self):
        """Saves face encodings to disk so registration persists across sessions."""
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(self.known_encodings, f)
        print(f"[SAVED] Face encodings saved to {ENCODINGS_FILE}")


    def _load_encodings(self):
        """Loads previously saved face encodings from disk."""
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                self.known_encodings = pickle.load(f)
            if self.known_encodings:
                self.is_registered = True
                self.verified_label = "VERIFIED"
                print(f"[LOADED] {len(self.known_encodings)} face encoding(s) loaded")
        else:
            print("[INFO] No saved encodings found. Please register candidate.")


    def clear_registrations(self):
        """Clears all registered faces. Use between exam sessions."""
        self.known_encodings  = {}
        self.is_registered    = False
        self.verified_label   = "NOT REGISTERED"
        self.no_match_counter = 0
        if os.path.exists(ENCODINGS_FILE):
            os.remove(ENCODINGS_FILE)
        print("[CLEARED] All face registrations removed")


# ═════════════════════════════════════════════════════════════════════════════
# Standalone test — run this file directly to test face registration
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    module     = FaceRecognitionModule()
    cap        = cv2.VideoCapture(0)
    registered = False

    print("=" * 55)
    print("  Face Recognition Module Test")
    print("  Press R to register your face")
    print("  Press C to clear all registrations")
    print("  Press Q to quit")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        if not registered:
            # Show registration prompt
            cv2.putText(frame, "Press R to register your face",
                        (10, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # Verify face every frame
            status, alert, message = module.verify_face(frame, "CANDIDATE_001")
            module.draw_verification_box(frame, status, message)

            if alert:
                print(f"[ALERT] {status}: {message}")

        cv2.imshow("Face Recognition Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') or key == ord('R'):
            ret, reg_frame = cap.read()
            success, msg   = module.register_candidate(reg_frame, "CANDIDATE_001")
            print(f"Registration: {msg}")
            if success:
                registered = True

        elif key == ord('c') or key == ord('C'):
            module.clear_registrations()
            registered = False

        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()