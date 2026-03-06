import cv2
import mediapipe as mp
import time
import math


class EyeSleepDetector:
    def __init__(self, ear_threshold=0.1):
        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        # Landmark mắt
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        # Ngưỡng EAR
        self.EYE_AR_THRESH = ear_threshold

        # Biến theo dõi thời gian mắt nhắm
        self.eye_closed_start = None

    def _eye_aspect_ratio(self, landmarks, eye_idx, frame_w, frame_h):
        coords = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_idx]
        A = math.hypot(coords[1][0] - coords[5][0], coords[1][1] - coords[5][1])
        B = math.hypot(coords[2][0] - coords[4][0], coords[2][1] - coords[4][1])
        C = math.hypot(coords[0][0] - coords[3][0], coords[0][1] - coords[3][1])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_sleepiness(self, frame):
        eye_closed_duration = 0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        frame_h, frame_w = frame.shape[:2]

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # ===== DẪN LANDMARK MẮT =====
            for idx in self.LEFT_EYE_IDX:
                x = int(face_landmarks.landmark[idx].x * frame_w)
                y = int(face_landmarks.landmark[idx].y * frame_h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in self.RIGHT_EYE_IDX:
                x = int(face_landmarks.landmark[idx].x * frame_w)
                y = int(face_landmarks.landmark[idx].y * frame_h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # ===== TÍNH EAR =====
            leftEAR = self._eye_aspect_ratio(face_landmarks.landmark, self.LEFT_EYE_IDX, frame_w, frame_h)
            rightEAR = self._eye_aspect_ratio(face_landmarks.landmark, self.RIGHT_EYE_IDX, frame_w, frame_h)

            left_closed = leftEAR < self.EYE_AR_THRESH
            right_closed = rightEAR < self.EYE_AR_THRESH

            # Nếu cả 2 mắt nhắm
            if left_closed and right_closed:
                if self.eye_closed_start is None:
                    self.eye_closed_start = time.time()
                eye_closed_duration = time.time() - self.eye_closed_start
                return "closed", eye_closed_duration

            # Nếu mở (hoặc chỉ nhắm 1 mắt)
            else:
                self.eye_closed_start = None
                return "open", 0

        else:
            self.eye_closed_start = None
            return "no_face", 0

