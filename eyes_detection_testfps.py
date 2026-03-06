import cv2
import mediapipe as mp
import time
import math


class EyeSleepDetector:
    def __init__(self, ear_threshold=0.25):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        self.EYE_AR_THRESH = ear_threshold
        self.eye_closed_start = None

    def _eye_aspect_ratio(self, lm, idx, w, h):
        p = [(lm[i].x * w, lm[i].y * h) for i in idx]
        A = math.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
        B = math.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
        C = math.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
        return (A + B) / (2.0 * C)

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self.eye_closed_start = None
            return "no_face", 0.0

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        leftEAR = self._eye_aspect_ratio(lm, self.LEFT_EYE_IDX, w, h)
        rightEAR = self._eye_aspect_ratio(lm, self.RIGHT_EYE_IDX, w, h)

        if leftEAR < self.EYE_AR_THRESH and rightEAR < self.EYE_AR_THRESH:
            if self.eye_closed_start is None:
                self.eye_closed_start = time.time()
            return "closed", time.time() - self.eye_closed_start
        else:
            self.eye_closed_start = None
            return "open", 0.0

detector = EyeSleepDetector(ear_threshold=0.25)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_time = time.time()
FRAME_SKIP = 6
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % FRAME_SKIP == 0:
        state, closed_time = detector.detect(frame)
    count += 1

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    print(f"FPS: {fps:.2f} | State: {state} | Closed: {closed_time:.2f}s")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
