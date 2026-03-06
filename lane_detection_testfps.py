import cv2
import numpy as np
import time


class LaneDetectorFPSOnly:
    def __init__(self):
        # Cố định HSV (không dùng trackbar)
        self.lower = np.array([0, 0, 113])
        self.upper = np.array([255, 43, 255])

        self.prevLx, self.prevRx = [], []

        # Perspective points
        self.tl, self.tr = (250, 350), (390, 350)
        self.bl, self.br = (80, 472), (560, 472)

        self.pts1 = np.float32([self.tl, self.bl, self.tr, self.br])
        self.pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        self.matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)

    def process_frame(self, frame):
        start_time = time.time()

        # Perspective transform
        warped = cv2.warpPerspective(frame, self.matrix, (640, 480))

        # HSV threshold
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # Histogram
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        # Sliding window
        y = 472
        lx, rx = [], []
        left_temp, right_temp = left_base, right_base

        while y > 0:
            # Left
            roi = mask[y - 40:y, left_temp - 50:left_temp + 50]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    left_temp = left_temp - 50 + cx
            lx.append(left_temp)

            # Right
            roi = mask[y - 40:y, right_temp - 50:right_temp + 50]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    right_temp = right_temp - 50 + cx
            rx.append(right_temp)

            y -= 40

        # Fallback
        lx = lx if lx else self.prevLx
        rx = rx if rx else self.prevRx
        self.prevLx, self.prevRx = lx, rx

        # Fit lane (chỉ để giữ pipeline, không vẽ)
        if lx and rx:
            min_len = min(len(lx), len(rx))
            y_vals = [472 - i * 40 for i in range(min_len)]
            np.polyfit(y_vals, lx[:min_len], 2)
            np.polyfit(y_vals, rx[:min_len], 2)

        fps = 1.0 / (time.time() - start_time)
        return fps

import cv2

detector = LaneDetectorFPSOnly()

cap = cv2.VideoCapture("VID/VN_1.mp4")
fps_sum = 0
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps = detector.process_frame(frame)

    fps_sum += fps
    count += 1

    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == 27:
        break

print(f"\nFPS trung bình: {fps_sum / max(count,1):.2f}")

cap.release()
cv2.destroyAllWindows()
