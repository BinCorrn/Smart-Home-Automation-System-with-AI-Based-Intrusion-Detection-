import cv2
import numpy as np
import time
import os
def nothing(x):
    pass

class LaneDetector:
    def __init__(self):
        # --- Trackbar để điều chỉnh HSV threshold ---
        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L - V", "Trackbars", 113, 255, nothing)
        cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U - S", "Trackbars", 43, 255, nothing)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

        self.prevLx, self.prevRx = [], []

    def process_frame(self, frame):
        start_time = time.time()
        # --- Điểm góc cho perspective transform ---
        tl, tr = (250, 350), (390, 350)
        bl, br = (80, 472), (560, 472)
        for pt in [tl, tr, bl, br]:
            cv2.circle(frame, pt, 5, (0,0,255), -1)

        # --- Perspective transform ---
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

        # --- HSV thresholding ---
        hsv_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_frame, lower, upper)

        # --- Tự động điều chỉnh L-V ---
        white_ratio = cv2.countNonZero(mask) / mask.size
        adjust_speed = 2
        l_v = min(255, l_v + adjust_speed) if white_ratio > 0.10 else max(0, l_v - adjust_speed) if white_ratio < 0.05 else l_v
        cv2.setTrackbarPos("L - V", "Trackbars", int(l_v))

        # --- Histogram để xác định lane bases ---
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
        midpoint = histogram.shape[0]//2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        # --- Sliding window ---
        y = 472
        lx, rx = [], []
        msk = mask.copy()
        left_temp, right_temp = left_base, right_base

        while y > 0:
            # Left lane
            img = mask[y-40:y, left_temp-50:left_temp+50]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found = False
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    lx.append(left_temp-50+cx)
                    left_temp = left_temp-50+cx
                    found = True
                    break
            if not found:
                lx.append(left_temp)

            # Right lane
            img = mask[y-40:y, right_temp-50:right_temp+50]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found = False
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    rx.append(right_temp-50+cx)
                    right_temp = right_temp-50+cx
                    found = True
                    break
            if not found:
                rx.append(right_temp)

            y -= 40

        # --- Xử lý khi không tìm thấy ---
        lx = lx if len(lx)>0 else self.prevLx if self.prevLx else [left_base]*12
        rx = rx if len(rx)>0 else self.prevRx if self.prevRx else [right_base]*12
        self.prevLx, self.prevRx = lx, rx

        # --- Fit lane & tính offset, angle ---
        min_len = min(len(lx), len(rx))
        y_values = [472 - i*40 for i in range(min_len)]
        left_points = [(lx[i], y_values[i]) for i in range(min_len)]
        right_points = [(rx[i], y_values[i]) for i in range(min_len)]

        try:
            left_fit = np.polyfit([p[1] for p in left_points], [p[0] for p in left_points], 2)
            right_fit = np.polyfit([p[1] for p in right_points], [p[0] for p in right_points], 2)
            y_eval = 480
            left_curv = ((1+(2*left_fit[0]*y_eval+left_fit[1])**2)**1.5)/abs(2*left_fit[0])
            right_curv = ((1+(2*right_fit[0]*y_eval+right_fit[1])**2)**1.5)/abs(2*right_fit[0])
            curvature = (left_curv+right_curv)/2
            lane_center = (left_base + right_base)/2
            car_pos = 320
            lane_offset = (car_pos-lane_center)*3.7/640
            steering_angle = np.degrees(np.arctan(lane_offset/curvature))

            # Vẽ lane & overlay
            top_left = (lx[0], 472)
            bottom_left = (lx[min_len-1], y_values[min_len-1])
            top_right = (rx[0], 472)
            bottom_right = (rx[min_len-1], y_values[min_len-1])
            quad = np.array([top_left,bottom_left,bottom_right,top_right], dtype=np.int32).reshape((-1,1,2))
            overlay = transformed_frame.copy()
            cv2.fillPoly(overlay,[quad],(0,255,0))
            cv2.addWeighted(overlay,0.2,transformed_frame,0.8,0,transformed_frame)

            inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            lane_img = cv2.warpPerspective(transformed_frame, inv_matrix, (640,480))
            result = cv2.addWeighted(frame,1,lane_img,0.5,0)

            # Draw center line
            end_x = int(320 + 100*np.sin(np.radians(steering_angle)))
            end_y = int(480 - 100*np.cos(np.radians(steering_angle)))
            cv2.line(result,(320,480),(end_x,end_y),(255,0,0),2)

            cv2.putText(result,f'Offset: {lane_offset:.2f} m',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            if abs(lane_offset)>0.7:
                cv2.putText(result,'WARNING',(30,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.putText(result,f'Angle: {steering_angle:.2f} deg',(30,110),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(result,'Lane detected',(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        except:
            result = frame.copy()
            cv2.putText(result,'Lane detection error',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # --- Hiển thị mask ---
        mask_display = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        # --- Combine views ---
        top_row = np.hstack((cv2.resize(frame,(320,240)), cv2.resize(transformed_frame,(320,240))))
        bottom_row = np.hstack((cv2.resize(mask_display,(320,240)), cv2.resize(result,(320,240))))
        combined = np.vstack((top_row,bottom_row))

        fps = 1.0/(time.time()-start_time)
        cv2.putText(combined,f"FPS: {fps:.2f}",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        return combined

cap = cv2.VideoCapture("VID/VN_3.mp4")
detector = LaneDetector()
save_dir = "lane_capture"
os.makedirs(save_dir, exist_ok=True)
i = 1980
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    fps = detector.process_frame(frame)

    cv2.imshow("Lane FPS Only", fps)
    if i % 20 == 0:
        filename = f"cap_{i}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), fps)
        print(f"[INFO] Saved {filename}")
    i=i+1
    if cv2.waitKey(1) & 0xFF == 27:
        break

