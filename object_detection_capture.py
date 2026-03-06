import cv2
import numpy as np
import time
import os

class ObjectDetector:
    def __init__(
        self,
        net,
        output_size=(640, 480),
        conf_threshold=0.5,
        nms_threshold=0.3,
        save_dir="detected_images",
        save_interval=10
    ):
        self.net = net
        self.output_size = output_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Lưu ảnh
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.frame_count = 0

        os.makedirs(self.save_dir, exist_ok=True)

    def detect_objects(self, frame):
        start_time = time.time()
        self.frame_count += 1

        frame_resized = cv2.resize(frame, self.output_size)
        height, width = frame_resized.shape[:2]

        # Vẽ các điểm góc
        tl = (250, 350)
        tr = (390, 350)
        bl = (80, 472)
        br = (560, 472)
        for pt in [tl, tr, bl, br]:
            cv2.circle(frame_resized, pt, 5, (0, 0, 255), -1)

        # Blob
        blob = cv2.dnn.blobFromImage(
            frame_resized, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = max(0, center_x - w // 2)
                    y = max(0, center_y - h // 2)
                    w = min(w, width - x)
                    h = min(h, height - y)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS
        indexes = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        final_boxes = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                final_boxes.append([x, y, w, h])

                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    "object",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(
            frame_resized,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # ===== LƯU ẢNH MỖI 10 FRAME =====
        if self.frame_count % self.save_interval == 0 and len(final_boxes) > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detect_{timestamp}_frame{self.frame_count}.jpg"
            save_path = os.path.join(self.save_dir, filename)
            cv2.imwrite(save_path, frame_resized)
            print(f"[INFO] Saved image: {save_path}")

        self.last_boxes = final_boxes
        return frame_resized, fps, final_boxes

weights = "H:/PBL5/yolov3_tiny/yolov3-tiny.weights"
cfg = "H:/PBL5/yolov3_tiny/yolov3-tiny.cfg"
    # Load YOLO
net = cv2.dnn.readNet(weights, cfg)

    # Tối ưu OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # Nếu có CUDA:
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

detector = ObjectDetector(net)

cap = cv2.VideoCapture("VID/VN_4.mp4")  # camera

fps_avg = 0
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps = detector.detect_objects(frame)

cap.release()
cv2.destroyAllWindows()