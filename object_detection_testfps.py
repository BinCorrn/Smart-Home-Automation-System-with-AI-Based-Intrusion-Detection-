import cv2
import numpy as np
import time

class ObjectDetector:
    def __init__(self, net, input_size=(416, 416), conf_threshold=0.5, nms_threshold=0.3):
        self.net = net
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_fps(self, frame):
        """
        Nhận frame, chỉ trả về FPS
        """
        start_time = time.time()
        height, width = frame.shape[:2]

        # Tạo blob
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                confidence = scores[np.argmax(scores)]

                if confidence > self.conf_threshold:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = cx - w // 2
                    y = cy - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # NMS (không vẽ)
        if boxes:
            cv2.dnn.NMSBoxes(boxes, confidences,
                             self.conf_threshold,
                             self.nms_threshold)

        fps = 1.0 / (time.time() - start_time)
        return fps

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

cap = cv2.VideoCapture("VID/VN_1.mp4")  # camera

fps_avg = 0
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps = detector.detect_fps(frame)

    fps_avg += fps
    count += 1

    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

print("FPS trung bình:", fps_avg / max(count, 1))

cap.release()
cv2.destroyAllWindows()

