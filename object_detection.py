import cv2
import numpy as np
import time

class ObjectDetector:
    def __init__(self, net, output_size=(640, 480), conf_threshold=0.5, nms_threshold=1):
        self.net = net
        self.output_size = output_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        """
        Nhận frame, trả về frame đã vẽ bbox và FPS
        """
        start_time = time.time()
        frame_resized = cv2.resize(frame, self.output_size)
        height, width = frame_resized.shape[:2]

        # Vẽ các điểm góc
        tl = (250, 350)
        tr = (390, 350)
        bl = (80, 472)
        br = (560, 472)
        for pt in [tl, tr, bl, br]:
            cv2.circle(frame_resized, pt, 5, (0, 0, 255), -1)

        # Tạo blob và forward
        blob = cv2.dnn.blobFromImage(frame_resized, 1/255.0, (416, 416), swapRB=True, crop=False)
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

        # Non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        final_boxes = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                final_boxes.append([x, y, w, h])
                label = "object"
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_resized, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.last_boxes = final_boxes
        return frame_resized, fps, final_boxes
