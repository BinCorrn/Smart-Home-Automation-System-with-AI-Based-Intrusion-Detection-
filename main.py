from multiprocessing import Process, Queue, set_start_method
import cv2
import numpy as np
from eyes_detection import EyeSleepDetector  # import class bạn đã viết
from object_detection import ObjectDetector
from lane_detection import LaneDetector
import winsound

def camera_reader_video(video_path, queue1,queue2):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        if not ret:
            queue1.put(None)
            queue2.put(None)
            break
        if not queue1.full() and not queue2.full():
            queue1.put(frame)
            queue2.put(frame)
    cap.release()

def camera_reader(video_path, queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    while True:
        ret, frame = cap.read()
        if not ret:
            queue.put(None)  # báo hiệu hết video
            break
        if not queue.full():
            queue.put(frame)
    cap.release()

def run_object_detection(queue):

    weights = "H:/PBL5/yolov3_tiny/yolov3-tiny.weights"
    cfg = "H:/PBL5/yolov3_tiny/yolov3-tiny.cfg"

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    detector = ObjectDetector(net)

    # Hình thang
    tl = (250, 350)
    tr = (390, 350)
    bl = (80, 472)
    br = (560, 472)
    trap = np.array([tl, tr, br, bl], dtype=np.int32)

    while True:
        frame = queue.get()
        if frame is None:
            break

        frame = cv2.resize(frame, (640, 480))

        # Detect
        output, fps, boxes = detector.detect_objects(frame)

        # Vẽ hình thang
        cv2.polylines(output, [trap], True, (255,0,0), 2)

        # Kiểm tra object trong vùng
        for (x, y, w, h) in boxes:
            cx = x + w//2
            cy = y + h//2

            # vẽ tâm object
            cv2.circle(output, (cx,cy), 5, (0,255,255), -1)

            inside = cv2.pointPolygonTest(trap, (cx,cy), False)
            if inside >= 0:
                cv2.putText(output, "CANH BAO: Object trong vung!", (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Object Detection", output)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()



def run_lane_detection(queue):
    detector = LaneDetector()  # Tạo object xử lý lane
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)

    while True:
        frame = queue.get()
        if frame is None:
            print("[INFO] Lane detection stopped.")
            break

        frame = cv2.resize(frame, (640, 480))
        lane_frame = detector.process_frame(frame)

        cv2.imshow("Lane Detection", lane_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()

def run_sleep_detection(queue):
    detector = EyeSleepDetector()
    while True:
        frame = queue.get()
        if frame is None:
            break
        eye_state, duration = detector.detect_sleepiness(frame)
        print(f"Eye state: {eye_state}, duration: {duration:.2f}s")
        # Có thể hiển thị frame nếu muốn
        cv2.putText(frame, f"{eye_state}: {duration:.2f}s", (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if eye_state=="closed" else (0,255,0), 2)
        cv2.imshow("Sleep Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# --- Gọi hàm ---
if __name__ == "__main__":
    set_start_method("spawn")  # Bắt buộc cho Windows
    frame_queue = Queue(maxsize=1)
    frame_queue2 = Queue(maxsize=1)
    frame_queue1 = Queue(maxsize=1)

    p1 = Process(target=camera_reader, args=("0", frame_queue))
    p2 = Process(target=run_sleep_detection, args=(frame_queue,))
    p3 = Process(target=camera_reader_video, args=("VID/VN_4.mp4",frame_queue2, frame_queue1))
    p4 = Process(target=run_object_detection, args=(frame_queue2,))
    p5 = Process(target=run_lane_detection, args=(frame_queue1,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()