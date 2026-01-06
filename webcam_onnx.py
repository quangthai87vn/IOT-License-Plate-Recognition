# webcam_onnx.py
import os
import time
import cv2
import numpy as np

# ====== CONFIG ======
ONNX_PATH = os.environ.get("LPR_ONNX", "model/LP_detector_nano_61.onnx")
IMG_SIZE  = int(os.environ.get("IMG_SIZE", "640"))
CONF_THRES = float(os.environ.get("CONF", "0.25"))
IOU_THRES  = float(os.environ.get("IOU", "0.45"))
SHOW = os.environ.get("SHOW", "1") == "1"   # SHOW=0 nếu SSH/headless

# CSI gstreamer pipeline (IMX219)
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280, capture_height=720,
    display_width=1280, display_height=720,
    framerate=30, flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
       # f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
         f"format=(string)NV12, framerate=(fraction)30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )

'''
pipeline = (
"nvarguscamerasrc ! "
"video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
"nvvidconv flip-method=0 ! "
"video/x-raw, format=BGRx ! videoconvert ! "
"video/x-raw, format=BGR ! "
"appsink max-buffers=1 drop=1 sync=false"
)

'''

def letterbox(im, new_shape=640, color=(114,114,114)):
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))  # w,h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (dw, dh)

def nms_xyxy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

class YOLOv5ONNX:
    def __init__(self, onnx_path):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        # thử bật CUDA nếu OpenCV build có hỗ trợ (đa số Jetson apt opencv: thường KHÔNG có DNN CUDA)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        except Exception:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, frame_bgr):
        img, r, (dw, dh) = letterbox(frame_bgr, IMG_SIZE)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1,25200,6) kiểu YOLOv5 export

        pred = out[0]  # (25200, 6) nếu 1 class; nếu nhiều class sẽ khác
        # YOLOv5 ONNX thường: [x,y,w,h,conf,cls] (cls là index hoặc score tuỳ export)
        # Với model 1 class: cột 5 thường là class_id (0). Conf ở cột 4.
        conf = pred[:, 4]
        mask = conf >= CONF_THRES
        pred = pred[mask]
        conf = conf[mask]
        if pred.shape[0] == 0:
            return np.empty((0,4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

        xywh = pred[:, 0:4]
        # xywh -> xyxy trên ảnh letterbox
        x = xywh[:, 0]; y = xywh[:, 1]; w = xywh[:, 2]; h = xywh[:, 3]
        x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
        boxes = np.stack([x1,y1,x2,y2], axis=1)

        keep = nms_xyxy(boxes, conf, IOU_THRES)
        boxes = boxes[keep]
        conf  = conf[keep]

        # scale ngược về frame gốc
        boxes[:, [0,2]] -= dw
        boxes[:, [1,3]] -= dh
        boxes /= r
        boxes = boxes.clip(min=0)

        cls = np.zeros((boxes.shape[0],), dtype=np.int32)
        return boxes, conf, cls

def main():
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"Không thấy ONNX: {ONNX_PATH}")

    model = YOLOv5ONNX(ONNX_PATH)

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Không mở được CSI camera (check /tmp/argus_socket mount + privileged)")

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Không đọc được frame")
            break

        boxes, confs, _ = model.infer(frame)

        for (x1,y1,x2,y2), c in zip(boxes.astype(int), confs):
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{c:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        frames += 1
        if frames % 30 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            print(f"FPS ~ {fps:.1f}, det={len(boxes)}")

        if SHOW:
            cv2.imshow("CSI - ONNX", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
