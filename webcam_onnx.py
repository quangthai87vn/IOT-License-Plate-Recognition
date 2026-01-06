import os
import time
import numpy as np
import cv2

SRC = os.getenv("SRC", "csi")                 # csi | rtsp
RTSP_URL = os.getenv("RTSP_URL", "")
RTSP_CODEC = os.getenv("RTSP_CODEC", "h264")
RTSP_LATENCY = int(os.getenv("RTSP_LATENCY", "0"))

CAM_W = int(os.getenv("CAM_W", "1280"))
CAM_H = int(os.getenv("CAM_H", "720"))
CAM_FPS = int(os.getenv("CAM_FPS", "30"))
SENSOR_MODE = int(os.getenv("SENSOR_MODE", "3"))
FLIP = int(os.getenv("FLIP", "0"))

IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CONF_THRES = float(os.getenv("CONF", "0.25"))
IOU_THRES = float(os.getenv("IOU", "0.45"))

ONNX_WEIGHTS = os.getenv("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")

SHOW = (os.getenv("SHOW", "1") == "1") and bool(os.environ.get("DISPLAY", ""))
WINDOW_NAME = os.getenv("WINDOW", "CSI-ONNX" if SRC == "csi" else "RTSP-ONNX")


def csi_pipeline(sensor_id=0, sensor_mode=3, w=1280, h=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=(string)BGRx ! videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def rtsp_pipeline(url, codec="h264", latency=0):
    codec = codec.lower().strip()
    depay = "rtph264depay" if codec == "h264" else "rtph265depay"
    parse = "h264parse" if codec == "h264" else "h265parse"
    return (
        f"rtspsrc location={url} latency={latency} drop-on-latency=true ! "
        f"{depay} ! {parse} ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape[1] - nw, new_shape[0] - nh
    dw /= 2
    dh /= 2

    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms_numpy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
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

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess_yolo(pred, conf_thres=0.25, iou_thres=0.45):
    pred = np.array(pred)
    # OpenCV dnn.forward() hay ra shape (1,25200,85) hoặc (1,25200,6)
    if pred.ndim == 3:
        pred = pred[0]

    if pred.shape[1] == 6:
        xywh = pred[:, 0:4]
        conf = pred[:, 4]
        cls = pred[:, 5]
        m = conf >= conf_thres
        xywh, conf, cls = xywh[m], conf[m], cls[m]
        boxes = xywh2xyxy(xywh)
        keep = nms_numpy(boxes, conf, iou_thres)
        return np.column_stack([boxes[keep], conf[keep], cls[keep]])

    xywh = pred[:, 0:4]
    obj = pred[:, 4:5]
    cls_scores = pred[:, 5:]
    cls = np.argmax(cls_scores, axis=1)
    cls_conf = np.max(cls_scores, axis=1, keepdims=True)
    conf = (obj * cls_conf).squeeze(1)

    m = conf >= conf_thres
    xywh, conf, cls = xywh[m], conf[m], cls[m]
    boxes = xywh2xyxy(xywh)
    keep = nms_numpy(boxes, conf, iou_thres)
    return np.column_stack([boxes[keep], conf[keep], cls[keep]])


def scale_boxes(boxes_xyxy, r, pad, orig_shape):
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= r
    h0, w0 = orig_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)
    return boxes


def main():
    print(f"[INFO] SRC={SRC} SHOW={SHOW} IMG_SIZE={IMG_SIZE} CONF={CONF_THRES} IOU={IOU_THRES}")
    print(f"[INFO] ONNX_WEIGHTS={ONNX_WEIGHTS}")

    # Open camera
    if SRC == "rtsp":
        if not RTSP_URL:
            raise SystemExit("Thiếu RTSP_URL")
        pipeline = rtsp_pipeline(RTSP_URL, codec=RTSP_CODEC, latency=RTSP_LATENCY)
    else:
        pipeline = csi_pipeline(sensor_id=0, sensor_mode=SENSOR_MODE, w=CAM_W, h=CAM_H, fps=CAM_FPS, flip=FLIP)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Không mở được camera")
    print("[OK] Camera opened")

    if not os.path.exists(ONNX_WEIGHTS):
        raise SystemExit(f"[ERROR] Không thấy ONNX: {ONNX_WEIGHTS}")

    # Load ONNX via OpenCV DNN
    net = cv2.dnn.readNetFromONNX(ONNX_WEIGHTS)

    # Nếu OpenCV build có CUDA thì bật (không có cũng chạy bình thường)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        print("[OK] OpenCV DNN CUDA FP16 enabled")
    except Exception:
        print("[WARN] OpenCV DNN không có CUDA, chạy CPU")

    if SHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    t_last = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        orig = frame
        img, r, pad = letterbox(orig, IMG_SIZE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(rgb, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=False, crop=False)

        net.setInput(blob)
        pred = net.forward()  # (1,25200,dim)

        det = postprocess_yolo(pred, CONF_THRES, IOU_THRES)
        det_count = 0

        if det is not None and len(det) > 0:
            det = det.astype(np.float32)
            boxes = scale_boxes(det[:, :4], r, pad, orig.shape)
            scores = det[:, 4]
            clss = det[:, 5]
            det_count = len(boxes)

            for (x1, y1, x2, y2), sc, c in zip(boxes, scores, clss):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig, f"{int(c)} {sc:.2f}", (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.time()
        dt = now - t_last
        t_last = now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = fps_smooth * 0.9 + fps * 0.1 if fps_smooth > 0 else fps

        print(f"FPS ~ {fps_smooth:.1f}, det={det_count}")

        if SHOW:
            cv2.imshow(WINDOW_NAME, orig)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):
                break

    cap.release()
    if SHOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
