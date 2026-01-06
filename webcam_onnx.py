import os
import time
import argparse
import numpy as np
import cv2

# ---------------------------
# CONFIG via environment
# ---------------------------
SRC = os.getenv("SRC", "csi")                 # "csi" or "rtsp"
RTSP_URL = os.getenv("RTSP_URL", "")
RTSP_CODEC = os.getenv("RTSP_CODEC", "h264")  # "h264" or "h265"
RTSP_LATENCY = int(os.getenv("RTSP_LATENCY", "0"))

CAM_W = int(os.getenv("CAM_W", "1280"))
CAM_H = int(os.getenv("CAM_H", "720"))
CAM_FPS = int(os.getenv("CAM_FPS", "30"))          # ép fps để đỡ giật
SENSOR_MODE = int(os.getenv("SENSOR_MODE", "3"))   # IMX219 hay dùng mode 3 (30fps)
FLIP = int(os.getenv("FLIP", "0"))

IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))       # input size ONNX
CONF_THRES = float(os.getenv("CONF", "0.25"))
IOU_THRES = float(os.getenv("IOU", "0.45"))

ONNX_WEIGHTS = os.getenv("ONNX_WEIGHTS", "./model/LP_detector_nano_61.onnx")

# SHOW: tự tắt nếu không có DISPLAY (SSH/headless)
SHOW = (os.getenv("SHOW", "1") == "1") and bool(os.environ.get("DISPLAY", ""))

WINDOW_NAME = os.getenv("WINDOW", "CSI-ONNX" if SRC == "csi" else "RTSP-ONNX")

# ---------------------------
# GStreamer pipelines
# ---------------------------
def csi_pipeline(sensor_id=0, sensor_mode=3,
                 capture_width=1280, capture_height=720,
                 display_width=1280, display_height=720,
                 framerate=30, flip_method=0):
    # appsink drop=1 max-buffers=1 sync=false => giảm lag/giật do queue
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

def rtsp_pipeline(url: str, codec="h264", latency=0):
    codec = codec.lower().strip()
    if codec not in ("h264", "h265"):
        codec = "h264"

    depay = "rtph264depay" if codec == "h264" else "rtph265depay"
    parse = "h264parse" if codec == "h264" else "h265parse"

    return (
        f"rtspsrc location={url} latency={latency} drop-on-latency=true ! "
        f"{depay} ! {parse} ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

# ---------------------------
# ONNX Runtime loader
# ---------------------------
def load_onnx_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except Exception as e:
        raise SystemExit(
            "Thiếu onnxruntime. Trong Docker/Jetson cài: pip3 install onnxruntime\n"
            f"Lỗi import: {e}"
        )

    # thử CUDA trước (nếu có), không có thì CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    avail = ort.get_available_providers()
    use_providers = [p for p in providers if p in avail]
    if not use_providers:
        use_providers = ["CPUExecutionProvider"]

    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = int(os.getenv("ORT_INTRA", "2"))
    sess_opt.inter_op_num_threads = int(os.getenv("ORT_INTER", "1"))

    session = ort.InferenceSession(onnx_path, sess_options=sess_opt, providers=use_providers)
    input_name = session.get_inputs()[0].name
    return session, input_name, use_providers

# ---------------------------
# Utils: letterbox + NMS
# ---------------------------
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im_padded, r, (left, top)

def xywh2xyxy(x):
    # x: (...,4) with xywh
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def nms_numpy(boxes, scores, iou_thres=0.45):
    # boxes: (N,4) xyxy, scores: (N,)
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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

def postprocess_yolo(output, conf_thres=0.25, iou_thres=0.45):
    """
    Hỗ trợ 2 dạng:
    A) output shape (1,25200,6): [x,y,w,h,conf,cls]
    B) output shape (1,25200,85): [x,y,w,h,obj,cls...]
    Trả về list det: [x1,y1,x2,y2,score,cls]
    """
    pred = output
    if isinstance(pred, list):
        pred = pred[0]
    pred = np.array(pred)

    if pred.ndim == 3:
        pred = pred[0]  # (N,dim)

    if pred.shape[1] == 6:
        xywh = pred[:, 0:4]
        conf = pred[:, 4]
        cls = pred[:, 5]
        mask = conf >= conf_thres
        xywh = xywh[mask]
        conf = conf[mask]
        cls = cls[mask]
        boxes = xywh2xyxy(xywh)
        keep = nms_numpy(boxes, conf, iou_thres)
        return np.column_stack([boxes[keep], conf[keep], cls[keep]])

    # YOLOv5 standard
    # [x,y,w,h,obj, class_scores...]
    xywh = pred[:, 0:4]
    obj = pred[:, 4:5]
    cls_scores = pred[:, 5:]
    cls = np.argmax(cls_scores, axis=1)
    cls_conf = np.max(cls_scores, axis=1, keepdims=True)

    conf = (obj * cls_conf).squeeze(1)
    mask = conf >= conf_thres

    xywh = xywh[mask]
    conf = conf[mask]
    cls = cls[mask]

    boxes = xywh2xyxy(xywh)
    keep = nms_numpy(boxes, conf, iou_thres)
    return np.column_stack([boxes[keep], conf[keep], cls[keep]])

def scale_boxes(boxes_xyxy, r, pad, orig_shape):
    # boxes in letterbox image -> original image
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

# ---------------------------
# Main
# ---------------------------
def main():
    print(f"[INFO] SRC={SRC} SHOW={SHOW} IMG_SIZE={IMG_SIZE} CONF={CONF_THRES} IOU={IOU_THRES}")
    print(f"[INFO] ONNX_WEIGHTS={ONNX_WEIGHTS}")

    # Open capture
    if SRC == "rtsp":
        if not RTSP_URL:
            raise SystemExit("Thiếu RTSP_URL. Ví dụ: SRC=rtsp RTSP_URL='rtsp://...' python3 rtsp.py")
        pipeline = rtsp_pipeline(RTSP_URL, codec=RTSP_CODEC, latency=RTSP_LATENCY)
    else:
        pipeline = csi_pipeline(
            sensor_id=0,
            sensor_mode=SENSOR_MODE,
            capture_width=CAM_W,
            capture_height=CAM_H,
            display_width=CAM_W,
            display_height=CAM_H,
            framerate=CAM_FPS,
            flip_method=FLIP,
        )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Không mở được camera. Kiểm tra pipeline, quyền /tmp/argus_socket, DISPLAY...")

    print("[OK] Camera opened")

    # Load onnx
    session, input_name, providers = load_onnx_session(ONNX_WEIGHTS)
    print(f"[OK] ONNX Runtime providers: {providers}")

    if SHOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    t_last = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # RTSP có lúc drop frame
                print("[WARN] frame None, retry...")
                time.sleep(0.01)
                continue

            orig = frame
            img, r, pad = letterbox(orig, IMG_SIZE)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # NCHW float32 0..1
            x = img_rgb.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]

            out = session.run(None, {input_name: x})

            det = postprocess_yolo(out, CONF_THRES, IOU_THRES)
            det_count = 0
            if det is not None and len(det) > 0:
                det = np.array(det, dtype=np.float32)
                boxes = det[:, 0:4]
                scores = det[:, 4]
                clss = det[:, 5]

                boxes = scale_boxes(boxes, r, pad, orig.shape)
                det_count = len(boxes)

                for (x1, y1, x2, y2), sc, c in zip(boxes, scores, clss):
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(orig, f"{int(c)} {sc:.2f}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # FPS
            now = time.time()
            dt = now - t_last
            t_last = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps_smooth * 0.9 + fps * 0.1 if fps_smooth > 0 else fps

            print(f"FPS ~ {fps_smooth:.1f}, det={det_count}")

            if SHOW:
                cv2.imshow(WINDOW_NAME, orig)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

    finally:
        cap.release()
        if SHOW:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
