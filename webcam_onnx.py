# webcam_onnx.py
import os, time
import cv2
import numpy as np

# =========================
# Config via ENV
# =========================
SRC        = os.getenv("SRC", "csi")          # csi | rtsp | usb
SHOW       = os.getenv("SHOW", "1") == "1"    # 1: imshow, 0: no window (ssh headless)
IMG_SIZE   = int(os.getenv("IMG_SIZE", "640"))
CONF_THRES = float(os.getenv("CONF", "0.25"))
IOU_THRES  = float(os.getenv("IOU", "0.45"))
SKIP       = int(os.getenv("SKIP", "0"))      # 0: process every frame, 1: skip 1 frame, etc.

# CSI
CSI_W      = int(os.getenv("CSI_W", "1280"))
CSI_H      = int(os.getenv("CSI_H", "720"))
CSI_FPS    = int(os.getenv("CSI_FPS", "30"))
CSI_FLIP   = int(os.getenv("CSI_FLIP", "0"))
CSI_MODE   = os.getenv("CSI_MODE", "")        # e.g. "4" or "2" ... optional

# RTSP
RTSP_URL     = os.getenv("RTSP_URL", "")
RTSP_LATENCY = int(os.getenv("RTSP_LATENCY", "200"))

# Models
DET_ONNX = os.getenv("DET_ONNX", "./model/LP_detector_nano_61.onnx")
OCR_ONNX = os.getenv("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

# OCR classes (default: 0-9 + A-Z)
DEFAULT_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =========================
# Utils: letterbox + NMS + scale
# =========================
def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_shape[1] - nw, new_shape[0] - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def xywh2xyxy(x):
    # x: [cx,cy,w,h]
    y = np.zeros_like(x)
    y[:,0] = x[:,0] - x[:,2]/2
    y[:,1] = x[:,1] - x[:,3]/2
    y[:,2] = x[:,0] + x[:,2]/2
    y[:,3] = x[:,1] + x[:,3]/2
    return y

def scale_coords(coords, r, pad, original_shape):
    # coords: Nx4 in padded space -> original frame
    coords[:, [0,2]] -= pad[0]
    coords[:, [1,3]] -= pad[1]
    coords[:, :4] /= r
    # clip
    h, w = original_shape[:2]
    coords[:,0] = np.clip(coords[:,0], 0, w-1)
    coords[:,2] = np.clip(coords[:,2], 0, w-1)
    coords[:,1] = np.clip(coords[:,1], 0, h-1)
    coords[:,3] = np.clip(coords[:,3], 0, h-1)
    return coords

def yolo_onnx_infer(net, frame_bgr, img_size, conf_thres, iou_thres, force_single_class=False):
    img, r, pad = letterbox(frame_bgr, img_size)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (img_size, img_size), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()

    # YOLOv5 ONNX usually: (1,25200,5+nc)
    out = np.squeeze(out, axis=0)  # (25200, no)
    if out.ndim != 2 or out.shape[1] < 6:
        return [], [], []

    boxes = out[:, 0:4]  # cx,cy,w,h
    obj   = out[:, 4:5]  # objectness
    cls   = out[:, 5:]   # class scores

    if cls.shape[1] == 1 or force_single_class:
        scores = (obj * cls[:, 0:1]).squeeze(1)
        class_ids = np.zeros_like(scores, dtype=np.int32)
    else:
        class_ids = np.argmax(cls, axis=1).astype(np.int32)
        scores = (obj.squeeze(1) * cls[np.arange(cls.shape[0]), class_ids])

    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if len(scores) == 0:
        return [], [], []

    boxes_xyxy = xywh2xyxy(boxes)
    boxes_xyxy = scale_coords(boxes_xyxy, r, pad, frame_bgr.shape)

    # NMS
    b = boxes_xyxy.astype(np.float32)
    x = b[:,0]; y = b[:,1]; w = (b[:,2]-b[:,0]); h = (b[:,3]-b[:,1])
    nms_boxes = [[float(x[i]), float(y[i]), float(w[i]), float(h[i])] for i in range(len(scores))]
    idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_thres, iou_thres)
    if len(idxs) == 0:
        return [], [], []
    idxs = idxs.flatten().tolist()

    final_boxes = b[idxs].astype(int)
    final_scores = scores[idxs]
    final_cls = class_ids[idxs]
    return final_boxes, final_scores, final_cls

def decode_plate(chars_boxes, chars_scores, chars_cls, classes):
    """
    chars_boxes: Nx4 (x1,y1,x2,y2) on cropped plate image
    decode by 2-line split via y-center
    """
    if len(chars_boxes) == 0:
        return ""

    items = []
    for (x1,y1,x2,y2), cid, sc in zip(chars_boxes, chars_cls, chars_scores):
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        ch = classes[int(cid)] if int(cid) < len(classes) else "?"
        items.append((cx, cy, ch, sc))

    # split into 1 or 2 lines
    ys = np.array([it[1] for it in items])
    y_med = float(np.median(ys))

    top = [it for it in items if it[1] <= y_med]
    bot = [it for it in items if it[1] >  y_med]

    top.sort(key=lambda x: x[0])
    bot.sort(key=lambda x: x[0])

    line1 = "".join([it[2] for it in top]).strip()
    line2 = "".join([it[2] for it in bot]).strip()

    if line2:
        return f"{line1}-{line2}"
    return line1

# =========================
# Camera pipelines
# =========================
def gst_csi_pipeline(width, height, fps, flip, sensor_mode=""):
    sm = f" sensor-mode={sensor_mode} " if str(sensor_mode).strip() != "" else " "
    return (
        f"nvarguscamerasrc{sm} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1 sync=false"
    )

def gst_rtsp_pipeline(url, latency=200):
    # Jetson decode H264 via nvv4l2decoder
    return (
        f"rtspsrc location={url} latency={latency} ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1 sync=false"
    )

def open_capture():
    if SRC == "csi":
        pipe = gst_csi_pipeline(CSI_W, CSI_H, CSI_FPS, CSI_FLIP, CSI_MODE)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap
    if SRC == "rtsp":
        if not RTSP_URL:
            raise RuntimeError("SRC=rtsp nhưng thiếu RTSP_URL")
        pipe = gst_rtsp_pipeline(RTSP_URL, RTSP_LATENCY)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # fallback direct
            cap = cv2.VideoCapture(RTSP_URL)
        return cap
    # usb
    cam_index = int(os.getenv("CAM_INDEX", "0"))
    return cv2.VideoCapture(cam_index)

# =========================
# Load ONNX nets
# =========================
def load_net(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không thấy file: {path}")
    net = cv2.dnn.readNetFromONNX(path)
    # Try CUDA if OpenCV has it
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def main():
    print(f"[INFO] SRC={SRC} SHOW={SHOW} IMG_SIZE={IMG_SIZE} CONF={CONF_THRES} IOU={IOU_THRES} SKIP={SKIP}")
    print(f"[INFO] DET_ONNX={DET_ONNX}")
    print(f"[INFO] OCR_ONNX={OCR_ONNX}")

    det_net = load_net(DET_ONNX)
    ocr_net = load_net(OCR_ONNX)

    cap = open_capture()
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera/stream. Check pipeline hoặc quyền /dev/video* / RTSP.")

    classes = DEFAULT_CLASSES

    t0 = time.time()
    fps_smooth = 0.0
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] frame None / stream ended")
            break

        frame_id += 1
        if SKIP > 0 and (frame_id % (SKIP+1) != 0):
            if SHOW:
                cv2.imshow(f"{SRC.upper()}-ONNX", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        t1 = time.time()

        # 1) Detect plate(s)
        plate_boxes, plate_scores, _ = yolo_onnx_infer(det_net, frame, IMG_SIZE, CONF_THRES, IOU_THRES, force_single_class=True)

        plate_texts = []
        for (x1,y1,x2,y2), psc in zip(plate_boxes, plate_scores):
            # pad crop
            pad = 8
            x1c = max(0, x1-pad); y1c = max(0, y1-pad)
            x2c = min(frame.shape[1]-1, x2+pad); y2c = min(frame.shape[0]-1, y2+pad)

            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            # 2) OCR on crop (characters)
            ch_boxes, ch_scores, ch_cls = yolo_onnx_infer(ocr_net, crop, IMG_SIZE, conf_thres=0.25, iou_thres=0.45, force_single_class=False)
            text = decode_plate(ch_boxes, ch_scores, ch_cls, classes)
            plate_texts.append((x1,y1,x2,y2, float(psc), text))

            # draw plate box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            label = text if text else f"plate {psc:.2f}"
            cv2.putText(frame, label, (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # FPS
        dt = time.time() - t1
        inst_fps = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9*fps_smooth + 0.1*inst_fps

        cv2.putText(frame, f"FPS {fps_smooth:.1f} plates={len(plate_boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

        if SHOW:
            cv2.imshow(f"{SRC.upper()}-ONNX", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    if SHOW:
        cv2.destroyAllWindows()
    print("[DONE]")

if __name__ == "__main__":
    main()
