import os
import time
import math
import argparse
from typing import List, Tuple

import cv2
import numpy as np

# ----------------------------
# Defaults (can override via env or args)
# ----------------------------
DEFAULT_DET_ONNX = "./model/LP_detector_nano_61.onnx"
DEFAULT_OCR_ONNX = "./model/LP_ocr_nano_62.onnx"
DEFAULT_DET_ENGINE = "./model/LP_detector_nano_61_fp16.engine"
DEFAULT_OCR_ENGINE = "./model/LP_ocr_nano_62_fp16.engine"

# Fallback OCR class names (you can replace with your repo's exact order if different)
# 0-9 + A-Z (common)
DEFAULT_OCR_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except:
        return default


# ----------------------------
# GStreamer pipelines
# ----------------------------
def gst_csi_pipeline(
    sensor_id=0,
    sensor_mode=3,          # mode=3 often ~1640x1232@30 on IMX219
    capture_w=1640,
    capture_h=1232,
    display_w=1280,
    display_h=720,
    framerate=30,
    flip=0,
) -> str:
    # appsink drop/max-buffers for less latency & smoother
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={display_w}, height={display_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

def gst_rtsp_pipeline(url: str, latency=200, width=1280, height=720) -> str:
    # HW decode: nvv4l2decoder
    return (
        f"rtspsrc location={url} latency={latency} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


# ----------------------------
# Utils: letterbox + NMS + decode YOLOv5 ONNX output
# ----------------------------
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def nms_xyxy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
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

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def decode_yolo(pred: np.ndarray, conf_thres=0.25, iou_thres=0.45):
    """
    Supports:
    - (N, 6): [x,y,w,h,conf,cls]
    - (N, 5+nc): [x,y,w,h,obj_conf, class_probs...]
    """
    if pred.ndim == 3:
        pred = pred[0]
    if pred.size == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

    if pred.shape[1] == 6:
        xywh = pred[:, 0:4]
        conf = pred[:, 4]
        cls = pred[:, 5].astype(np.int32)
        mask = conf >= conf_thres
        xywh, conf, cls = xywh[mask], conf[mask], cls[mask]
        if xywh.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

    else:
        xywh = pred[:, 0:4]
        obj = pred[:, 4]
        probs = pred[:, 5:]
        cls = np.argmax(probs, axis=1).astype(np.int32)
        cls_conf = probs[np.arange(probs.shape[0]), cls]
        conf = obj * cls_conf
        mask = conf >= conf_thres
        xywh, conf, cls = xywh[mask], conf[mask], cls[mask]
        if xywh.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)

    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_xyxy(boxes, conf, iou_thres)
    return boxes[keep], conf[keep], cls[keep]

def scale_coords(boxes, r, dwdh, orig_shape):
    # boxes are in letterbox image coords
    dw, dh = dwdh
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= r
    # clip
    h, w = orig_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


# ----------------------------
# ONNX runtime wrapper (CPU fallback)
# ----------------------------
class ORTSession:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.use_ort = False
        self.sess = None
        self.input_name = None
        self.out_names = None

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            # Prefer TensorRT/CUDA if exists
            prefer = []
            for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
                if p in providers:
                    prefer.append(p)
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=prefer)
            self.input_name = self.sess.get_inputs()[0].name
            self.out_names = [o.name for o in self.sess.get_outputs()]
            self.use_ort = True
            print(f"[INFO] ORT providers for {onnx_path}: {self.sess.get_providers()}")
        except Exception as e:
            print(f"[WARN] onnxruntime not available or failed: {e}")
            self.use_ort = False

        # OpenCV DNN fallback (CPU only by default to avoid cuDNN mismatch crash)
        self.net = None
        if not self.use_ort:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            # Force CPU to avoid cudnn mismatch in many Jetson images
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[INFO] Using OpenCV DNN CPU for {onnx_path}")

    def run(self, blob: np.ndarray) -> np.ndarray:
        if self.use_ort:
            out = self.sess.run(self.out_names, {self.input_name: blob})[0]
            return out
        else:
            self.net.setInput(blob)
            out = self.net.forward()
            return out


# ----------------------------
# OCR helper (sort into 2 lines)
# ----------------------------
def ocr_from_dets(boxes, cls_ids, class_names: List[str]) -> str:
    if boxes.shape[0] == 0:
        return ""

    # centers
    yc = (boxes[:, 1] + boxes[:, 3]) / 2.0
    xc = (boxes[:, 0] + boxes[:, 2]) / 2.0

    y_min, y_max = float(np.min(yc)), float(np.max(yc))
    if (y_max - y_min) < 10:  # mostly single line
        order = np.argsort(xc)
        return "".join(class_names[int(cls_ids[i])] for i in order)

    split = (y_min + y_max) / 2.0
    top_idx = np.where(yc <= split)[0]
    bot_idx = np.where(yc > split)[0]

    top_sorted = top_idx[np.argsort(xc[top_idx])] if top_idx.size else np.array([], dtype=np.int32)
    bot_sorted = bot_idx[np.argsort(xc[bot_idx])] if bot_idx.size else np.array([], dtype=np.int32)

    top = "".join(class_names[int(cls_ids[i])] for i in top_sorted)
    bot = "".join(class_names[int(cls_ids[i])] for i in bot_sorted)

    if top and bot:
        return f"{top} {bot}"
    return top or bot


# ----------------------------
# Drawing
# ----------------------------
def draw_plate(frame, x1, y1, x2, y2, text="", alpha=0.25):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # red translucent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # thin green border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if text:
        # red text with black outline for readability
        tx, ty = x1, max(0, y1 - 8)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.getenv("SRC", "csi"), choices=["csi", "rtsp", "webcam"])
    parser.add_argument("--rtsp", default=os.getenv("RTSP_URL", ""))
    parser.add_argument("--cam", type=int, default=env_int("CAMERA_INDEX", 0))

    parser.add_argument("--det", default=os.getenv("DET_ONNX", DEFAULT_DET_ONNX))
    parser.add_argument("--ocr", default=os.getenv("OCR_ONNX", DEFAULT_OCR_ONNX))

    parser.add_argument("--img", type=int, default=env_int("IMG_SIZE", 640))
    parser.add_argument("--conf", type=float, default=env_float("CONF", 0.25))
    parser.add_argument("--iou", type=float, default=env_float("IOU", 0.45))
    parser.add_argument("--skip", type=int, default=env_int("SKIP", 0))  # skip frames for speed

    parser.add_argument("--show", type=int, default=1 if env_bool("SHOW", True) else 0)
    parser.add_argument("--csi_fps", type=int, default=env_int("CSI_FPS", 30))
    parser.add_argument("--csi_mode", type=int, default=env_int("CSI_MODE", 3))
    parser.add_argument("--csi_w", type=int, default=env_int("CSI_W", 1640))
    parser.add_argument("--csi_h", type=int, default=env_int("CSI_H", 1232))
    parser.add_argument("--out_w", type=int, default=env_int("OUT_W", 1280))
    parser.add_argument("--out_h", type=int, default=env_int("OUT_H", 720))
    parser.add_argument("--latency", type=int, default=env_int("RTSP_LATENCY", 200))
    args = parser.parse_args()

    SHOW = bool(args.show)
    print(f"[INFO] SRC={args.src} SHOW={SHOW} IMG_SIZE={args.img} CONF={args.conf} IOU={args.iou} SKIP={args.skip}")
    print(f"[INFO] DET_ONNX={args.det}")
    print(f"[INFO] OCR_ONNX={args.ocr}")

    # Open capture
    if args.src == "csi":
        pipe = gst_csi_pipeline(
            sensor_id=0,
            sensor_mode=args.csi_mode,
            capture_w=args.csi_w,
            capture_h=args.csi_h,
            display_w=args.out_w,
            display_h=args.out_h,
            framerate=args.csi_fps,
            flip=0,
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    elif args.src == "rtsp":
        if not args.rtsp:
            raise SystemExit("RTSP_URL is empty. Set env RTSP_URL or pass --rtsp")
        pipe = gst_rtsp_pipeline(args.rtsp, latency=args.latency, width=args.out_w, height=args.out_h)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise SystemExit("[FATAL] Cannot open video source")

    # Load ONNX models
    det_sess = ORTSession(args.det)
    ocr_sess = ORTSession(args.ocr)

    ocr_names = DEFAULT_OCR_CLASSES

    last_boxes = np.empty((0, 4))
    last_texts = []
    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # avoid crashing into grayscale/None -> DNN "channels=1"
            print("[WARN] Empty frame, retry...")
            time.sleep(0.01)
            continue

        # Ensure BGR 3ch
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        frames += 1
        do_infer = True
        if args.skip > 0:
            do_infer = (frames % (args.skip + 1) == 1)

        if do_infer:
            im_lb, r, dwdh = letterbox(frame, args.img)
            blob = im_lb.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[None]  # 1,3,H,W

            pred_det = det_sess.run(blob)
            boxes, confs, clsids = decode_yolo(pred_det, args.conf, args.iou)
            boxes = scale_coords(boxes, r, dwdh, frame.shape)

            last_boxes = boxes
            last_texts = []

            # OCR each detected plate (limit to top few for speed)
            # sort by confidence (if available)
            if boxes.shape[0] > 0:
                # Just process up to 3 plates per frame to keep FPS stable
                for bi in range(min(3, boxes.shape[0])):
                    x1, y1, x2, y2 = boxes[bi]
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        last_texts.append("")
                        continue

                    # OCR on crop
                    crop_lb, r2, dwdh2 = letterbox(crop, 320)  # smaller input for OCR speed
                    blob2 = crop_lb.astype(np.float32) / 255.0
                    blob2 = blob2.transpose(2, 0, 1)[None]

                    pred_ocr = ocr_sess.run(blob2)
                    c_boxes, c_confs, c_cls = decode_yolo(pred_ocr, conf_thres=0.25, iou_thres=0.35)
                    # scale back to crop coords
                    c_boxes = scale_coords(c_boxes, r2, dwdh2, crop.shape)

                    text = ocr_from_dets(c_boxes, c_cls, ocr_names)
                    last_texts.append(text)

        # Draw last results
        plates = last_boxes.shape[0]
        for i in range(plates):
            x1, y1, x2, y2 = last_boxes[i]
            text = last_texts[i] if i < len(last_texts) else ""
            draw_plate(frame, x1, y1, x2, y2, text=text, alpha=0.22)

        # FPS
        if frames % 15 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            print(f"FPS ~ {fps:.1f}, plates={plates}")

        if SHOW:
            title = "CSI-ONNX" if args.src == "csi" else ("RTSP-ONNX" if args.src == "rtsp" else "WEBCAM-ONNX")
            cv2.putText(frame, f"FPS {frames/(time.time()-t0+1e-9):.1f} plates={plates}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    if SHOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
