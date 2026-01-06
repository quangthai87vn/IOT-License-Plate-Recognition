#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np

# ----------------------------
# Config (env-first)
# ----------------------------
SRC         = os.getenv("SRC", "csi")          # csi | rtsp | webcam
SHOW        = int(os.getenv("SHOW", "1"))      # 1 = imshow, 0 = headless
IMG_SIZE    = int(os.getenv("IMG_SIZE", "640"))
CONF_THRES  = float(os.getenv("CONF", "0.25"))
IOU_THRES   = float(os.getenv("IOU", "0.45"))
SKIP        = int(os.getenv("SKIP", "0"))      # skip frames to reduce load

# Model paths
DET_ONNX    = os.getenv("DET_ONNX", "./model/LP_detector_nano_61.onnx")
# Nếu bạn có OCR ONNX riêng thì set thêm:
OCR_ONNX    = os.getenv("OCR_ONNX", "")        # ví dụ: ./model/LP_ocr_nano_62.onnx (optional)

# Camera/RTSP options
CSI_W       = int(os.getenv("CSI_W", "1280"))
CSI_H       = int(os.getenv("CSI_H", "720"))
CSI_FPS     = int(os.getenv("CSI_FPS", "30"))  # ép về 30 cho mượt, đừng để 120
CSI_SENSOR  = int(os.getenv("CSI_SENSOR", "0"))

RTSP_URL    = os.getenv("RTSP_URL", "")
RTSP_LAT    = int(os.getenv("RTSP_LATENCY", "120"))  # 80~200 tùy mạng
RTSP_CODEC  = os.getenv("RTSP_CODEC", "h264")        # h264/h265 (h264 phổ biến)

# Drawing style
BORDER_THICK = int(os.getenv("BORDER_THICK", "2"))   # viền xanh mảnh
FILL_ALPHA   = float(os.getenv("FILL_ALPHA", "0.25"))# tô đỏ mờ
TEXT_SCALE   = float(os.getenv("TEXT_SCALE", "0.9"))

# OCR (optional) - nếu cần bạn chỉnh sau
OCR_W       = int(os.getenv("OCR_W", "160"))
OCR_H       = int(os.getenv("OCR_H", "40"))
OCR_CHARS   = os.getenv("OCR_CHARS", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")  # tuỳ model
OCR_BLANK   = int(os.getenv("OCR_BLANK", "0"))       # CTC blank index

# ----------------------------
# GStreamer pipelines
# ----------------------------
def gst_csi(sensor_id=0, width=1280, height=720, fps=30):
    # appsink drop/max-buffers để realtime, không bị dồn frame
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def gst_rtsp(url, latency=120, codec="h264"):
    # decode HW: nvv4l2decoder
    depay = "rtph264depay" if codec.lower() == "h264" else "rtph265depay"
    parse = "h264parse" if codec.lower() == "h264" else "h265parse"
    return (
        f"rtspsrc location={url} latency={latency} drop-on-late=true ! "
        f"{depay} ! {parse} ! nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

# ----------------------------
# Threaded capture (queue=1)
# ----------------------------
class VideoStream:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.t = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.t.start()
        return self

    def update(self):
        while not self.stopped:
            ok, frm = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frm
            if not ok:
                time.sleep(0.02)

    def read(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

# ----------------------------
# YOLOv5 ONNX inference (onnxruntime -> fallback OpenCV DNN)
# Output assumed: (1,25200, 5+nc). With 1 class => 6
# ----------------------------
def letterbox(im, new_shape=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def nms_numpy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1+1) * (y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

class YoloDetONNX:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.backend = None
        self.session = None
        self.net = None
        self.input_name = None

        # Try onnxruntime first
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            avail = ort.get_available_providers()
            use = [p for p in providers if p in avail]
            if not use:
                use = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=use)
            self.input_name = self.session.get_inputs()[0].name
            self.backend = f"onnxruntime:{use}"
            print("[INFO] DET backend =", self.backend)
        except Exception as e:
            # Fallback OpenCV DNN
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            # Try CUDA if OpenCV supports it
            try:
                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                    self.backend = "opencv-dnn:cuda-fp16"
                else:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.backend = "opencv-dnn:cpu"
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "opencv-dnn:cpu"
            print("[INFO] DET backend =", self.backend, "| onnxruntime not used:", str(e)[:120])

    def infer(self, bgr, img_size=640, conf_thres=0.25, iou_thres=0.45):
        im0 = bgr
        img, r, (padx, pady) = letterbox(im0, img_size)
        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        img = np.transpose(img, (2,0,1))[None, ...]       # 1x3xHxW

        if self.session is not None:
            out = self.session.run(None, {self.input_name: img})[0]
        else:
            blob = img
            self.net.setInput(blob)
            out = self.net.forward()
        out = np.squeeze(out)  # Nx(5+nc)

        if out.ndim == 1:
            out = out[None, :]

        # YOLOv5: [x,y,w,h,obj,cls1..]
        xywh = out[:, :4]
        obj  = out[:, 4:5]
        cls  = out[:, 5:]
        if cls.shape[1] == 0:
            conf = obj[:,0]
        else:
            conf = (obj * cls.max(axis=1, keepdims=True))[:,0]
        keep = conf >= conf_thres
        if not np.any(keep):
            return []

        xywh = xywh[keep]
        conf = conf[keep]

        # xywh -> xyxy (on padded img space)
        x = xywh[:,0]; y = xywh[:,1]; w = xywh[:,2]; h = xywh[:,3]
        x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
        boxes = np.stack([x1,y1,x2,y2], axis=1)

        # NMS
        idx = nms_numpy(boxes, conf, iou_thres=iou_thres)
        boxes = boxes[idx]
        conf  = conf[idx]

        # Scale back to original image coords
        boxes[:, [0,2]] -= padx
        boxes[:, [1,3]] -= pady
        boxes /= r

        # Clip
        h0, w0 = im0.shape[:2]
        boxes[:,0] = np.clip(boxes[:,0], 0, w0-1)
        boxes[:,1] = np.clip(boxes[:,1], 0, h0-1)
        boxes[:,2] = np.clip(boxes[:,2], 0, w0-1)
        boxes[:,3] = np.clip(boxes[:,3], 0, h0-1)

        dets = []
        for b, c in zip(boxes.astype(int), conf):
            dets.append((b[0], b[1], b[2], b[3], float(c)))
        return dets

# ----------------------------
# Optional OCR (CTC greedy) - nếu model bạn đúng kiểu CTC
# Nếu OCR model bạn khác format, bạn báo shape output mình chỉnh tiếp.
# ----------------------------
class OcrONNX:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.ok = False
        self.backend = None
        self.session = None
        self.net = None
        self.input_name = None

        if not onnx_path or not os.path.exists(onnx_path):
            return

        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            use = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            use = [p for p in use if p in avail] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=use)
            self.input_name = self.session.get_inputs()[0].name
            self.backend = f"onnxruntime:{use}"
            self.ok = True
            print("[INFO] OCR backend =", self.backend)
        except Exception as e:
            try:
                self.net = cv2.dnn.readNetFromONNX(onnx_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.backend = "opencv-dnn:cpu"
                self.ok = True
                print("[INFO] OCR backend =", self.backend, "| onnxruntime not used:", str(e)[:120])
            except Exception:
                self.ok = False

    def decode_ctc(self, logits, chars=OCR_CHARS, blank=OCR_BLANK):
        # logits: (T, C) or (1, T, C) or (T, 1, C)
        a = logits
        if a.ndim == 3:
            if a.shape[0] == 1:
                a = a[0]
            elif a.shape[1] == 1:
                a = a[:,0,:]
        # now (T, C)
        if a.ndim != 2:
            return ""

        ids = np.argmax(a, axis=1).tolist()
        res = []
        prev = None
        for i in ids:
            if i == blank:
                prev = i
                continue
            if prev == i:
                continue
            # map (assume blank=0 => chars starts at 1)
            idx = i - 1 if blank == 0 else i
            if 0 <= idx < len(chars):
                res.append(chars[idx])
            prev = i
        return "".join(res)

    def infer_text(self, crop_bgr):
        if not self.ok:
            return ""
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        # common OCR: 1x1xHxW
        blob = img[None, None, :, :]

        if self.session is not None:
            out = self.session.run(None, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        return self.decode_ctc(np.squeeze(out))

# ----------------------------
# Draw helpers
# ----------------------------
def draw_plate(frame, x1, y1, x2, y2, text="", conf=0.0):
    # red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    frame[:] = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0)

    # thin green border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BORDER_THICK)

    # red text
    if text:
        label = f"{text}"
    else:
        label = f"plate {conf:.2f}"
    ty = max(0, y1 - 10)
    cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 255), 2, cv2.LINE_AA)

# ----------------------------
# Main
# ----------------------------
def open_capture():
    if SRC == "csi":
        pipe = gst_csi(CSI_SENSOR, CSI_W, CSI_H, CSI_FPS)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap
    if SRC == "rtsp":
        if not RTSP_URL:
            raise RuntimeError("RTSP_URL is empty. Example: RTSP_URL=rtsp://ip:8554/xxx python3 rtsp.py")
        pipe = gst_rtsp(RTSP_URL, RTSP_LAT, RTSP_CODEC)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap
    # webcam
    idx = int(os.getenv("WEBCAM_INDEX", "0"))
    cap = cv2.VideoCapture(idx)
    return cap

def main():
    print(f"[INFO] SRC={SRC} SHOW={SHOW} IMG_SIZE={IMG_SIZE} CONF={CONF_THRES} IOU={IOU_THRES} SKIP={SKIP}")
    print(f"[INFO] DET_ONNX={DET_ONNX}")
    if OCR_ONNX:
        print(f"[INFO] OCR_ONNX={OCR_ONNX}")

    # reduce OpenCV CPU thread overhead (helps Nano)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    det = YoloDetONNX(DET_ONNX)
    ocr = OcrONNX(OCR_ONNX) if OCR_ONNX else None

    cap = open_capture()
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source. Check pipeline / DISPLAY / camera connection.")

    vs = VideoStream(cap).start()

    fps_hist = deque(maxlen=20)
    t_prev = time.time()
    frame_id = 0

    win_name = f"{SRC.upper()}-ONNX"
    if SHOW:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_id += 1
        if SKIP > 0 and (frame_id % (SKIP + 1) != 0):
            # show raw frame (optional)
            if SHOW:
                cv2.imshow(win_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        dets = det.infer(frame, IMG_SIZE, CONF_THRES, IOU_THRES)

        for (x1, y1, x2, y2, conf) in dets:
            # crop for OCR
            pad = int(0.05 * max((x2 - x1), (y2 - y1)))
            xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
            xx2 = min(frame.shape[1]-1, x2 + pad); yy2 = min(frame.shape[0]-1, y2 + pad)
            crop = frame[yy1:yy2, xx1:xx2].copy()

            text = ""
            if ocr is not None and ocr.ok:
                text = ocr.infer_text(crop)

            draw_plate(frame, x1, y1, x2, y2, text=text, conf=conf)

        # FPS
        t_now = time.time()
        fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        fps_hist.append(fps)
        fps_s = sum(fps_hist) / len(fps_hist)

        # HUD
        cv2.putText(frame, f"FPS {fps_s:.1f} plates={len(dets)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

        if SHOW:
            cv2.imshow(win_name, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break

    vs.stop()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        sys.exit(1)
