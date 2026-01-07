#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webcam_onnx.py  (giữ tên theo project cũ, nhưng đây là file MAIN chạy cả CSI/RTSP/Webcam)

- Ưu tiên TensorRT engine (det + ocr). Fallback OpenCV DNN nếu thiếu TRT.
- Fix treo (không subprocess), fix decode OCR CTC, fix resize shape theo engine binding.
- Hỗ trợ biển số 2 dòng.
"""

import os
import re
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

cv2.setNumThreads(0)

# -----------------------------
# Utils
# -----------------------------
def now_ms() -> float:
    return time.time() * 1000.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def xywh2xyxy(x):
    # x: [N,4] center_x, center_y, w, h
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_thres=0.45):
    # boxes: [N,4] xyxy
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # YOLO letterbox
    shape = im.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# -----------------------------
# TensorRT runner
# -----------------------------
class TRTRunner:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.ok = False

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except Exception as e:
            print("[TRT] tensorrt/pycuda import failed:", e)
            return

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            print("[TRT] Failed to load engine:", engine_path)
            return

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # binding indices
        self.input_idx = None
        self.output_idxs = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_idxs.append(i)

        if self.input_idx is None or len(self.output_idxs) == 0:
            print("[TRT] Invalid bindings in engine:", engine_path)
            return

        self.device_alloc = {}  # idx -> device ptr
        self.host_alloc = {}    # idx -> host (pagelocked)
        self.last_shapes = {}   # idx -> tuple(shape)
        self.ok = True

    def _profile_shape(self, binding_idx: int):
        # returns (min,opt,max) if available
        try:
            # profile 0
            return self.engine.get_profile_shape(0, binding_idx)
        except Exception:
            return None

    def _ensure_alloc(self, binding_idx: int, shape: Tuple[int, ...]):
        # allocate host/device for this binding shape
        import numpy as np

        dtype = self.trt.nptype(self.engine.get_binding_dtype(binding_idx))
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize

        if self.last_shapes.get(binding_idx) == tuple(shape):
            return

        # (re)alloc
        if binding_idx in self.device_alloc:
            try:
                self.device_alloc[binding_idx].free()
            except Exception:
                pass

        host = self.cuda.pagelocked_empty(size, dtype)
        device = self.cuda.mem_alloc(nbytes)

        self.host_alloc[binding_idx] = host
        self.device_alloc[binding_idx] = device
        self.last_shapes[binding_idx] = tuple(shape)

    def input_hw(self) -> Tuple[int, int]:
        # infer expected H,W from engine binding shape
        shp = tuple(self.engine.get_binding_shape(self.input_idx))
        # dynamic? -> use opt shape if possible
        if any(d < 0 for d in shp):
            prof = self._profile_shape(self.input_idx)
            if prof:
                _, opt, _ = prof
                shp = tuple(opt)
        # NCHW
        if len(shp) == 4:
            return int(shp[2]), int(shp[3])
        raise ValueError("Unsupported input shape: " + str(shp))

    def infer(self, input_nchw: np.ndarray) -> List[np.ndarray]:
        assert self.ok, "TRTRunner not initialized"
        assert input_nchw.ndim == 4, "input must be NCHW"
        assert input_nchw.dtype == np.float32, "input must be float32"

        # set dynamic shape if needed
        in_shape = tuple(input_nchw.shape)
        if any(d < 0 for d in self.engine.get_binding_shape(self.input_idx)):
            # validate against profile range
            prof = self._profile_shape(self.input_idx)
            if prof:
                mn, _, mx = prof
                for i, d in enumerate(in_shape):
                    if d < mn[i] or d > mx[i]:
                        raise ValueError(f"[TRT] Input shape {in_shape} out of profile range {mn}..{mx}")
            self.context.set_binding_shape(self.input_idx, in_shape)

        # allocate for input/output based on context shapes
        self._ensure_alloc(self.input_idx, in_shape)

        outputs = []
        bindings = [None] * self.engine.num_bindings

        bindings[self.input_idx] = int(self.device_alloc[self.input_idx])

        # outputs
        for out_i in self.output_idxs:
            out_shape = tuple(self.context.get_binding_shape(out_i))
            # some engines return (-1) even after set shape, try engine binding shape fallback
            if any(d < 0 for d in out_shape):
                out_shape = tuple(self.engine.get_binding_shape(out_i))
            self._ensure_alloc(out_i, out_shape)
            bindings[out_i] = int(self.device_alloc[out_i])

        # H2D input
        np.copyto(self.host_alloc[self.input_idx], input_nchw.ravel())
        self.cuda.memcpy_htod_async(self.device_alloc[self.input_idx], self.host_alloc[self.input_idx], self.stream)

        # execute
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # D2H outputs
        for out_i in self.output_idxs:
            self.cuda.memcpy_dtoh_async(self.host_alloc[out_i], self.device_alloc[out_i], self.stream)

        self.stream.synchronize()

        for out_i in self.output_idxs:
            out_shape = tuple(self.context.get_binding_shape(out_i))
            if any(d < 0 for d in out_shape):
                out_shape = tuple(self.engine.get_binding_shape(out_i))
            out = np.array(self.host_alloc[out_i], copy=True).reshape(out_shape)
            outputs.append(out)

        return outputs

# -----------------------------
# Detector (YOLO-ish)
# -----------------------------
@dataclass
class DetResult:
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2
    conf: float

class PlateDetector:
    def __init__(self, det_engine: Optional[str], det_onnx: Optional[str], img_size=640, conf=0.25, nms_thres=0.45):
        self.img_size = img_size
        self.conf = conf
        self.nms = nms_thres

        self.trt = None
        self.net = None

        if det_engine and os.path.exists(det_engine):
            print("[DET] Using TensorRT engine:", det_engine)
            self.trt = TRTRunner(det_engine)
            if not self.trt.ok:
                print("[DET] TRT init failed, fallback OpenCV DNN")
                self.trt = None

        if self.trt is None:
            if det_onnx and os.path.exists(det_onnx):
                print("[DET] Using OpenCV DNN (CPU):", det_onnx)
                self.net = cv2.dnn.readNetFromONNX(det_onnx)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                raise FileNotFoundError("No detector engine/onnx found")

    def _decode_yolo(self, out: np.ndarray) -> np.ndarray:
        # Normalize output shapes to [N, M]
        # Common: [1,25200,85] or [1,25200,6] or [1,84,8400]
        o = out
        o = np.squeeze(o)
        if o.ndim == 3:
            o = np.squeeze(o)
        if o.ndim == 2:
            # could be [N,M] or [M,N]
            if o.shape[0] < o.shape[1] and o.shape[0] in (6, 7, 8, 84, 85):
                o = o.T
            return o
        if o.ndim == 1:
            return o.reshape(1, -1)
        raise ValueError("Unsupported detector output shape: " + str(out.shape))

    def detect(self, frame_bgr: np.ndarray) -> List[DetResult]:
        im, r, (dw, dh) = letterbox(frame_bgr, (self.img_size, self.img_size))
        blob = im[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        blob = np.transpose(blob, (2, 0, 1))[None, ...]    # NCHW

        if self.trt:
            outs = self.trt.infer(blob)
            out = outs[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()

        pred = self._decode_yolo(out)  # [N,M]
        if pred.shape[1] < 6:
            return []

        # If format is [x,y,w,h,conf,cls] or [x1,y1,x2,y2,conf,cls]
        data = pred
        # If includes class scores (>= 7 or >= 85)
        if data.shape[1] > 6:
            # YOLOv5: [x,y,w,h,obj,cls1..]
            obj = data[:, 4:5]
            cls_scores = data[:, 5:]
            cls_conf = cls_scores.max(axis=1, keepdims=True)
            conf = (obj * cls_conf).squeeze(1)
            # choose the best class only
            data6 = np.concatenate([data[:, :4], conf[:, None], np.zeros((data.shape[0], 1), dtype=np.float32)], axis=1)
        else:
            data6 = data[:, :6].copy()

        # conf filter
        confs = data6[:, 4]
        keep = confs >= self.conf
        data6 = data6[keep]
        confs = confs[keep]
        if data6.shape[0] == 0:
            return []

        # detect if coords are normalized / xywh
        coords = data6[:, :4]
        # if coords max <= 1.5 -> normalized xywh
        if coords.max() <= 1.5:
            coords[:, 0] *= self.img_size
            coords[:, 1] *= self.img_size
            coords[:, 2] *= self.img_size
            coords[:, 3] *= self.img_size
            boxes = xywh2xyxy(coords)
        else:
            # heuristic: if x2 > x1 likely xyxy; else xywh
            if np.mean(coords[:, 2] > coords[:, 0]) > 0.7:
                boxes = coords
            else:
                boxes = xywh2xyxy(coords)

        # NMS
        keep_idx = nms(boxes, confs, self.nms)
        boxes = boxes[keep_idx]
        confs = confs[keep_idx]

        # map back from letterbox to original
        results = []
        h0, w0 = frame_bgr.shape[:2]
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = b
            x1 = (x1 - dw) / r
            x2 = (x2 - dw) / r
            y1 = (y1 - dh) / r
            y2 = (y2 - dh) / r

            x1 = int(clamp(x1, 0, w0 - 1))
            y1 = int(clamp(y1, 0, h0 - 1))
            x2 = int(clamp(x2, 0, w0 - 1))
            y2 = int(clamp(y2, 0, h0 - 1))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            results.append(DetResult((x1, y1, x2, y2), float(c)))
        return results

# -----------------------------
# OCR (CTC greedy + VN plate formatting)
# -----------------------------
def build_charset(C: int) -> Tuple[str, int]:
    """
    Return (charset, blank_index_mode)
    blank_index_mode:
      0 -> blank = 0
      1 -> blank = C-1
    """
    # common charsets
    charset1 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
    charset2 = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-."  # drop I,O
    # choose best length match
    for cs in (charset1, charset2):
        if len(cs) + 1 == C:
            return cs, 0  # assume blank=0 common in many CTC exports
        if len(cs) + 1 == C:
            return cs, 1

    # fallback: make a charset of length C-1
    base = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
    cs = (base * 10)[: max(1, C - 1)]
    return cs, 0

def ctc_greedy_decode(logits: np.ndarray) -> str:
    """
    logits: [T,C] or [1,T,C] or [T,1,C]
    """
    x = logits
    x = np.squeeze(x)
    if x.ndim == 3:
        x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError("OCR logits must be 2D after squeeze, got " + str(logits.shape))
    T, C = x.shape
    charset, blank_mode = build_charset(C)

    # probabilities or logits -> argmax
    pred = x.argmax(axis=1).astype(np.int32)

    # try blank=0 and blank=C-1, pick plausible plate
    def decode_with_blank(blank_idx: int) -> str:
        out = []
        prev = -1
        for p in pred.tolist():
            if p == prev:
                continue
            prev = p
            if p == blank_idx:
                continue
            ci = p - 1 if blank_idx == 0 else p
            if 0 <= ci < len(charset):
                out.append(charset[ci])
        return "".join(out)

    s0 = decode_with_blank(0)
    s1 = decode_with_blank(C - 1)

    def score_plate(s: str) -> int:
        # prefer strings containing digits and reasonable length
        if not s:
            return -999
        sc = 0
        sc += 2 * sum(ch.isdigit() for ch in s)
        sc += 1 * sum(ch.isalpha() for ch in s)
        if 6 <= len(s) <= 12:
            sc += 5
        # penalize too many repeats
        if re.search(r"(.)\1\1\1", s):
            sc -= 10
        return sc

    return s0 if score_plate(s0) >= score_plate(s1) else s1

def normalize_plate_text(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[^0-9A-Z\-.]", "", s)
    # VN style: XX-YY ZZZ.ZZ or similar; keep as is but remove duplicate dots/dashes
    s = re.sub(r"-{2,}", "-", s)
    s = re.sub(r"\.{2,}", ".", s)
    return s

class PlateOCR:
    def __init__(self, ocr_engine: Optional[str], ocr_onnx: Optional[str]):
        self.trt = None
        self.net = None

        if ocr_engine and os.path.exists(ocr_engine):
            print("[OCR] Using TensorRT engine:", ocr_engine)
            self.trt = TRTRunner(ocr_engine)
            if not self.trt.ok:
                print("[OCR] TRT init failed, fallback OpenCV DNN")
                self.trt = None

        if self.trt is None:
            if ocr_onnx and os.path.exists(ocr_onnx):
                print("[OCR] Using OpenCV DNN (CPU):", ocr_onnx)
                self.net = cv2.dnn.readNetFromONNX(ocr_onnx)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                raise FileNotFoundError("No OCR engine/onnx found")

    def _preprocess(self, crop_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
        # improve contrast a bit
        img = crop_bgr
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        return img

    def recognize_once(self, crop_bgr: np.ndarray) -> str:
        # get expected H,W from engine if possible
        if self.trt:
            H, W = self.trt.input_hw()
        else:
            # fallback common CRNN size
            H, W = 40, 160

        inp = self._preprocess(crop_bgr, H, W)

        if self.trt:
            outs = self.trt.infer(inp)
            logits = outs[0]
        else:
            self.net.setInput(inp)
            logits = self.net.forward()

        text = ctc_greedy_decode(logits)
        return normalize_plate_text(text)

    def recognize_plate(self, crop_bgr: np.ndarray) -> str:
        h, w = crop_bgr.shape[:2]
        # heuristic: 2-line plate if height relatively large vs width
        if h / (w + 1e-6) > 0.55:
            # split to top/bottom
            mid = int(h * 0.52)
            top = crop_bgr[:mid, :]
            bot = crop_bgr[mid:, :]
            t1 = self.recognize_once(top)
            t2 = self.recognize_once(bot)
            if t1 and t2:
                return f"{t1}-{t2}"
            return t1 or t2
        else:
            return self.recognize_once(crop_bgr)

# -----------------------------
# Video sources
# -----------------------------
def gst_csi(sensor_id=0, w=1280, h=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1 sync=false"
    )

def gst_rtsp(url: str, latency=120):
    # H264 RTSP -> decode NVDEC
    return (
        f"rtspsrc location={url} latency={latency} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1 sync=false"
    )

# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    ap.add_argument("--rtsp", type=str, default="")
    ap.add_argument("--cam", type=int, default=0)

    ap.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")
    ap.add_argument("--det_onnx", type=str, default="model/LP_detector_nano_61.onnx")
    ap.add_argument("--ocr_onnx", type=str, default="model/LP_ocr_nano_62.onnx")

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)

    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--sensor_id", type=int, default=0)
    ap.add_argument("--flip", type=int, default=0)

    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()

    detector = PlateDetector(args.det_engine, args.det_onnx, img_size=args.imgsz, conf=args.conf, nms_thres=args.nms)
    ocr = PlateOCR(args.ocr_engine, args.ocr_onnx)

    # open video
    if args.source == "csi":
        src = gst_csi(args.sensor_id, args.csi_w, args.csi_h, args.csi_fps, args.flip)
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        if not args.rtsp:
            raise SystemExit("Missing --rtsp URL")
        src = gst_rtsp(args.rtsp)
        cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open source={args.source}")

    print(f"[RUN] source={args.source} show={args.show} imgsz={args.imgsz}")

    last_t = time.time()
    fps = 0.0
    fail_count = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_count += 1
            print("[WARN] frame read failed", fail_count)
            time.sleep(0.1)
            if fail_count >= 10:
                print("[WARN] reopen capture...")
                cap.release()
                time.sleep(0.5)
                if args.source == "csi":
                    cap = cv2.VideoCapture(gst_csi(args.sensor_id, args.csi_w, args.csi_h, args.csi_fps, args.flip), cv2.CAP_GSTREAMER)
                elif args.source == "rtsp":
                    cap = cv2.VideoCapture(gst_rtsp(args.rtsp), cv2.CAP_GSTREAMER)
                else:
                    cap = cv2.VideoCapture(args.cam)
                fail_count = 0
            continue
        fail_count = 0

        t0 = now_ms()
        dets = detector.detect(frame)
        plates = 0

        for det in dets:
            x1, y1, x2, y2 = det.box
            # pad a bit
            pw = int((x2 - x1) * 0.06)
            ph = int((y2 - y1) * 0.12)
            x1p = clamp(x1 - pw, 0, frame.shape[1]-1)
            y1p = clamp(y1 - ph, 0, frame.shape[0]-1)
            x2p = clamp(x2 + pw, 0, frame.shape[1]-1)
            y2p = clamp(y2 + ph, 0, frame.shape[0]-1)

            crop = frame[int(y1p):int(y2p), int(x1p):int(x2p)].copy()
            text = ""
            try:
                text = ocr.recognize_plate(crop)
            except Exception as e:
                text = ""
                # vẫn vẽ bbox, nhưng không crash
                # print("[OCR] error:", e)

            if text:
                plates += 1

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{text}" if text else "plate"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # FPS
        dt = time.time() - last_t
        if dt > 0:
            fps = 1.0 / dt
        last_t = time.time()
        cv2.putText(frame, f"FPS {fps:.1f} plates={plates}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.show:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
