#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALPR Jetson (CSI / RTSP / USB webcam)

Mục tiêu:
- Detect biển số + OCR ra TEXT chuẩn
- Ưu tiên TensorRT .engine (nhanh). Fallback OpenCV DNN (chậm).

Mặc định tìm model trong ./model/
  - LP_detector_nano_61_fp16.engine (detector)
  - LP_ocr_nano_62_fp16.engine      (ocr)

Chạy:
  python3 webcam_onnx.py --source csi --show
  python3 webcam_onnx.py --source rtsp --rtsp rtsp://192.168.50.2:8554/mac --show
  python3 webcam_onnx.py --source webcam --cam 0 --show

Tip FPS:
  - --det_every 2  (detect 1 lần/2 frame)
  - --ocr_every 5  (ocr 1 lần/5 frame)
"""

import os
import re
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

# ===== Optional TensorRT =====
TRT_OK = True
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    cuda.init()
except Exception as e:
    TRT_OK = False
    trt = None
    cuda = None


# ==========================
# Utils
# ==========================
def now_ms() -> float:
    return time.time() * 1000.0


def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_th=0.45) -> List[int]:
    """boxes: Nx4 (x1,y1,x2,y2)"""
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
        keep.append(int(i))
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
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return keep


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad giống YOLO. Return: img, ratio, (dw, dh)."""
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w, h
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


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ==========================
# TensorRT wrapper
# ==========================
@dataclass
class TrtBinding:
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    is_input: bool


class TrtEngine:
    def __init__(self, engine_path: str, device_id: int = 0):
        if not TRT_OK:
            raise RuntimeError("TensorRT/pycuda not available in this environment")

        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self.device = cuda.Device(device_id)
        # NOTE: dùng primary context đã được init; tránh make_context() để giảm crash trong Docker
        self.bindings_meta: List[TrtBinding] = []
        self._build_meta()

        self.host_inputs = {}
        self.dev_inputs = {}
        self.host_outputs = {}
        self.dev_outputs = {}

    def _build_meta(self):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)
            shape = tuple(self.engine.get_binding_shape(i))
            self.bindings_meta.append(TrtBinding(name=name, dtype=dtype, shape=shape, is_input=is_input))

    def _alloc_if_needed(self):
        # allocate buffers based on current binding shapes in context
        for i, b in enumerate(self.bindings_meta):
            if self.engine.is_shape_binding(i):
                continue

            shape = tuple(self.context.get_binding_shape(i))
            # dynamic shape might be -1 before set; ignore until valid
            if any(d < 0 for d in shape):
                continue

            size = int(np.prod(shape))
            dtype = b.dtype
            nbytes = size * np.dtype(dtype).itemsize

            if b.is_input:
                if b.name not in self.host_inputs or self.host_inputs[b.name].nbytes != nbytes:
                    self.host_inputs[b.name] = cuda.pagelocked_empty(size, dtype)
                    self.dev_inputs[b.name] = cuda.mem_alloc(nbytes)
            else:
                if b.name not in self.host_outputs or self.host_outputs[b.name].nbytes != nbytes:
                    self.host_outputs[b.name] = cuda.pagelocked_empty(size, dtype)
                    self.dev_outputs[b.name] = cuda.mem_alloc(nbytes)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # set binding shapes for dynamic inputs
        for i, b in enumerate(self.bindings_meta):
            if b.is_input and b.name in inputs:
                arr = inputs[b.name]
                if self.engine.is_shape_binding(i):
                    continue
                if self.engine.get_binding_shape(i).count(-1) > 0:
                    self.context.set_binding_shape(i, tuple(arr.shape))

        self._alloc_if_needed()

        bindings_ptrs = [None] * self.engine.num_bindings

        # copy inputs
        for i, b in enumerate(self.bindings_meta):
            if b.is_input:
                if b.name not in inputs:
                    raise ValueError(f"Missing input tensor: {b.name}")
                arr = inputs[b.name].ravel()
                host = self.host_inputs[b.name]
                if host.size != arr.size:
                    raise ValueError(f"Input size mismatch for {b.name}: host {host.size}, arr {arr.size}")
                np.copyto(host, arr)
                cuda.memcpy_htod_async(self.dev_inputs[b.name], host, self.stream)
                bindings_ptrs[i] = int(self.dev_inputs[b.name])
            else:
                bindings_ptrs[i] = int(self.dev_outputs[b.name])

        ok = self.context.execute_async_v2(bindings=bindings_ptrs, stream_handle=int(self.stream.handle))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        outputs = {}
        for i, b in enumerate(self.bindings_meta):
            if not b.is_input:
                host = self.host_outputs[b.name]
                cuda.memcpy_dtoh_async(host, self.dev_outputs[b.name], self.stream)

        self.stream.synchronize()

        for i, b in enumerate(self.bindings_meta):
            if not b.is_input:
                shape = tuple(self.context.get_binding_shape(i))
                outputs[b.name] = np.array(self.host_outputs[b.name]).reshape(shape)

        return outputs

    def input_names(self) -> List[str]:
        return [b.name for b in self.bindings_meta if b.is_input]

    def output_names(self) -> List[str]:
        return [b.name for b in self.bindings_meta if not b.is_input]


# ==========================
# OCR decode (CTC greedy)
# ==========================
# blank index = 0
OCR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# optional thêm '-' '.' nếu model có; nhiều model không cần vì format mình tự chèn
# OCR_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."

def ctc_greedy_decode(logits: np.ndarray) -> str:
    """
    logits shapes could be:
      (1, T, C) or (T, C) or (1, C, T)
    returns raw string without formatting
    """
    x = logits
    x = np.asarray(x)

    if x.ndim == 3:
        # try common patterns
        if x.shape[0] == 1 and x.shape[2] > 8 and x.shape[1] < x.shape[2]:
            # (1, T, C)
            x = x[0]
        elif x.shape[0] == 1 and x.shape[1] == len(OCR_CHARS) + 1:
            # (1, C, T)
            x = x[0].transpose(1, 0)
        else:
            x = x[0]

    if x.ndim != 2:
        return ""

    # x: (T, C)
    seq = np.argmax(x, axis=1).astype(int)

    out = []
    prev = 0
    for s in seq:
        if s != 0 and s != prev:
            idx = s - 1
            if 0 <= idx < len(OCR_CHARS):
                out.append(OCR_CHARS[idx])
        prev = s
    return "".join(out)


def format_vn_plate(s: str) -> str:
    """
    Heuristic format VN plate:
    - Example raw: 63B995164 -> 63-B9 951.64
    - raw: 54S70661 -> 54-S7 066.1? (tuỳ)
    """
    s = re.sub(r"[^0-9A-Z]", "", s.upper())
    if len(s) < 6:
        return s

    # 2 digits province at start
    if len(s) >= 4 and s[0:2].isdigit():
        # common: 2 digits + 1 letter + 1 digit + rest digits
        if len(s) >= 5 and s[2].isalpha() and s[3].isdigit():
            head = f"{s[0:2]}-{s[2:4]}"
            tail = s[4:]
            # tail 5 digits -> 3.2
            if len(tail) == 5 and tail.isdigit():
                return f"{head} {tail[0:3]}.{tail[3:5]}"
            # tail 4 digits -> 4 digits
            if len(tail) == 4 and tail.isdigit():
                return f"{head} {tail}"
            # tail 6 digits -> 3.3
            if len(tail) == 6 and tail.isdigit():
                return f"{head} {tail[0:3]}.{tail[3:6]}"
            return f"{head} {tail}"
    return s


# ==========================
# Detector postprocess (YOLO-ish generic)
# ==========================
def parse_yolo_output(output: np.ndarray, conf_th=0.45) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try parse common YOLO export:
      - (1, N, 6) [x,y,w,h,conf,cls] or [x1,y1,x2,y2,conf,cls]
      - (1, N, 5+nc) [x,y,w,h,obj,cls...]
      - (1, 85, N) etc -> transpose to (N,85)
    Returns boxes_xywh (N,4) in input space (640x640) and scores (N,)
    """
    out = np.asarray(output)

    # squeeze batch
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    # handle (C, N)
    if out.ndim == 2 and out.shape[0] < out.shape[1] and out.shape[0] in (6, 7, 84, 85, 86):
        out = out.transpose(1, 0)

    if out.ndim != 2:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if out.shape[1] == 6:
        boxes = out[:, 0:4].astype(np.float32)
        conf = out[:, 4].astype(np.float32)
        # some exports are xyxy, some are xywh. we guess by check if x2>x1 for most
        if np.mean(boxes[:, 2] > boxes[:, 0]) > 0.9 and np.mean(boxes[:, 3] > boxes[:, 1]) > 0.9:
            # seems xyxy -> convert to xywh
            xyxy = boxes
            boxes = np.stack([
                (xyxy[:, 0] + xyxy[:, 2]) / 2,
                (xyxy[:, 1] + xyxy[:, 3]) / 2,
                (xyxy[:, 2] - xyxy[:, 0]),
                (xyxy[:, 3] - xyxy[:, 1]),
            ], axis=1)
        mask = conf >= conf_th
        return boxes[mask], conf[mask]

    if out.shape[1] >= 5:
        boxes = out[:, 0:4].astype(np.float32)  # usually xywh
        obj = out[:, 4].astype(np.float32)
        if out.shape[1] > 5:
            cls_scores = out[:, 5:].astype(np.float32)
            if cls_scores.shape[1] == 1:
                conf = obj * cls_scores[:, 0]
            else:
                conf = obj * np.max(cls_scores, axis=1)
        else:
            conf = obj
        mask = conf >= conf_th
        return boxes[mask], conf[mask]

    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


# ==========================
# Video sources
# ==========================
def gst_csi(sensor_id=0, w=1280, h=720, fps=30) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def gst_rtsp(url: str, latency=200) -> str:
    # HW decode with nvv4l2decoder (best on Jetson). fallback will be handled by OpenCV if fail.
    return (
        f"rtspsrc location={url} latency={latency} ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


# ==========================
# ALPR Pipeline
# ==========================
class ALPR:
    def __init__(
        self,
        det_engine: Optional[str],
        ocr_engine: Optional[str],
        det_onnx: Optional[str],
        ocr_onnx: Optional[str],
        conf=0.45,
        nms=0.45,
        input_size=640,
        ocr_w=160,
        ocr_h=40,
        use_fp16=True,
        max_plates=3,
    ):
        self.conf = conf
        self.nms = nms
        self.input_size = int(input_size)
        self.ocr_w = int(ocr_w)
        self.ocr_h = int(ocr_h)
        self.max_plates = int(max_plates)

        self.det_trt = None
        self.ocr_trt = None
        self.det_net = None
        self.ocr_net = None

        self.det_input_name = None
        self.ocr_input_name = None

        # --- Load detector ---
        if det_engine and os.path.exists(det_engine) and TRT_OK:
            self.det_trt = TrtEngine(det_engine)
            self.det_input_name = self.det_trt.input_names()[0]
        elif det_onnx and os.path.exists(det_onnx):
            self.det_net = cv2.dnn.readNet(det_onnx)
            self.det_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.det_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # --- Load OCR ---
        if ocr_engine and os.path.exists(ocr_engine) and TRT_OK:
            self.ocr_trt = TrtEngine(ocr_engine)
            self.ocr_input_name = self.ocr_trt.input_names()[0]
        elif ocr_onnx and os.path.exists(ocr_onnx):
            self.ocr_net = cv2.dnn.readNet(ocr_onnx)
            self.ocr_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.ocr_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def preprocess_det(self, frame_bgr: np.ndarray):
        img, r, (dw, dh) = letterbox(frame_bgr, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW
        return x, r, dw, dh

    def postprocess_det(self, det_out: np.ndarray, r, dw, dh, orig_w, orig_h):
        xywh, scores = parse_yolo_output(det_out, conf_th=self.conf)
        if xywh.shape[0] == 0:
            return []

        boxes = xywh_to_xyxy(xywh)

        # map back to padded image -> original
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :] /= r

        # clip
        out = []
        for b, s in zip(boxes, scores):
            bb = clip_box(b, orig_w, orig_h)
            if bb is None:
                continue
            out.append((bb, float(s)))

        if not out:
            return []

        bxs = np.array([o[0] for o in out], dtype=np.float32)
        scs = np.array([o[1] for o in out], dtype=np.float32)
        keep = nms_xyxy(bxs, scs, iou_th=self.nms)
        out = [out[i] for i in keep]

        # sort by score
        out.sort(key=lambda x: x[1], reverse=True)
        return out[: self.max_plates]

    def detect(self, frame_bgr: np.ndarray):
        orig_h, orig_w = frame_bgr.shape[:2]
        x, r, dw, dh = self.preprocess_det(frame_bgr)

        # infer
        if self.det_trt is not None:
            dtype = np.float16 if self.det_trt.bindings_meta[0].dtype == np.float16 else np.float32
            x_in = x.astype(dtype)
            outs = self.det_trt.infer({self.det_input_name: x_in})
            # pick first output
            det_out = outs[self.det_trt.output_names()[0]]
        elif self.det_net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame_bgr, (self.input_size, self.input_size)),
                1 / 255.0,
                (self.input_size, self.input_size),
                swapRB=True,
                crop=False,
            )
            self.det_net.setInput(blob)
            det_out = self.det_net.forward()
        else:
            return []

        return self.postprocess_det(det_out, r, dw, dh, orig_w, orig_h)

    def preprocess_ocr(self, plate_bgr: np.ndarray):
        # resize to (ocr_w, ocr_h)
        img = cv2.resize(plate_bgr, (self.ocr_w, self.ocr_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xH xW
        return x

    def ocr(self, plate_bgr: np.ndarray) -> str:
        x = self.preprocess_ocr(plate_bgr)

        if self.ocr_trt is not None:
            dtype = np.float16 if self.ocr_trt.bindings_meta[0].dtype == np.float16 else np.float32
            x_in = x.astype(dtype)
            outs = self.ocr_trt.infer({self.ocr_input_name: x_in})
            o = outs[self.ocr_trt.output_names()[0]]
            raw = ctc_greedy_decode(o)
            return format_vn_plate(raw)

        if self.ocr_net is not None:
            self.ocr_net.setInput(x.astype(np.float32))
            o = self.ocr_net.forward()
            raw = ctc_greedy_decode(o)
            return format_vn_plate(raw)

        return ""

    def draw_overlay(self, frame, box, text, score):
        x1, y1, x2, y2 = box
        # tô đỏ nhẹ ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            overlay = roi.copy()
            overlay[:, :, 2] = np.clip(overlay[:, :, 2] + 60, 0, 255)
            frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.35, roi, 0.65, 0)

        # viền xanh
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = text if text else f"plate {score:.2f}"
        # chữ đỏ + outline đen cho dễ đọc
        org = (x1, max(0, y1 - 10))
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    ap.add_argument("--rtsp", default="", help="RTSP URL (if source=rtsp)")
    ap.add_argument("--cam", type=int, default=0, help="USB webcam index (if source=webcam)")

    ap.add_argument("--det_engine", default="./model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", default="./model/LP_ocr_nano_62_fp16.engine")
    ap.add_argument("--det_onnx", default="./model/LP_detector_nano_61.onnx")
    ap.add_argument("--ocr_onnx", default="./model/LP_ocr_nano_62.onnx")

    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--nms", type=float, default=0.45)

    ap.add_argument("--csi_id", type=int, default=0)
    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)

    ap.add_argument("--rtsp_latency", type=int, default=200)

    ap.add_argument("--det_every", type=int, default=1, help="detect every N frames")
    ap.add_argument("--ocr_every", type=int, default=3, help="ocr every N frames per plate")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max_plates", type=int, default=3)
    args, extra = ap.parse_known_args()

    # allow positional rtsp url
    if args.source == "rtsp" and not args.rtsp and extra:
        if extra[0].startswith("rtsp://"):
            args.rtsp = extra[0]

    alpr = ALPR(
        det_engine=args.det_engine,
        ocr_engine=args.ocr_engine,
        det_onnx=args.det_onnx,
        ocr_onnx=args.ocr_onnx,
        conf=args.conf,
        nms=args.nms,
        input_size=640,
        ocr_w=160,
        ocr_h=40,
        max_plates=args.max_plates,
    )

    # open video
    cap = None
    if args.source == "csi":
        pipeline = gst_csi(args.csi_id, args.csi_w, args.csi_h, args.csi_fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        if not args.rtsp:
            raise SystemExit("Missing --rtsp URL")
        pipeline = gst_rtsp(args.rtsp, args.rtsp_latency)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # fallback: try OpenCV direct
            cap = cv2.VideoCapture(args.rtsp)
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise SystemExit("Cannot open video source")

    # cache OCR results to avoid spam
    last_text = {}  # key: box idx -> (text, frame_id)

    fps_t0 = time.time()
    fps_cnt = 0
    frame_id = 0
    det_results = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_id += 1

        if frame_id % max(1, args.det_every) == 0:
            try:
                det_results = alpr.detect(frame)
            except Exception as e:
                print("[DET ERROR]", e)
                det_results = []

        # OCR & draw
        plates_found = 0
        for i, (box, score) in enumerate(det_results):
            x1, y1, x2, y2 = box
            pad = 3
            crop = frame[max(0, y1 - pad):min(frame.shape[0], y2 + pad),
                         max(0, x1 - pad):min(frame.shape[1], x2 + pad)]

            text = ""
            # ocr only every N frames to reduce lag
            if crop.size > 0:
                if (i not in last_text) or (frame_id - last_text[i][1] >= max(1, args.ocr_every)):
                    try:
                        text = alpr.ocr(crop)
                        last_text[i] = (text, frame_id)
                    except Exception as e:
                        print("[OCR ERROR]", e)
                        text = last_text.get(i, ("", frame_id))[0]
                else:
                    text = last_text[i][0]

            if text:
                plates_found += 1
            alpr.draw_overlay(frame, box, text, score)

        fps_cnt += 1
        dt = time.time() - fps_t0
        fps = fps_cnt / dt if dt > 0 else 0.0

        cv2.putText(frame, f"FPS {fps:.1f} plates={plates_found}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        if args.show:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
