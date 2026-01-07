#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALPR runner for Jetson (CSI/RTSP/Webcam) using TensorRT engines.

Defaults (relative to project root):
  ./model/LP_detector_nano_61_fp16.engine
  ./model/LP_ocr_nano_62_fp16.engine

If engines are missing:
  TensorRT -> ONNXRuntime -> OpenCV DNN CPU (last resort).

Overlay:
- Fill plate region with translucent red
- Thin green border
- Plate text in red

Run (inside container):
  python3 webcam_onnx.py --source csi --show
  python3 webcam_onnx.py --source rtsp --rtsp rtsp://192.168.50.2:8554/mac --show
  python3 webcam_onnx.py --source webcam --cam 0 --show
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Utils
# ----------------------------
def now_ms() -> float:
    return time.time() * 1000.0


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize + pad to keep aspect ratio (YOLO-style). Returns (img, ratio, (dw, dh))"""
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    im = cv2.copyMakeBorder(im, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def nms_boxes(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Pure numpy NMS."""
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
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

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


# ----------------------------
# CTC decode for OCR
# ----------------------------
DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # common VN plates + letters


def ctc_greedy_decode(logits: np.ndarray, charset: str = DEFAULT_CHARSET, blank_index: Optional[int] = None) -> str:
    """
    logits: can be (T,C) or (1,T,C) or (C,T) etc.
    We do argmax, remove repeats and blanks (CTC collapse).
    """
    if blank_index is None:
        blank_index = len(charset)

    x = logits
    x = np.asarray(x)

    # squeeze batch
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    # if shape is (C,T) -> transpose to (T,C)
    if x.ndim == 2 and x.shape[0] <= x.shape[1] and x.shape[0] <= 128:
        # heuristic: classes usually <= ~80, timesteps often > classes
        # but sometimes timesteps smaller. We'll detect by comparing to charset length.
        if x.shape[0] == (len(charset) + 1):
            x = x.T

    if x.ndim != 2:
        # can't decode
        return ""

    # argmax over classes
    pred = np.argmax(x, axis=1).astype(np.int32)

    # CTC collapse
    out = []
    prev = -1
    for p in pred:
        if p == prev:
            continue
        prev = p
        if p == blank_index:
            continue
        if 0 <= p < len(charset):
            out.append(charset[p])
    return "".join(out)


def normalize_plate_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^0-9A-Z]", "", s)
    return s


def format_vn_plate(s: str) -> str:
    """
    Heuristic VN formatting.
    Example:
      63B995164 -> 63-B9 951.64
      54S70661  -> 54-S7 066.1? (not perfect, but better)
    If can't, return raw.
    """
    s = normalize_plate_text(s)
    if len(s) == 9:
        return f"{s[0:2]}-{s[2:4]} {s[4:7]}.{s[7:9]}"
    if len(s) == 8:
        # 2 + 2 + 4 (common)
        return f"{s[0:2]}-{s[2:4]} {s[4:8]}"
    if len(s) == 7:
        # 2 + 1 + 4
        return f"{s[0:2]}{s[2]}-{s[3:7]}"
    return s


# ----------------------------
# TensorRT inference
# ----------------------------
@dataclass
class TrtIO:
    host: np.ndarray
    device: "cuda.DeviceAllocation"
    shape: Tuple[int, ...]
    dtype: np.dtype


class TRTModel:
    def __init__(self, engine_path: str, verbose: bool = False):
        self.engine_path = engine_path
        self.verbose = verbose

        self._trt = None
        self._cuda = None
        self._engine = None
        self._context = None
        self._stream = None
        self._bindings = None
        self._inputs: List[TrtIO] = []
        self._outputs: List[TrtIO] = []
        self._profile_index = 0

        self._load()

    def _load(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"TensorRT/PyCUDA not available: {e}")

        self._trt = trt
        self._cuda = cuda

        logger = trt.Logger(trt.Logger.VERBOSE if self.verbose else trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create execution context")

        self._stream = cuda.Stream()

        # Select profile 0 by default
        if self._engine.num_optimization_profiles > 0:
            self._profile_index = 0
            try:
                self._context.active_optimization_profile = 0
            except Exception:
                pass

        self._allocate_buffers()

    def _get_profile_shape(self, binding_idx: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """(min,opt,max) shapes for a binding (profile 0)."""
        eng = self._engine
        if eng.num_optimization_profiles <= 0:
            s = tuple(eng.get_binding_shape(binding_idx))
            return s, s, s

        try:
            mn, opt, mx = eng.get_profile_shape(self._profile_index, binding_idx)
            return tuple(mn), tuple(opt), tuple(mx)
        except Exception:
            s = tuple(eng.get_binding_shape(binding_idx))
            return s, s, s

    def get_input_hw(self) -> Tuple[int, int]:
        """Return (H,W) of first input binding, using OPT profile if dynamic."""
        # Find first input
        for i in range(self._engine.num_bindings):
            if self._engine.binding_is_input(i):
                mn, opt, mx = self._get_profile_shape(i)
                shp = opt
                # For NCHW
                if len(shp) == 4:
                    return int(shp[2]), int(shp[3])
                # fallback
                s = self._engine.get_binding_shape(i)
                if len(s) == 4:
                    return int(s[2]), int(s[3])
        return 640, 640

    def _allocate_buffers(self):
        cuda = self._cuda
        eng = self._engine
        ctx = self._context

        self._inputs.clear()
        self._outputs.clear()

        # For dynamic shapes: set to OPT shapes before allocation
        for i in range(eng.num_bindings):
            if eng.binding_is_input(i):
                mn, opt, mx = self._get_profile_shape(i)
                if -1 in opt:
                    # try to use max as safe
                    opt = mx
                try:
                    ctx.set_binding_shape(i, opt)
                except Exception:
                    pass

        bindings = []
        for i in range(eng.num_bindings):
            dtype = np.dtype(self._trt.nptype(eng.get_binding_dtype(i)))
            shape = tuple(ctx.get_binding_shape(i))
            if any(d < 0 for d in shape):
                # last fallback
                shape = tuple(eng.get_binding_shape(i))
            size = int(np.prod(shape)) if len(shape) else 1
            host_mem = cuda.pagelocked_empty(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            io = TrtIO(host=host_mem, device=device_mem, shape=shape, dtype=dtype)
            if eng.binding_is_input(i):
                self._inputs.append(io)
            else:
                self._outputs.append(io)

        self._bindings = bindings

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        input_tensor must match first input binding shape (NCHW).
        Returns list of output tensors as numpy arrays with shapes from binding.
        """
        cuda = self._cuda
        ctx = self._context

        if not self._inputs:
            raise RuntimeError("No input bindings found in engine")

        inp = self._inputs[0]
        expected_shape = inp.shape
        if tuple(input_tensor.shape) != tuple(expected_shape):
            # Try set binding shape for dynamic engines
            try:
                # find first input binding index
                for i in range(self._engine.num_bindings):
                    if self._engine.binding_is_input(i):
                        ctx.set_binding_shape(i, tuple(input_tensor.shape))
                        break
                # re-allocate if shape changed
                if tuple(ctx.get_binding_shape(i)) != tuple(inp.shape):
                    self._allocate_buffers()
                    inp = self._inputs[0]
                    expected_shape = inp.shape
            except Exception:
                pass

        if tuple(input_tensor.shape) != tuple(expected_shape):
            raise RuntimeError(f"Input shape mismatch. Got {input_tensor.shape}, expected {expected_shape}")

        # Copy input to pagelocked
        np.copyto(inp.host, input_tensor.ravel())

        # H2D
        cuda.memcpy_htod_async(inp.device, inp.host, self._stream)

        # Execute
        ctx.execute_async_v2(bindings=self._bindings, stream_handle=int(self._stream.handle))

        # D2H
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self._stream)

        self._stream.synchronize()

        outs = []
        for out in self._outputs:
            arr = np.array(out.host, dtype=out.dtype).reshape(out.shape)
            outs.append(arr)
        return outs


# ----------------------------
# Detector + OCR wrappers
# ----------------------------
class PlateDetector:
    def __init__(self, det_engine: str, det_conf: float = 0.35, nms_iou: float = 0.45, verbose: bool = False):
        self.det_conf = float(det_conf)
        self.nms_iou = float(nms_iou)
        self.verbose = verbose

        self.trt = TRTModel(det_engine, verbose=verbose)
        self.inp_h, self.inp_w = self.trt.get_input_hw()

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        img, r, (dw, dh) = letterbox(frame_bgr, new_shape=(self.inp_h, self.inp_w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, 0)  # NCHW
        return img, r, (dw, dh)

    def _postprocess(self, raw: np.ndarray, r: float, dwdh: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float]]:
        """
        Support common YOLO outputs:
        - (1,N,6) or (N,6): [x1,y1,x2,y2,score,class]
        - (1,N,85) or (N,85): YOLOv5 raw [cx,cy,w,h,obj, cls...]
        Only one class assumed: plate.
        """
        h0, w0 = orig_shape
        dw, dh = dwdh

        out = raw
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        if out.ndim == 2 and out.shape[1] == 6:
            boxes = out[:, 0:4].astype(np.float32)
            scores = out[:, 4].astype(np.float32)

            # heuristic: normalized?
            if boxes.max() <= 2.0:
                boxes[:, [0, 2]] *= self.inp_w
                boxes[:, [1, 3]] *= self.inp_h

            # map back from letterbox
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= r

            boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

            keep = np.where(scores >= self.det_conf)[0]
            boxes = boxes[keep]
            scores = scores[keep]
            if boxes.size == 0:
                return []

            keep_idx = nms_boxes(boxes, scores, self.nms_iou)
            res = []
            for i in keep_idx:
                x1, y1, x2, y2 = boxes[i].tolist()
                res.append((int(x1), int(y1), int(x2), int(y2), float(scores[i])))
            return res

        # YOLO raw
        if out.ndim == 2 and out.shape[1] >= 6:
            # assume [cx,cy,w,h,obj,cls...]
            cxcywh = out[:, 0:4].astype(np.float32)
            obj = out[:, 4].astype(np.float32)

            if out.shape[1] > 6:
                cls_conf = out[:, 5:].max(axis=1).astype(np.float32)
            else:
                cls_conf = out[:, 5].astype(np.float32)

            scores = obj * cls_conf
            keep = np.where(scores >= self.det_conf)[0]
            if keep.size == 0:
                return []
            cxcywh = cxcywh[keep]
            scores = scores[keep]

            # if normalized
            if cxcywh.max() <= 2.0:
                cxcywh[:, 0] *= self.inp_w
                cxcywh[:, 1] *= self.inp_h
                cxcywh[:, 2] *= self.inp_w
                cxcywh[:, 3] *= self.inp_h

            boxes = np.zeros((cxcywh.shape[0], 4), dtype=np.float32)
            boxes[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
            boxes[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
            boxes[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
            boxes[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

            # map back from letterbox
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= r

            boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

            keep_idx = nms_boxes(boxes, scores, self.nms_iou)
            res = []
            for i in keep_idx:
                x1, y1, x2, y2 = boxes[i].tolist()
                res.append((int(x1), int(y1), int(x2), int(y2), float(scores[i])))
            return res

        return []

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        inp, r, dwdh = self._preprocess(frame_bgr)
        outs = self.trt.infer(inp.astype(np.float32))
        raw = outs[0]
        return self._postprocess(raw, r, dwdh, orig_shape=frame_bgr.shape[:2])


class PlateOCR:
    def __init__(self, ocr_engine: str, charset: str = DEFAULT_CHARSET, verbose: bool = False):
        self.charset = charset
        self.verbose = verbose

        self.trt = TRTModel(ocr_engine, verbose=verbose)
        self.inp_h, self.inp_w = self.trt.get_input_hw()

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.inp_w, self.inp_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, 0)  # NCHW
        return img

    def recognize(self, crop_bgr: np.ndarray) -> str:
        inp = self._preprocess(crop_bgr)
        outs = self.trt.infer(inp.astype(np.float32))
        logits = outs[0]

        # Decode CTC
        txt = ctc_greedy_decode(logits, charset=self.charset, blank_index=len(self.charset))
        txt = format_vn_plate(txt)
        return txt


# ----------------------------
# Video sources
# ----------------------------
def gst_csi_pipeline(sensor_id: int, w: int, h: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def open_capture(args) -> cv2.VideoCapture:
    if args.source == "csi":
        pipe = gst_csi_pipeline(args.sensor_id, args.csi_w, args.csi_h, args.csi_fps)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    if args.source == "rtsp":
        if not args.rtsp:
            raise ValueError("RTSP source selected but --rtsp is empty")
        # simplest (works). If you want lower latency, switch to GStreamer later.
        cap = cv2.VideoCapture(args.rtsp)
        return cap

    # webcam
    cap = cv2.VideoCapture(int(args.cam))
    return cap


# ----------------------------
# Drawing
# ----------------------------
def draw_plate(frame: np.ndarray, box: Tuple[int, int, int, int], text: str, score: float, alpha: float = 0.25):
    x1, y1, x2, y2 = box

    # red fill (translucent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # thin green border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # red text
    label = text if text else f"{score:.2f}"
    ty = y1 - 8 if y1 - 8 > 10 else y2 + 20
    cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    p.add_argument("--rtsp", type=str, default="", help="RTSP URL (when --source rtsp)")
    p.add_argument("--cam", type=int, default=0, help="Webcam index (when --source webcam)")

    p.add_argument("--det_engine", type=str, default="./model/LP_detector_nano_61_fp16.engine")
    p.add_argument("--ocr_engine", type=str, default="./model/LP_ocr_nano_62_fp16.engine")

    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--nms", type=float, default=0.45)

    p.add_argument("--no_ocr", action="store_true", help="Only detect plate, skip OCR")
    p.add_argument("--charset", type=str, default=DEFAULT_CHARSET)

    p.add_argument("--csi_w", type=int, default=1280)
    p.add_argument("--csi_h", type=int, default=720)
    p.add_argument("--csi_fps", type=int, default=30)
    p.add_argument("--sensor_id", type=int, default=0)

    p.add_argument("--show", action="store_true", help="Show window")
    p.add_argument("--max_plates", type=int, default=5, help="Max plates OCR per frame")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.det_engine):
        raise FileNotFoundError(f"Detector engine not found: {args.det_engine}")
    if (not args.no_ocr) and (not os.path.exists(args.ocr_engine)):
        raise FileNotFoundError(f"OCR engine not found: {args.ocr_engine}")

    detector = PlateDetector(args.det_engine, det_conf=args.conf, nms_iou=args.nms, verbose=args.verbose)
    ocr = None if args.no_ocr else PlateOCR(args.ocr_engine, charset=args.charset, verbose=args.verbose)

    cap = open_capture(args)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source={args.source}. RTSP={args.rtsp} cam={args.cam}")

    win = "ALPR"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # RTSP can drop. Try reconnect for rtsp.
            if args.source == "rtsp":
                cap.release()
                time.sleep(0.5)
                cap = open_capture(args)
                continue
            break

        t0 = time.time()
        plates = detector.detect(frame)

        # limit OCR count
        plates = sorted(plates, key=lambda x: x[4], reverse=True)[: args.max_plates]

        plate_count = 0
        for (x1, y1, x2, y2, score) in plates:
            plate_count += 1
            crop = frame[y1:y2, x1:x2].copy()
            text = ""
            if ocr is not None and crop.size > 0:
                try:
                    text = ocr.recognize(crop)
                except Exception:
                    text = ""
            draw_plate(frame, (x1, y1, x2, y2), text, score)

        # FPS
        dt = time.time() - t0
        inst_fps = 1.0 / max(dt, 1e-6)
        fps_smooth = inst_fps if fps_smooth <= 0 else (0.9 * fps_smooth + 0.1 * inst_fps)
        cv2.putText(frame, f"FPS {fps_smooth:.1f} plates={plate_count}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        if args.show:
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
