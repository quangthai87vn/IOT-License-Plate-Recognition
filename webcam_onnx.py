#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALPR on Jetson (CSI / RTSP / webcam) using TensorRT engines.

Expected files (default):
  model/LP_detector_nano_61_fp16.engine
  model/LP_ocr_nano_62_fp16.engine

If your engines have different names, pass --det_engine / --ocr_engine.

Run:
  python3 csi.py
  python3 rtsp.py "rtsp://user:pass@ip:554/stream"
  python3 webcam_onnx.py --source webcam --cam 0 --show 1
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# =========================
# Utils
# =========================

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

def make_trt_logger(severity=trt.Logger.WARNING):
    try:
        return trt.Logger(severity)
    except TypeError:
        try:
            return trt.Logger(min_severity=severity)
        except TypeError:
            return trt.Logger()

def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with padding (YOLO style). Returns: img, ratio, (dw, dh)."""
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
    return im, r, (left, top)

def scale_coords(boxes_xyxy: np.ndarray, ratio: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]):
    """Map boxes from letterboxed image back to original image."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    left, top = pad
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, :4] /= ratio
    h, w = orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """NMS using OpenCV (expects xywh), so convert."""
    if len(boxes) == 0:
        return []
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    idxs = cv2.dnn.NMSBoxes(
        bboxes=xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_thres),
    )
    if len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


# =========================
# TensorRT runner
# =========================

class TRTModule:
    def __init__(self, engine_path: str, logger_severity=trt.Logger.WARNING):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.engine_path = engine_path
        self.logger = make_trt_logger(logger_severity)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self.bindings = [None] * self.engine.num_bindings

        self.host_inputs: Dict[int, np.ndarray] = {}
        self.device_inputs: Dict[int, int] = {}
        self.host_outputs: Dict[int, np.ndarray] = {}
        self.device_outputs: Dict[int, int] = {}

        self.input_binding_idxs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        self.output_binding_idxs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

    def get_input_shape(self, binding_idx: int = 0) -> Tuple[int, ...]:
        idx = self.input_binding_idxs[binding_idx]
        return tuple(int(x) for x in self.engine.get_binding_shape(idx))

    def _allocate(self, binding_idx: int, shape: Tuple[int, ...], dtype: np.dtype):
        size = int(np.prod(shape))
        host = cuda.pagelocked_empty(size, dtype)
        device = cuda.mem_alloc(host.nbytes)
        self.bindings[binding_idx] = int(device)
        return host, device

    def _ensure_binding(self, binding_idx: int, shape: Tuple[int, ...], dtype: np.dtype, is_input: bool):
        if is_input:
            if binding_idx not in self.host_inputs or self.host_inputs[binding_idx].size != int(np.prod(shape)):
                host, dev = self._allocate(binding_idx, shape, dtype)
                self.host_inputs[binding_idx] = host
                self.device_inputs[binding_idx] = dev
        else:
            if binding_idx not in self.host_outputs or self.host_outputs[binding_idx].size != int(np.prod(shape)):
                host, dev = self._allocate(binding_idx, shape, dtype)
                self.host_outputs[binding_idx] = host
                self.device_outputs[binding_idx] = dev

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for bidx in self.input_binding_idxs:
            name = self.engine.get_binding_name(bidx)
            if name not in inputs:
                raise KeyError(f"Missing input '{name}' for engine {self.engine_path}")
            arr = inputs[name]
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            dtype = trt.nptype(self.engine.get_binding_dtype(bidx))
            shape = tuple(int(x) for x in arr.shape)

            if -1 in self.engine.get_binding_shape(bidx):
                self.context.set_binding_shape(bidx, shape)

            final_shape = tuple(int(x) for x in self.context.get_binding_shape(bidx))
            self._ensure_binding(bidx, final_shape, np.dtype(dtype), is_input=True)

            np.copyto(self.host_inputs[bidx], arr.ravel().astype(dtype, copy=False))
            cuda.memcpy_htod_async(self.device_inputs[bidx], self.host_inputs[bidx], self.stream)

        for bidx in self.output_binding_idxs:
            dtype = trt.nptype(self.engine.get_binding_dtype(bidx))
            shape = tuple(int(x) for x in self.context.get_binding_shape(bidx))
            self._ensure_binding(bidx, shape, np.dtype(dtype), is_input=False)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        for bidx in self.output_binding_idxs:
            cuda.memcpy_dtoh_async(self.host_outputs[bidx], self.device_outputs[bidx], self.stream)

        self.stream.synchronize()

        out_arrays: Dict[str, np.ndarray] = {}
        for bidx in self.output_binding_idxs:
            name = self.engine.get_binding_name(bidx)
            shape = tuple(int(x) for x in self.context.get_binding_shape(bidx))
            dtype = trt.nptype(self.engine.get_binding_dtype(bidx))
            out = np.array(self.host_outputs[bidx], copy=True).astype(dtype, copy=False).reshape(shape)
            out_arrays[name] = out

        return out_arrays


# =========================
# Detector
# =========================

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

class PlateDetector:
    def __init__(self, engine_path: str, conf: float = 0.35, iou: float = 0.45, img_size: int = 640):
        self.trt = TRTModule(engine_path, logger_severity=trt.Logger.WARNING)
        self.conf = float(conf)
        self.iou = float(iou)
        self.img_size = int(img_size)
        self.inp_name = self.trt.engine.get_binding_name(self.trt.input_binding_idxs[0])

    def _decode_outputs(self, out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]

        # already NMSed: (N,6)
        if out.ndim == 2 and out.shape[1] >= 6 and out.shape[1] <= 8:
            boxes = out[:, 0:4].astype(np.float32)
            scores = out[:, 4].astype(np.float32)
            classes = out[:, 5].astype(np.int32)
            keep = scores >= self.conf
            return boxes[keep], scores[keep], classes[keep]

        # YOLO raw: (N, 5+nc)
        if out.ndim == 2 and out.shape[1] > 8:
            boxes_cxcywh = out[:, 0:4].astype(np.float32)
            obj = out[:, 4].astype(np.float32)
            cls_probs = out[:, 5:].astype(np.float32)
            cls_ids = np.argmax(cls_probs, axis=1)
            cls_conf = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]
            scores = obj * cls_conf

            keep = scores >= self.conf
            boxes_cxcywh = boxes_cxcywh[keep]
            scores = scores[keep]
            cls_ids = cls_ids[keep]

            boxes = np.zeros_like(boxes_cxcywh, dtype=np.float32)
            boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
            boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
            boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
            boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
            return boxes, scores, cls_ids

        raise RuntimeError(f"Unknown detector output shape: {out.shape}")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        im0 = frame_bgr
        h0, w0 = im0.shape[:2]
        img, r, pad = letterbox(im0, new_shape=(self.img_size, self.img_size))
        img = img[:, :, ::-1]  # BGR->RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW

        outs = self.trt.infer({self.inp_name: img})
        out = max(outs.values(), key=lambda a: a.size)

        boxes, scores, classes = self._decode_outputs(out)

        if out.ndim == 2 and out.shape[1] > 8:
            keep = nms_xyxy(boxes, scores, self.iou)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        boxes = scale_coords(boxes, r, pad, (h0, w0))

        dets: List[Detection] = []
        for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, classes):
            dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(sc), int(cl)))
        return dets


# =========================
# OCR
# =========================

def ctc_decode_best(logits: np.ndarray) -> Tuple[str, float]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (T,C), got {logits.shape}")

    probs = logits
    if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-2):
        probs = _softmax(probs, axis=1)

    pred = np.argmax(probs, axis=1).astype(np.int32)
    pred_prob = np.max(probs, axis=1).astype(np.float32)
    C = probs.shape[1]

    def decode_with(blank_id: int, offset: int) -> Tuple[str, float]:
        out = []
        prev = -999
        confs = []
        for i, p in zip(pred, pred_prob):
            if i == prev:
                continue
            prev = i
            if i == blank_id:
                continue
            j = int(i) - offset
            if 0 <= j < len(CHARS):
                out.append(CHARS[j])
                confs.append(float(p))
        text = "".join(out)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    candidates = [
        decode_with(blank_id=0, offset=1),
        decode_with(blank_id=C - 1, offset=0),
        decode_with(blank_id=0, offset=0),
    ]

    def score(t_conf):
        t, c = t_conf
        L = len(t)
        len_bonus = 0
        if 6 <= L <= 10:
            len_bonus = 2
        elif 4 <= L <= 12:
            len_bonus = 1
        if L > 14:
            len_bonus -= 3
        return (len_bonus, c)

    return max(candidates, key=score)

def format_vn_plate(top: str, bottom: Optional[str] = None) -> str:
    def fmt_top(s: str) -> str:
        s = "".join([ch for ch in s if ch.isalnum()])
        if len(s) == 4:
            return s[:2] + "-" + s[2:]
        if len(s) == 5:
            return s[:3] + "-" + s[3:]
        return s

    def fmt_bottom(s: str) -> str:
        s = "".join([ch for ch in s if ch.isalnum()])
        if len(s) == 5:
            return s[:3] + "." + s[3:]
        if len(s) == 4:
            return s[:2] + "." + s[2:]
        return s

    top_f = fmt_top(top)
    if bottom is None or bottom == "":
        return top_f
    bot_f = fmt_bottom(bottom)
    return f"{top_f} {bot_f}"

class PlateOCR:
    def __init__(self, engine_path: str):
        self.trt = TRTModule(engine_path, logger_severity=trt.Logger.WARNING)
        self.inp_name = self.trt.engine.get_binding_name(self.trt.input_binding_idxs[0])

        ishape = self.trt.engine.get_binding_shape(self.trt.input_binding_idxs[0])
        if -1 in ishape:
            self.c, self.h, self.w = 3, 40, 160
        else:
            _, c, h, w = [int(x) for x in ishape]
            self.c, self.h, self.w = c, h, w

    def _prep(self, plate_bgr: np.ndarray) -> np.ndarray:
        img = plate_bgr
        if self.c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = img[None, None, :, :]
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def _extract_logits(self, outs: Dict[str, np.ndarray]) -> np.ndarray:
        out = max(outs.values(), key=lambda a: a.size)
        arr = out
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            if arr.shape[0] <= 128 and arr.shape[1] > arr.shape[0]:
                return arr.T
            return arr
        if arr.ndim == 1:
            raise RuntimeError(f"OCR output is 1D: {out.shape}")
        arr = np.reshape(arr, (-1, arr.shape[-1]))
        return arr

    def recognize(self, plate_bgr: np.ndarray) -> Tuple[str, float]:
        h, w = plate_bgr.shape[:2]
        two_line = (h / (w + 1e-6)) > 0.55

        if two_line:
            top = plate_bgr[: h // 2, :]
            bottom = plate_bgr[h // 2 :, :]
            t_text, t_conf = self._recognize_one(top)
            b_text, b_conf = self._recognize_one(bottom)
            text = format_vn_plate(t_text, b_text)
            conf = (t_conf + b_conf) / 2.0
            return text, conf

        text, conf = self._recognize_one(plate_bgr)
        raw = "".join([ch for ch in text if ch.isalnum()])
        if len(raw) >= 7:
            top = raw[:4]
            bottom = raw[4:]
            return format_vn_plate(top, bottom), conf
        return text, conf

    def _recognize_one(self, img_bgr: np.ndarray) -> Tuple[str, float]:
        inp = self._prep(img_bgr)
        outs = self.trt.infer({self.inp_name: inp})
        logits = self._extract_logits(outs)
        text, conf = ctc_decode_best(logits)
        return text, conf


# =========================
# Video sources
# =========================

def gst_csi(sensor_id=0, width=1640, height=1232, fps=30, flip=0, out_w=1280, out_h=720):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )

def gst_rtsp(url: str, latency: int = 200):
    return (
        f"rtspsrc location={url} latency={latency} ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )

def open_capture(args) -> cv2.VideoCapture:
    if args.source == "csi":
        return cv2.VideoCapture(
            gst_csi(sensor_id=args.cam, fps=args.csi_fps, out_w=args.csi_w, out_h=args.csi_h),
            cv2.CAP_GSTREAMER,
        )
    if args.source == "rtsp":
        cap = cv2.VideoCapture(args.rtsp)
        if cap.isOpened():
            return cap
        return cv2.VideoCapture(gst_rtsp(args.rtsp, latency=args.rtsp_latency), cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(int(args.cam))


# =========================
# Main
# =========================

def run(args) -> int:
    detector = PlateDetector(args.det_engine, conf=args.conf, iou=args.nms, img_size=args.det_size)
    ocr = PlateOCR(args.ocr_engine)

    cap = open_capture(args)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera/luồng video. Kiểm tra --source, --cam, --rtsp và GStreamer trong Docker.")

    win = "ALPR"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            dets = detector.detect(frame)

            plates = 0
            for d in dets:
                x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
                pad = int(0.06 * max(x2 - x1, y2 - y1))
                x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
                x2p = min(frame.shape[1]-1, x2 + pad); y2p = min(frame.shape[0]-1, y2 + pad)
                crop = frame[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                text, conf = ocr.recognize(crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{text} ({conf:.2f})"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                plates += 1

            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
            cv2.putText(frame, f"FPS {fps:.1f} plates={plates}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

    finally:
        try:
            cap.release()
        except Exception:
            pass
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    p.add_argument("--cam", default=0, help="CSI sensor-id hoặc webcam index")
    p.add_argument("--rtsp", default="", help="RTSP URL nếu --source rtsp")
    p.add_argument("--rtsp_latency", type=int, default=200)

    p.add_argument("--csi_w", type=int, default=1280)
    p.add_argument("--csi_h", type=int, default=720)
    p.add_argument("--csi_fps", type=int, default=30)

    p.add_argument("--det_engine", default="model/LP_detector_nano_61_fp16.engine")
    p.add_argument("--ocr_engine", default="model/LP_ocr_nano_62_fp16.engine")
    p.add_argument("--det_size", type=int, default=640)

    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--nms", type=float, default=0.45)

    p.add_argument("--show", type=int, default=1)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        args.cam = int(args.cam)
    except Exception:
        pass

    if args.source == "rtsp" and not args.rtsp:
        raise SystemExit("Bạn chọn --source rtsp thì phải truyền --rtsp <URL>")

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
