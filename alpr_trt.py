
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALPR (License Plate Detection + OCR) on Jetson using TensorRT engines (or ONNX fallback).

Supports:
- CSI camera (Jetson) via OpenCV GStreamer nvarguscamerasrc
- USB webcam via /dev/videoX
- RTSP via GStreamer rtspsrc (recommended on Jetson to avoid "blurry"/decoder issues)

Models expected (default paths):
- model/LP_detector_nano_61_fp16.engine  (or model/LP_detector_nano_61.onnx)
- model/LP_ocr_nano_62_fp16.engine       (or model/LP_ocr_nano_62.onnx)
- model/LP_ocr_nano_62.names             (labels for OCR classes, one label per line)

Run examples:
  # RTSP
  python3 alpr_trt.py --source rtsp --rtsp "rtsp://192.168.50.2:8554/mac" --show 1

  # CSI cam
  python3 alpr_trt.py --source csi --cam 0 --csi_w 1280 --csi_h 720 --csi_fps 30 --show 1

  # USB cam
  python3 alpr_trt.py --source usb --cam 0 --w 1280 --h 720 --show 1
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# TensorRT + PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# ----------------------------
# Utils
# ----------------------------
def now_ms() -> float:
    return time.time() * 1000.0


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize+pad to meet stride-multiple while keeping aspect ratio (YOLOv5 style)."""
    h, w = im.shape[:2]
    new_w, new_h = new_shape[0], new_shape[1]
    r = min(new_w / w, new_h / h)
    rw, rh = int(round(w * r)), int(round(h * r))

    im_resized = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - rw, new_h - rh
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2

    im_padded = cv2.copyMakeBorder(im_resized, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (pad_left, pad_top)


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    y = xywh.copy()
    y[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    y[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    y[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    y[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Pure numpy NMS. boxes: Nx4 xyxy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = boxes[:, 0].clip(0, w - 1)
    boxes[:, 1] = boxes[:, 1].clip(0, h - 1)
    boxes[:, 2] = boxes[:, 2].clip(0, w - 1)
    boxes[:, 3] = boxes[:, 3].clip(0, h - 1)
    return boxes


def read_labels(names_path: str) -> List[str]:
    if not os.path.exists(names_path):
        return []
    labels = []
    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            labels.append(s)
    return labels


# ----------------------------
# TensorRT runner
# ----------------------------
@dataclass
class TRTBinding:
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    is_input: bool
    host: np.ndarray
    device: int


class TrtRunner:
    def __init__(self, engine_path: str, logger_severity: int = trt.Logger.WARNING):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        self.logger = trt.Logger(logger_severity)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self.bindings: List[int] = [0] * self.engine.num_bindings
        self.binding_info: Dict[int, TRTBinding] = {}

        # allocate for static shapes
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.engine.get_binding_shape(i))
            is_input = self.engine.binding_is_input(i)

            # Some engines store -1 in shape (dynamic). This script expects fixed-shape engines.
            if any(d < 0 for d in shape):
                raise ValueError(
                    f"Engine has dynamic shape at binding '{name}'={shape}. "
                    "Rebuild engine with fixed shapes, or extend code for dynamic profiles."
                )

            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype)
            device = int(cuda.mem_alloc(host.nbytes))
            self.binding_info[i] = TRTBinding(name, dtype, shape, is_input, host, device)
            self.bindings[i] = device

        self.input_binding_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        self.output_binding_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]

    def infer(self, input_array: np.ndarray) -> List[np.ndarray]:
        """input_array must match the single input shape (NCHW) of the engine."""
        if len(self.input_binding_indices) != 1:
            raise RuntimeError("This runner expects exactly 1 input binding.")
        in_idx = self.input_binding_indices[0]
        in_info = self.binding_info[in_idx]

        # flatten copy to host
        np.copyto(in_info.host, input_array.ravel())
        cuda.memcpy_htod_async(in_info.device, in_info.host, self.stream)

        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        outputs = []
        for out_idx in self.output_binding_indices:
            out_info = self.binding_info[out_idx]
            cuda.memcpy_dtoh_async(out_info.host, out_info.device, self.stream)
        self.stream.synchronize()

        for out_idx in self.output_binding_indices:
            out_info = self.binding_info[out_idx]
            out = np.array(out_info.host, copy=True).reshape(out_info.shape)
            outputs.append(out)
        return outputs


# ----------------------------
# YOLOv5 decode (generic)
# ----------------------------
def yolo_decode(output: np.ndarray,
                conf_thres: float,
                iou_thres: float,
                num_classes: int,
                img_shape: Tuple[int, int],
                ratio: float,
                pad: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    output: (1, N, 5+nc) or (N, 5+nc) -> xywh, obj, cls...
    Returns boxes_xyxy (M,4) in original image coords, scores (M,), class_ids (M,)
    """
    if output.ndim == 3:
        pred = output[0]
    else:
        pred = output

    if pred.shape[1] < 5 + num_classes:
        raise ValueError(f"Unexpected output shape {pred.shape}, num_classes={num_classes}")

    xywh = pred[:, 0:4]
    obj = pred[:, 4:5]
    cls = pred[:, 5:5 + num_classes]

    cls_id = cls.argmax(axis=1)
    cls_score = cls.max(axis=1, keepdims=True)
    scores = (obj * cls_score).squeeze(1)

    keep = scores > conf_thres
    if not np.any(keep):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    xywh = xywh[keep]
    scores = scores[keep]
    cls_id = cls_id[keep].astype(np.int32)

    boxes = xywh2xyxy(xywh).astype(np.float32)

    # Undo letterbox: boxes are on padded-resized image (640x640)
    pad_x, pad_y = pad
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    h0, w0 = img_shape
    boxes = clip_boxes(boxes, w0, h0)

    # NMS per class (safe)
    final_boxes = []
    final_scores = []
    final_cls = []
    for c in np.unique(cls_id):
        idxs = np.where(cls_id == c)[0]
        if idxs.size == 0:
            continue
        k = nms_boxes(boxes[idxs], scores[idxs], iou_thres)
        sel = idxs[k]
        final_boxes.append(boxes[sel])
        final_scores.append(scores[sel])
        final_cls.append(cls_id[sel])

    if not final_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return (np.concatenate(final_boxes, axis=0),
            np.concatenate(final_scores, axis=0),
            np.concatenate(final_cls, axis=0))


# ----------------------------
# Video sources (GStreamer)
# ----------------------------
def gst_csi_pipeline(sensor_id: int, width: int, height: int, fps: int, flip: int) -> str:
    # nvarguscamerasrc output is NV12; convert to BGR for OpenCV
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def gst_rtsp_pipeline(rtsp_url: str, latency: int, width: int, height: int, use_tcp: bool) -> str:
    proto = "tcp" if use_tcp else "udp"
    return (
        f"rtspsrc location={rtsp_url} latency={latency} protocols={proto} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def open_capture(args) -> cv2.VideoCapture:
    if args.source == "csi":
        pipe = gst_csi_pipeline(args.cam, args.csi_w, args.csi_h, args.csi_fps, args.flip)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        pipe = gst_rtsp_pipeline(args.rtsp, args.rtsp_latency, args.w, args.h, args.rtsp_tcp)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    elif args.source == "gst":
        cap = cv2.VideoCapture(args.gst, cv2.CAP_GSTREAMER)
    elif args.source == "usb":
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    elif args.source == "file":
        cap = cv2.VideoCapture(args.file)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")
    return cap


# ----------------------------
# OCR postprocess
# ----------------------------
def ocr_from_dets(boxes: np.ndarray, cls_ids: np.ndarray, labels: List[str]) -> str:
    """
    OCR model detects characters on the plate crop.
    Sort by x-center and join labels.
    """
    if boxes.shape[0] == 0:
        return "unknown"
    # sort by x center
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    order = np.argsort(cx)
    chars = []
    for i in order:
        cid = int(cls_ids[i])
        if 0 <= cid < len(labels):
            chars.append(labels[cid])
        else:
            chars.append("?")
    text = "".join(chars)

    # light formatting for VN plates (optional):
    # - insert '-' after first 2-3 chars if looks like province code
    if len(text) >= 8:
        # common patterns: 63V60151 / 63B367409 / 51A99999 ...
        # Heuristic: after 2 digits, add '-'
        if text[:2].isdigit() and text[2].isalpha():
            text = text[:2] + "-" + text[2:]
        elif text[:2].isdigit() and text[2].isdigit():
            text = text[:2] + "-" + text[2:]
    return text


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csi", "usb", "rtsp", "file", "gst"], default="rtsp")
    ap.add_argument("--cam", type=int, default=0, help="camera index (usb) or sensor-id (csi)")
    ap.add_argument("--file", type=str, default="", help="video file path if --source file")
    ap.add_argument("--gst", type=str, default="", help="custom gstreamer pipeline if --source gst")
    ap.add_argument("--rtsp", type=str, default="", help="rtsp url if --source rtsp")
    ap.add_argument("--rtsp_latency", type=int, default=150)
    ap.add_argument("--rtsp_tcp", type=int, default=1, help="1=tcp, 0=udp")

    ap.add_argument("--w", type=int, default=1280, help="output capture width (rtsp/usb)")
    ap.add_argument("--h", type=int, default=720, help="output capture height (rtsp/usb)")
    ap.add_argument("--fps", type=int, default=30, help="usb requested fps")
    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)

    ap.add_argument("--imgsz", type=int, default=640, help="model input size (square)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.45)

    ap.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--det_onnx", type=str, default="model/LP_detector_nano_61.onnx")
    ap.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")
    ap.add_argument("--ocr_onnx", type=str, default="model/LP_ocr_nano_62.onnx")
    ap.add_argument("--ocr_names", type=str, default="model/LP_ocr_nano_62.names")
    ap.add_argument("--ocr_every", type=int, default=2, help="run OCR every N frames per detected plate")

    ap.add_argument("--show", type=int, default=1)
    ap.add_argument("--save", type=str, default="", help="optional output video path (mp4)")
    return ap.parse_args()


def build_engine_with_trtexec(onnx_path: str, engine_path: str, fp16: bool = True, workspace: int = 1024):
    """
    Build engine with trtexec (recommended on Jetson).
    NOTE: for static ONNX (fixed 1x3x640x640), DO NOT pass --minShapes/--optShapes/--maxShapes.
    """
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec):
        trtexec = "trtexec"  # hope it's in PATH

    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", f"--workspace={workspace}"]
    if fp16:
        cmd.append("--fp16")
    cmd.append("--verbose=0")

    print("\n[TRT] Build engine with:", " ".join(cmd), flush=True)
    import subprocess
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0 or (not os.path.exists(engine_path)):
        raise RuntimeError("trtexec build failed (see log above)")


def ensure_engine(engine_path: str, onnx_path: str):
    if os.path.exists(engine_path):
        return
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(onnx_path)
    build_engine_with_trtexec(onnx_path, engine_path)


def main():
    args = parse_args()

    # sanity checks for rtsp
    if args.source == "rtsp" and not args.rtsp:
        raise SystemExit("Missing --rtsp URL")
    if args.source == "file" and not args.file:
        raise SystemExit("Missing --file path")
    if args.source == "gst" and not args.gst:
        raise SystemExit("Missing --gst pipeline string")

    labels = read_labels(args.ocr_names)
    if not labels:
        print(f"[WARN] OCR labels not found/empty: {args.ocr_names}")
        print("       Create it on Mac/PC then copy to Jetson, example labels: 0-9,A,B,C,... one per line.")
    else:
        print(f"[OK] OCR labels: {len(labels)} classes")

    # Build engines if missing (on Jetson only)
    ensure_engine(args.det_engine, args.det_onnx)
    ensure_engine(args.ocr_engine, args.ocr_onnx)

    det = TrtRunner(args.det_engine)
    ocr = TrtRunner(args.ocr_engine)

    cap = open_capture(args)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (args.w, args.h))

    frame_id = 0
    t0 = time.time()
    fps_ema = 0.0

    last_ocr: Dict[int, Tuple[str, int]] = {}  # key=plate_idx, value=(text, last_frame_id)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] cannot read frame")
            time.sleep(0.05)
            continue

        h0, w0 = frame.shape[:2]

        # preprocess for detector
        img, r, (padx, pady) = letterbox(frame, (args.imgsz, args.imgsz))
        blob = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0-1
        blob = np.transpose(blob, (2, 0, 1))[None, ...]     # NCHW

        det_outs = det.infer(blob)
        # Most YOLOv5 ONNX has single output
        det_pred = det_outs[0]

        det_boxes, det_scores, det_cls = yolo_decode(
            det_pred, args.conf, args.nms,
            num_classes=1,  # plate detector typically 1 class
            img_shape=(h0, w0),
            ratio=r,
            pad=(padx, pady),
        )

        plate_texts: List[str] = []

        for i, box in enumerate(det_boxes):
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                plate_texts.append("unknown")
                continue

            # run OCR every N frames to boost FPS
            if args.ocr_every > 1 and (frame_id % args.ocr_every != 0):
                plate_texts.append(last_ocr.get(i, ("unknown", -9999))[0])
                continue

            # preprocess for OCR (same letterbox to 640)
            cimg, cr, (cpx, cpy) = letterbox(crop, (args.imgsz, args.imgsz))
            cblob = cimg[:, :, ::-1].astype(np.float32) / 255.0
            cblob = np.transpose(cblob, (2, 0, 1))[None, ...]

            ocr_outs = ocr.infer(cblob)
            ocr_pred = ocr_outs[0]

            # OCR model is multi-class (30 classes)
            nc = len(labels) if labels else 30
            o_boxes, o_scores, o_cls = yolo_decode(
                ocr_pred, conf_thres=0.25, iou_thres=0.3,
                num_classes=nc,
                img_shape=(crop.shape[0], crop.shape[1]),
                ratio=cr,
                pad=(cpx, cpy),
            )
            text = ocr_from_dets(o_boxes, o_cls, labels) if labels else "unknown"
            last_ocr[i] = (text, frame_id)
            plate_texts.append(text)

        # draw
        for i, box in enumerate(det_boxes):
            x1, y1, x2, y2 = box.astype(int)
            text = plate_texts[i] if i < len(plate_texts) else "unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS calc
        dt = (time.time() - t0)
        if dt > 0:
            cur_fps = 1.0 / dt
            fps_ema = cur_fps if fps_ema == 0 else (0.9 * fps_ema + 0.1 * cur_fps)
        t0 = time.time()

        cv2.putText(frame, f"FPS {fps_ema:.1f} plates={len(det_boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            out = frame
            if out.shape[1] != args.w or out.shape[0] != args.h:
                out = cv2.resize(out, (args.w, args.h))
            writer.write(out)

        if args.show:
            cv2.imshow("ALPR", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
