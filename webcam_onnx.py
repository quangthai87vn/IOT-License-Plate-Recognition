#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

# TensorRT (optional but recommended)
TRT_OK = True
try:
    import tensorrt as trt
    import pycuda.driver as cuda
except Exception:
    TRT_OK = False


# -----------------------------
# Utils
# -----------------------------
def now_ms() -> float:
    return time.time() * 1000.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_imshow(win, img, show: int):
    if show:
        cv2.imshow(win, img)
        cv2.waitKey(1)


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize with unchanged aspect ratio using padding (YOLO-style)."""
    h, w = im.shape[:2]
    new_w, new_h = new_shape[0], new_shape[1]

    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))

    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - nw
    pad_h = new_h - nh
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)


def nms_boxes(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """NMS using OpenCV."""
    if len(boxes_xyxy) == 0:
        return []
    b = boxes_xyxy.astype(np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    w = (x2 - x1).clip(min=0)
    h = (y2 - y1).clip(min=0)
    rects = np.stack([x1, y1, w, h], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(rects, scores.tolist(), score_threshold=0.0, nms_threshold=float(iou_thres))
    if len(idxs) == 0:
        return []
    return [int(i) for i in idxs.flatten()]


# -----------------------------
# TensorRT Inference Wrapper
# -----------------------------
@dataclass
class TrtBinding:
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    is_input: bool
    host_mem: np.ndarray
    device_mem: int  # cuda device pointer


class TrtRunner:
    """
    Simple TensorRT runner for single input engine.
    - Creates its own CUDA context (stable on Jetson)
    - Supports fixed-shape engines and explicit-batch.
    """
    def __init__(self, engine_path: str, logger_severity: int = 2):
        if not TRT_OK:
            raise RuntimeError("TensorRT/pycuda not available in this environment.")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.engine_path = engine_path
        self.logger = trt.Logger(logger_severity)  # WARNING=2
        self.runtime = trt.Runtime(self.logger)

        self.ctx = None
        self.stream = None
        self.engine = None
        self.context = None
        self.bindings: List[TrtBinding] = []
        self.binding_addrs: List[int] = []

    def __enter__(self):
        cuda.init()
        dev = cuda.Device(0)
        self.ctx = dev.make_context()  # IMPORTANT: stable context per process
        self.stream = cuda.Stream()

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context.")

        # If engine has dynamic shapes, we will set shape when calling infer().
        # For allocation now, we allocate based on current binding shapes (may include -1).
        self._allocate_buffers_initial()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except Exception:
            pass

        try:
            for b in self.bindings:
                try:
                    cuda.mem_free(b.device_mem)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.ctx is not None:
                self.ctx.pop()
                self.ctx.detach()
        except Exception:
            pass

    def _np_dtype(self, trt_dtype):
        if trt_dtype == trt.DataType.FLOAT:
            return np.float32
        if trt_dtype == trt.DataType.HALF:
            return np.float16
        if trt_dtype == trt.DataType.INT8:
            return np.int8
        if trt_dtype == trt.DataType.INT32:
            return np.int32
        raise ValueError(f"Unsupported TRT dtype: {trt_dtype}")

    def _allocate_buffers_initial(self):
        self.bindings = []
        self.binding_addrs = [0] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            dtype = self._np_dtype(self.engine.get_binding_dtype(i))

            shape = tuple(self.engine.get_binding_shape(i))
            # If dynamic (-1), allocate a tiny placeholder; real allocation happens in infer()
            if any(s < 0 for s in shape):
                shape = (1,)

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(
                TrtBinding(
                    name=name, dtype=dtype, shape=shape, is_input=is_input,
                    host_mem=host_mem, device_mem=int(device_mem)
                )
            )
            self.binding_addrs[i] = int(device_mem)

    def _realloc_for_shapes(self, input_index: int, input_shape: Tuple[int, ...]):
        """
        Re-allocate buffers when engine uses dynamic shapes or when we want to force input shape.
        """
        # Set input binding shape
        self.context.set_binding_shape(input_index, input_shape)

        # Realloc all buffers based on runtime shapes
        for i, b in enumerate(self.bindings):
            real_shape = tuple(self.context.get_binding_shape(i))
            if any(s < 0 for s in real_shape):
                raise RuntimeError(f"Binding still dynamic after set_binding_shape: {b.name} shape={real_shape}")
            real_size = int(np.prod(real_shape))

            # If size differs -> realloc
            if real_size != b.host_mem.size:
                # free old
                try:
                    cuda.mem_free(b.device_mem)
                except Exception:
                    pass

                host_mem = cuda.pagelocked_empty(real_size, b.dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                self.bindings[i] = TrtBinding(
                    name=b.name, dtype=b.dtype, shape=real_shape, is_input=b.is_input,
                    host_mem=host_mem, device_mem=int(device_mem)
                )
                self.binding_addrs[i] = int(device_mem)
            else:
                # update shape
                self.bindings[i].shape = real_shape

    def infer(self, inp: np.ndarray) -> List[np.ndarray]:
        """
        inp: contiguous numpy array matching input binding dtype/shape
        returns list of outputs as numpy arrays
        """
        # Find input binding index
        input_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        if len(input_indices) != 1:
            raise RuntimeError("This runner supports engines with exactly 1 input.")
        in_idx = input_indices[0]

        # Ensure correct shape
        inp = np.ascontiguousarray(inp)
        target_shape = tuple(inp.shape)

        # Dynamic or mismatched -> set and realloc
        engine_in_shape = tuple(self.engine.get_binding_shape(in_idx))
        if any(s < 0 for s in engine_in_shape) or tuple(self.context.get_binding_shape(in_idx)) != target_shape:
            self._realloc_for_shapes(in_idx, target_shape)

        # Copy input host
        b_in = self.bindings[in_idx]
        if inp.dtype != b_in.dtype:
            inp = inp.astype(b_in.dtype)

        np.copyto(b_in.host_mem.reshape(-1), inp.reshape(-1))
        cuda.memcpy_htod_async(b_in.device_mem, b_in.host_mem, self.stream)

        # Execute
        ok = self.context.execute_async_v2(bindings=self.binding_addrs, stream_handle=int(self.stream.handle))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed.")

        # Copy outputs back
        outputs: List[np.ndarray] = []
        for i, b in enumerate(self.bindings):
            if b.is_input:
                continue
            cuda.memcpy_dtoh_async(b.host_mem, b.device_mem, self.stream)
        self.stream.synchronize()

        for i, b in enumerate(self.bindings):
            if b.is_input:
                continue
            out = np.array(b.host_mem, copy=True).reshape(b.shape)
            outputs.append(out)

        return outputs


# -----------------------------
# OCR decoding (CTC)
# -----------------------------
def default_charset():
    """
    Charset for VN plates (common set).
    If your repo uses a different mapping, chỉnh ở đây 1 phát là xong.
    """
    chars = list("0123456789")
    chars += list("ABCDEFGHKLMNPRSTUVXYZ")  # bỏ mấy chữ dễ nhầm
    chars += list("-.")
    return chars


def ctc_greedy_decode(logits: np.ndarray, charset: List[str], blank: int = 0) -> str:
    """
    logits: (1, T, C) or (T, C) or (1, C, T) depending model.
    We'll auto-fix to (T, C).
    """
    arr = logits
    arr = np.array(arr)

    # Squeeze batch
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # If shape looks like (C, T) -> transpose
    if arr.ndim == 2:
        # Heuristic: if first dim is classes and second is timesteps
        # classes usually ~ (len(charset)+1) ~ 40-60
        if arr.shape[0] <= 80 and arr.shape[1] > arr.shape[0]:
            # could be (C,T) -> make (T,C)
            if arr.shape[0] == (len(charset) + 1):
                arr = arr.T
        # else keep (T,C)
    elif arr.ndim == 3:
        # e.g. (C, H, W) not expected
        arr = arr.reshape(arr.shape[0], -1).T
    else:
        arr = arr.reshape(-1, arr.shape[-1])

    # Greedy
    pred = np.argmax(arr, axis=1).tolist()

    # CTC collapse repeats + remove blank
    res = []
    prev = None
    for p in pred:
        if p == prev:
            continue
        prev = p
        if p == blank:
            continue
        # map: (blank=0) => charset index (p-1)
        ci = p - 1
        if 0 <= ci < len(charset):
            res.append(charset[ci])

    return "".join(res)


def format_vn_plate(raw: str) -> str:
    """
    Make it look like repo: "63-B9 951.64"
    Input raw maybe: "63B995164"
    """
    s = raw.upper()
    s = re.sub(r"[^0-9A-Z]", "", s)

    if len(s) < 6:
        return raw.strip()

    # province
    p = s[:2]
    rest = s[2:]

    # series: often starts with a letter; sometimes 1 letter + 1 digit (e.g. B9)
    series = ""
    number = ""

    if len(rest) >= 2 and rest[0].isalpha():
        series = rest[:2]
        number = rest[2:]
    else:
        series = rest[:1]
        number = rest[1:]

    # format number with dot if length matches
    if len(number) == 5:
        number = f"{number[:3]}.{number[3:]}"
    elif len(number) == 4:
        number = f"{number[:2]}.{number[2:]}"

    out = f"{p}-{series} {number}".strip()
    return out


# -----------------------------
# Postprocess detector output (robust)
# -----------------------------
def parse_detector_output(out: np.ndarray, conf_thres: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      boxes_xyxy (N,4) in input-image pixel coords
      scores (N,)
    Supports common layouts:
      - (1, N, 6): [x1,y1,x2,y2,conf,cls]
      - (N, 6)
      - (1, N, 85) style: [cx,cy,w,h,obj,cls...]
      - (N, 85)
    """
    x = np.array(out)
    x = np.squeeze(x)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    # If (N,6) -> assume xyxy
    if x.ndim == 2 and x.shape[1] == 6:
        boxes = x[:, 0:4].astype(np.float32)
        scores = x[:, 4].astype(np.float32)
        keep = scores >= conf_thres
        return boxes[keep], scores[keep]

    # If YOLO raw (N, >=6)
    if x.ndim == 2 and x.shape[1] >= 6:
        # If has many classes -> assume [cx,cy,w,h,obj,classes...]
        if x.shape[1] > 10:
            obj = x[:, 4].astype(np.float32)
            cls_probs = x[:, 5:].astype(np.float32)
            cls = np.max(cls_probs, axis=1)
            scores = obj * cls
            keep = scores >= conf_thres

            y = x[keep]
            scores = scores[keep]
            if len(y) == 0:
                return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

            cxcywh = y[:, 0:4].astype(np.float32)
            cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            return boxes, scores.astype(np.float32)

        # else unknown, try treat as xyxy + score
        boxes = x[:, 0:4].astype(np.float32)
        scores = x[:, 4].astype(np.float32)
        keep = scores >= conf_thres
        return boxes[keep], scores[keep]

    # If (something, something, 6/85)
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
        return parse_detector_output(x, conf_thres)

    return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)


def undo_letterbox(boxes_xyxy: np.ndarray, ratio: float, pad: Tuple[int, int], orig_w: int, orig_h: int):
    """Map boxes from letterboxed image to original image coordinates."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy

    left, top = pad
    b = boxes_xyxy.copy().astype(np.float32)
    b[:, [0, 2]] -= left
    b[:, [1, 3]] -= top
    b /= ratio

    b[:, 0] = np.clip(b[:, 0], 0, orig_w - 1)
    b[:, 2] = np.clip(b[:, 2], 0, orig_w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, orig_h - 1)
    b[:, 3] = np.clip(b[:, 3], 0, orig_h - 1)
    return b


# -----------------------------
# Video sources
# -----------------------------
def gst_csi_pipeline(sensor_id=0, capture_w=1280, capture_h=720, framerate=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_w}, height=(int){capture_h}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width=(int){capture_w}, height=(int){capture_h}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 sync=false"
    )


def gst_rtsp_pipeline(url: str, latency=200):
    return (
        f"rtspsrc location={url} latency={latency} ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink drop=1 sync=false"
    )


# -----------------------------
# Main loop
# -----------------------------
def run(args):
    det_engine = args.det_engine
    ocr_engine = args.ocr_engine

    if not os.path.exists(det_engine):
        raise FileNotFoundError(f"Missing det_engine: {det_engine}")
    if not os.path.exists(ocr_engine):
        raise FileNotFoundError(f"Missing ocr_engine: {ocr_engine}")

    # Open source
    if args.source == "csi":
        cap = cv2.VideoCapture(gst_csi_pipeline(args.sensor_id, args.csi_w, args.csi_h, args.csi_fps, args.flip), cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        if not args.rtsp:
            raise ValueError("RTSP source selected but --rtsp is empty.")
        cap = cv2.VideoCapture(gst_rtsp_pipeline(args.rtsp, args.rtsp_latency), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source. (Check docker args + /tmp/argus_socket for CSI + network for RTSP)")

    charset = default_charset()

    # TRT runners
    with TrtRunner(det_engine) as det_trt, TrtRunner(ocr_engine) as ocr_trt:
        last_t = time.time()
        fps = 0.0
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # RTSP sometimes returns None briefly
                time.sleep(0.02)
                continue

            frame_id += 1
            H, W = frame.shape[:2]

            # FPS
            t = time.time()
            dt = t - last_t
            last_t = t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Detector preprocess
            inp_det, ratio, pad = letterbox(frame, (args.det_size, args.det_size))
            rgb = cv2.cvtColor(inp_det, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1,3,H,W)

            # TRT infer detector
            det_outs = det_trt.infer(chw)
            det_out = det_outs[0]

            boxes_inp, scores = parse_detector_output(det_out, args.conf)
            if len(boxes_inp) > 0:
                # If normalized coords, scale to det_size
                m = float(np.max(boxes_inp))
                if m <= 2.0:  # heuristic normalized
                    boxes_inp[:, [0, 2]] *= args.det_size
                    boxes_inp[:, [1, 3]] *= args.det_size

            # Undo letterbox to original frame
            boxes = undo_letterbox(boxes_inp, ratio, pad, W, H)

            # NMS
            keep = nms_boxes(boxes, scores, args.nms)
            boxes = boxes[keep] if len(keep) else np.zeros((0, 4), np.float32)
            scores_kept = scores[keep] if len(keep) else np.zeros((0,), np.float32)

            plates_text = []

            # OCR only every N frames if set
            do_ocr = (args.ocr_every <= 1) or (frame_id % args.ocr_every == 0)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].astype(int)
                x1 = clamp(x1, 0, W - 1)
                y1 = clamp(y1, 0, H - 1)
                x2 = clamp(x2, 0, W - 1)
                y2 = clamp(y2, 0, H - 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if not do_ocr:
                    continue

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # OCR preprocess
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, (args.ocr_w, args.ocr_h), interpolation=cv2.INTER_LINEAR)
                o = roi_resized.astype(np.float32) / 255.0
                o = np.transpose(o, (2, 0, 1))[None, ...]  # (1,3,H,W)

                # TRT infer OCR
                ocr_outs = ocr_trt.infer(o)
                logits = ocr_outs[0]

                raw = ctc_greedy_decode(logits, charset, blank=0)
                plate = format_vn_plate(raw)

                plates_text.append(plate)

                # draw text above box (repo style)
                cv2.putText(
                    frame,
                    plate,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # HUD
            cv2.putText(frame, f"FPS {fps:.1f} plates={len(boxes)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            safe_imshow("ALPR", frame, args.show)

            # Exit key
            if args.show and (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


def build_argparser():
    p = argparse.ArgumentParser("ALPR TensorRT (CSI/RTSP/Webcam)")

    p.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi",
                   help="Video source type")
    p.add_argument("--show", type=int, default=1, help="1 to show window, 0 headless")

    # CSI
    p.add_argument("--sensor_id", type=int, default=0)
    p.add_argument("--csi_w", type=int, default=1280)
    p.add_argument("--csi_h", type=int, default=720)
    p.add_argument("--csi_fps", type=int, default=30)
    p.add_argument("--flip", type=int, default=0)

    # RTSP
    p.add_argument("--rtsp", type=str, default="")
    p.add_argument("--rtsp_latency", type=int, default=200)

    # Webcam
    p.add_argument("--cam", type=int, default=0)

    # Models
    p.add_argument("--det_engine", type=str, default="./model/LP_detector_nano_61_fp16.engine")
    p.add_argument("--ocr_engine", type=str, default="./model/LP_ocr_nano_62_fp16.engine")

    # Params
    p.add_argument("--det_size", type=int, default=640)
    p.add_argument("--ocr_w", type=int, default=160)
    p.add_argument("--ocr_h", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--nms", type=float, default=0.35)
    p.add_argument("--ocr_every", type=int, default=1, help="OCR every N frames (>=1)")

    return p


def main():
    args = build_argparser().parse_args()

    if not TRT_OK:
        raise RuntimeError(
            "Thiếu TensorRT/pycuda trong container. "
            "Bạn đang chạy Jetson thì thường có sẵn. Nếu không có, image/Dockerfile đang sai base."
        )

    run(args)


if __name__ == "__main__":
    main()
