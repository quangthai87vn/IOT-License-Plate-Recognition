#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # tạo CUDA context 1 lần (quan trọng, đỡ "unspecified launch failure")


# ----------------------------
# Utils: labels + format VN
# ----------------------------
def try_load_labels(paths: List[str]) -> Optional[List[str]]:
    for p in paths:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                lines = [x.strip() for x in f.read().splitlines() if x.strip()]
            if lines:
                return lines
    return None


def default_ocr_labels():
    # fallback nếu không có file labels
    # thường YOLO OCR dùng 0-9 + A-Z (bỏ I,O,Q,W) nhưng mỗi repo khác nhau.
    # -> tốt nhất để file labels.txt trong model/
    chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")
    return chars


def format_vn_plate(raw: str) -> str:
    """
    Heuristic format biển số VN phổ biến:
      - Motor: 63B995164 -> 63-B9 951.64
      - Car:   51F99999  -> 51F-999.99 (tuỳ series)
    Nếu raw đã có '-' '.' ' ' thì giữ nguyên.
    """
    s = raw.strip().upper()
    if not s:
        return s
    # nếu đã có format
    if any(ch in s for ch in ["-", ".", " "]):
        return s

    # chỉ giữ A-Z0-9
    s2 = "".join([c for c in s if c.isalnum()])

    # motor kiểu: 2 số + 2 ký tự + 5 số (vd 63B995164 hoặc 63B965814)
    if len(s2) == 9 and s2[:2].isdigit() and s2[2:4].isalnum() and s2[4:].isdigit():
        return f"{s2[:2]}-{s2[2:4]} {s2[4:7]}.{s2[7:]}"
    # 8 ký tự: 2 số + 1 chữ + 5 số (vd 51F99999)
    if len(s2) == 8 and s2[:2].isdigit() and s2[2].isalnum() and s2[3:].isdigit():
        return f"{s2[:3]}-{s2[3:6]}.{s2[6:]}"
    return s


# ----------------------------
# GStreamer pipelines
# ----------------------------
def gst_csi_pipeline(sensor_id=0, csi_w=1640, csi_h=1232, fps=30, flip=0, out_w=1280, out_h=720):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={csi_w}, height={csi_h}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def gst_rtsp_pipeline(url, latency=250, tcp=True, out_w=1280, out_h=720):
    proto = "tcp" if tcp else "udp"
    # pipeline decode H264 bằng HW (nvv4l2decoder) để nhẹ
    return (
        f"rtspsrc location={url} latency={latency} protocols={proto} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


# ----------------------------
# TensorRT runner
# ----------------------------
@dataclass
class TrtIO:
    host: np.ndarray
    device: cuda.DeviceAllocation
    size: int


class TrtRunner:
    """
    Load TensorRT engine (.engine), infer 1 input -> 1 output (hoặc nhiều outputs).
    Hỗ trợ dynamic shapes nhưng mặc định bạn đang build 640x640 cố định.
    """
    def __init__(self, engine_path: str, logger_severity: int = trt.Logger.WARNING):
        self.engine_path = engine_path
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"Không thấy engine: {engine_path}")

        with open(engine_path, "rb") as f:
            buf = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(buf)
        if self.engine is None:
            raise RuntimeError(f"Deserialize engine failed: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed")

        self.stream = cuda.Stream()

        # bindings
        self.bindings = [None] * self.engine.num_bindings
        self.inputs_idx = []
        self.outputs_idx = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.inputs_idx.append(i)
            else:
                self.outputs_idx.append(i)

        # allocate once with max shapes (ở đây assume fixed)
        self.io = {}
        for i in range(self.engine.num_bindings):
            shape = self.context.get_binding_shape(i)
            if -1 in shape:
                # dynamic -> set default
                # bạn đang dùng 1x3x640x640 nên set vậy luôn
                if self.engine.binding_is_input(i):
                    self.context.set_binding_shape(i, (1, 3, 640, 640))
                    shape = self.context.get_binding_shape(i)

            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            vol = int(trt.volume(shape))
            host = cuda.pagelocked_empty(vol, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.bindings[i] = int(device)
            self.io[i] = TrtIO(host=host, device=device, size=host.nbytes)

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        # assume 1 input
        in_idx = self.inputs_idx[0]

        # set shape if needed
        if -1 in tuple(self.context.get_binding_shape(in_idx)):
            self.context.set_binding_shape(in_idx, input_tensor.shape)

        # copy input
        np.copyto(self.io[in_idx].host, input_tensor.ravel())
        cuda.memcpy_htod_async(self.io[in_idx].device, self.io[in_idx].host, self.stream)

        # execute
        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        # copy outputs
        outputs = []
        for out_idx in self.outputs_idx:
            cuda.memcpy_dtoh_async(self.io[out_idx].host, self.io[out_idx].device, self.stream)
        self.stream.synchronize()

        for out_idx in self.outputs_idx:
            shape = tuple(self.context.get_binding_shape(out_idx))
            out = np.array(self.io[out_idx].host, copy=True).reshape(shape)
            outputs.append(out)
        return outputs


# ----------------------------
# YOLO postprocess (ONNX/TRT output kiểu (1, N, 5+nc))
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def yolo_nms(boxes, scores, iou_th=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_th]
    return keep


def iou(box, boxes):
    # box: (x1,y1,x2,y2), boxes: (M,4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def preprocess_bgr(frame: np.ndarray, input_size=640):
    """
    letterbox -> (1,3,640,640) float32, BGR->RGB, normalize 0..1
    return: tensor, ratio, pad
    """
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - nw) // 2
    pad_y = (input_size - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    return tensor, scale, pad_x, pad_y


def postprocess_yolo(outputs: List[np.ndarray], conf_th=0.25, iou_th=0.45,
                     scale=1.0, pad_x=0, pad_y=0, orig_w=1280, orig_h=720):
    """
    outputs[0] shape thường: (1, N, 5+nc)
    return list detections: (x1,y1,x2,y2, conf, cls)
    """
    out = outputs[0]
    if out.ndim == 3:
        pred = out[0]
    else:
        pred = out.reshape(-1, out.shape[-1])

    # pred: [cx,cy,w,h,obj, cls...]
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_id = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
    conf = obj * cls_conf

    mask = conf > conf_th
    pred = pred[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if pred.shape[0] == 0:
        return []

    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # undo letterbox
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = np.clip(x1, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    keep = yolo_nms(boxes, conf.astype(np.float32), iou_th=iou_th)
    dets = []
    for i in keep:
        dets.append((int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]),
                     float(conf[i]), int(cls_id[i])))
    return dets


# ----------------------------
# OCR decode: sort boxes -> string
# ----------------------------
def decode_plate_from_chars(dets_chars: List[Tuple[int,int,int,int,float,int]],
                            labels: List[str],
                            y_gap=18) -> str:
    """
    dets_chars: char boxes on crop
    sort by line (y), then x
    """
    if not dets_chars:
        return "unknown"

    # group into lines by y-center
    items = []
    for (x1,y1,x2,y2,conf,cls) in dets_chars:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ch = labels[cls] if cls < len(labels) else "?"
        items.append((cy, cx, ch, conf))

    items.sort(key=lambda t: t[0])  # sort by y
    lines = []
    for it in items:
        placed = False
        for ln in lines:
            if abs(ln["cy"] - it[0]) < y_gap:
                ln["items"].append(it)
                ln["cy"] = (ln["cy"] + it[0]) / 2
                placed = True
                break
        if not placed:
            lines.append({"cy": it[0], "items": [it]})

    # sort each line by x
    lines.sort(key=lambda ln: ln["cy"])
    text_lines = []
    for ln in lines[:2]:  # biển số thường 1-2 dòng
        ln["items"].sort(key=lambda t: t[1])
        text_lines.append("".join([t[2] for t in ln["items"]]))

    raw = " ".join(text_lines).strip()
    raw = raw.replace("..", ".").replace("--", "-")
    return raw if raw else "unknown"


# ----------------------------
# Main
# ----------------------------
def open_capture(args):
    if args.source == "csi":
        pipe = gst_csi_pipeline(
            sensor_id=args.cam,
            csi_w=args.csi_w, csi_h=args.csi_h, fps=args.csi_fps,
            flip=args.flip, out_w=args.out_w, out_h=args.out_h
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    if args.source == "rtsp":
        pipe = gst_rtsp_pipeline(
            args.rtsp, latency=args.rtsp_latency, tcp=not args.rtsp_udp,
            out_w=args.out_w, out_h=args.out_h
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    return cv2.VideoCapture(args.webcam)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    ap.add_argument("--show", type=int, default=1)

    # CSI
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--csi_w", type=int, default=1640)
    ap.add_argument("--csi_h", type=int, default=1232)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--out_w", type=int, default=1280)
    ap.add_argument("--out_h", type=int, default=720)

    # RTSP
    ap.add_argument("--rtsp", type=str, default="")
    ap.add_argument("--rtsp_latency", type=int, default=300)
    ap.add_argument("--rtsp_udp", action="store_true")  # default TCP

    # webcam
    ap.add_argument("--webcam", type=int, default=0)

    # TensorRT engines
    ap.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")

    # thresholds
    ap.add_argument("--det_conf", type=float, default=0.35)
    ap.add_argument("--ocr_conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.45)

    # labels
    ap.add_argument("--ocr_labels", type=str, default="")  # optional

    args = ap.parse_args()

    # labels
    labels = try_load_labels([
        args.ocr_labels,
        "model/ocr_labels.txt",
        "model/ocr.names",
        "model/classes_ocr.txt",
        "model/classes.txt",
    ]) or default_ocr_labels()

    print("[INFO] OCR labels =", len(labels))
    print("[INFO] det_engine =", args.det_engine)
    print("[INFO] ocr_engine =", args.ocr_engine)

    det_trt = TrtRunner(args.det_engine, logger_severity=trt.Logger.WARNING)
    ocr_trt = TrtRunner(args.ocr_engine, logger_severity=trt.Logger.WARNING)

    cap = open_capture(args)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Không mở được video source. Kiểm tra CSI/RTSP pipeline.")

    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        H, W = frame.shape[:2]

        # -------- DET plate --------
        det_in, scale, pad_x, pad_y = preprocess_bgr(frame, input_size=640)
        det_outs = det_trt.infer(det_in)
        plates = postprocess_yolo(det_outs, conf_th=args.det_conf, iou_th=args.nms,
                                 scale=scale, pad_x=pad_x, pad_y=pad_y, orig_w=W, orig_h=H)

        plates_count = 0

        for (x1,y1,x2,y2,conf,cls) in plates:
            # crop
            pad = 6
            x1p = max(0, x1-pad); y1p = max(0, y1-pad)
            x2p = min(W-1, x2+pad); y2p = min(H-1, y2+pad)
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            # -------- OCR char detect on crop --------
            ch_in, s2, px2, py2 = preprocess_bgr(crop, input_size=640)
            ch_outs = ocr_trt.infer(ch_in)

            ch_dets = postprocess_yolo(ch_outs, conf_th=args.ocr_conf, iou_th=args.nms,
                                      scale=s2, pad_x=px2, pad_y=py2,
                                      orig_w=crop.shape[1], orig_h=crop.shape[0])

            raw = decode_plate_from_chars(ch_dets, labels)
            text = format_vn_plate(raw) if raw != "unknown" else "unknown"

            if text != "unknown":
                plates_count += 1

            # draw bbox + text
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f} plates={plates_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        if args.show == 1:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
