#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webcam_onnx.py

ALPR realtime on Jetson (CSI / RTSP / USB webcam) using TensorRT engines.

Fixes for your case:
- Có CLI đúng: --source {csi,rtsp,webcam}
- Fix TensorRT Logger (TRT8)
- OCR không bị mismatch shape (auto đọc input shape từ engine)
- OCR decode kiểu CTC (collapse repeat + remove blank) => không còn ra chuỗi rác 5555....
- RTSP dùng GStreamer TCP + latency để tránh nhoè / decode error
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

# TensorRT + PyCUDA (Jetson thường có sẵn trong image l4t)
import tensorrt as trt
import pycuda.driver as cuda


# ----------------------------
# Utils
# ----------------------------
def load_charset(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    # cho phép file dạng "0123..." hoặc mỗi dòng 1 ký tự
    if "\n" in s:
        chars = [x.strip() for x in s.splitlines() if x.strip()]
        return "".join(chars)
    return s


def format_vn_plate(raw: str) -> str:
    """
    Heuristic format biển VN:
    - Ví dụ motor: 63B995164 -> 63-B9 951.64
    - Ví dụ motor 4 số cuối: 54S70661 -> 54-S7 0661
    - Nếu raw đã có '-' '.' thì giữ nguyên tương đối.
    """
    if not raw:
        return ""

    s = raw.upper()
    s = re.sub(r"[^0-9A-Z\-\.]", "", s)

    # Nếu đã có dấu, trả về luôn nhưng gọn lại khoảng trắng
    if "-" in s or "." in s:
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # motor phổ biến: 2 số + (A-Z)(0-9) + 3 số + 2 số
    m = re.match(r"^(\d{2})([A-Z]\d)(\d{3})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)} {m.group(3)}.{m.group(4)}"

    # motor: 2 số + (A-Z)(0-9) + 4 số
    m = re.match(r"^(\d{2})([A-Z]\d)(\d{4})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)} {m.group(3)}"

    # car dạng đơn giản: 2 số + (A-Z) + 5 số
    m = re.match(r"^(\d{2})([A-Z])(\d{5})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)} {m.group(3)}"

    return s


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    YOLO-style letterbox. returns resized_img, scale, pad (dw, dh)
    """
    h, w = img.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw, dh = new_w - nw, new_h - nh
    dw //= 2
    dh //= 2

    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    out[dh:dh + nh, dw:dw + nw] = img_resized
    return out, r, (dw, dh)


def nms_boxes(boxes_xyxy, scores, iou_thr=0.45):
    """
    OpenCV NMSBoxes expects xywh, so convert.
    """
    if len(boxes_xyxy) == 0:
        return []

    b = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        b.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    idxs = cv2.dnn.NMSBoxes(b, scores, score_threshold=0.0, nms_threshold=iou_thr)
    if idxs is None or len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


# ----------------------------
# TensorRT Runner
# ----------------------------
@dataclass
class Binding:
    name: str
    dtype: np.dtype
    shape: tuple
    is_input: bool
    host: np.ndarray
    device: int


class TrtRunner:
    """
    Minimal TensorRT engine runner (supports static + dynamic input shapes)
    """
    def __init__(self, engine_path: str, logger_severity=trt.Logger.WARNING):
        self.engine_path = engine_path
        self.logger = trt.Logger(logger_severity)
        self.runtime = trt.Runtime(self.logger)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.bindings = []
        self.binding_addrs = [None] * self.engine.num_bindings
        self.stream = cuda.Stream()

        # allocate with current (maybe -1) shapes later
        self._allocated = False

    def get_input_shape(self):
        # assume 1 input
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                shape = tuple(self.context.get_binding_shape(i))
                return i, shape, self.engine.get_binding_name(i)
        raise RuntimeError("No input binding")

    def _alloc_if_needed(self, input_shape=None):
        """
        Allocate buffers. If dynamic, set binding shape first.
        """
        if input_shape is not None:
            # set input shape if dynamic
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.context.set_binding_shape(i, tuple(input_shape))

        # now allocate based on context binding shapes
        self.bindings = []
        self.binding_addrs = [None] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))
            is_input = self.engine.binding_is_input(i)

            if any(s < 0 for s in shape):
                raise RuntimeError(f"Unresolved dynamic shape for binding {name}: {shape}")

            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)

            self.bindings.append(Binding(name=name, dtype=dtype, shape=shape, is_input=is_input,
                                         host=host, device=int(device)))
            self.binding_addrs[i] = int(device)

        self._allocated = True

    def infer(self, inp: np.ndarray):
        """
        inp must be contiguous float32 and match input binding shape
        returns list of outputs as numpy arrays with correct shapes
        """
        inp = np.ascontiguousarray(inp)

        # allocate buffers based on inp shape
        if (not self._allocated) or True:
            self._alloc_if_needed(input_shape=inp.shape)

        # copy input
        # assume first input binding is index where is_input True
        for b in self.bindings:
            if b.is_input:
                if int(np.prod(b.shape)) != inp.size:
                    raise RuntimeError(f"Input size mismatch. engine expects {b.shape} ({int(np.prod(b.shape))}), got {inp.shape} ({inp.size})")
                np.copyto(np.frombuffer(b.host, dtype=b.dtype, count=inp.size), inp.reshape(-1))
                cuda.memcpy_htod_async(b.device, b.host, self.stream)

        # execute
        ok = self.context.execute_async_v2(bindings=self.binding_addrs, stream_handle=int(self.stream.handle))
        if not ok:
            raise RuntimeError("TensorRT execute failed")

        # copy outputs back
        outputs = []
        for b in self.bindings:
            if not b.is_input:
                cuda.memcpy_dtoh_async(b.host, b.device, self.stream)

        self.stream.synchronize()

        for b in self.bindings:
            if not b.is_input:
                out = np.array(b.host, copy=True).reshape(b.shape)
                outputs.append(out)

        return outputs


# ----------------------------
# OCR Decoder (CTC greedy)
# ----------------------------
class CTCDecoder:
    def __init__(self, charset: str, blank_index: int = 0):
        self.charset = charset
        self.blank = blank_index

        # mapping index -> char:
        # if blank=0, chars start at 1
        self.idx2char = {}
        start = 0
        if self.blank == 0:
            start = 1
        for i, ch in enumerate(self.charset):
            self.idx2char[start + i] = ch

    def decode(self, logits: np.ndarray):
        """
        Supports logits shape:
        - (1, T, C)
        - (T, C)
        - (1, C, T)
        Returns (text, avg_conf)
        """
        x = logits
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        # if (C, T) -> transpose
        if x.ndim == 2:
            T, C = x.shape
            if C < 10 and T > C:
                # maybe (C, T)
                x = x.transpose(1, 0)

        if x.ndim != 2:
            # fallback flatten
            x = x.reshape(-1, x.shape[-1])

        # softmax over classes
        x = x.astype(np.float32)
        x = x - np.max(x, axis=1, keepdims=True)
        probs = np.exp(x)
        probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-9)

        idx = np.argmax(probs, axis=1).tolist()
        conf = np.max(probs, axis=1).tolist()

        # CTC collapse repeats + remove blank
        out = []
        out_conf = []
        prev = None
        for i, p in zip(idx, conf):
            if i == self.blank:
                prev = i
                continue
            if prev == i:
                continue
            ch = self.idx2char.get(i, "")
            if ch:
                out.append(ch)
                out_conf.append(p)
            prev = i

        if len(out) == 0:
            return "", 0.0
        return "".join(out), float(np.mean(out_conf))


# ----------------------------
# Video capture builders
# ----------------------------
def gst_csi(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink drop=1 sync=false max-buffers=1"
    )


def gst_rtsp(url, width=1280, height=720, latency=200, tcp=True):
    # TCP giảm packet loss => đỡ nhoè.
    proto = "tcp" if tcp else "udp"
    return (
        f"rtspsrc location={url} latency={latency} protocols={proto} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false max-buffers=1"
    )


def open_capture(args):
    if args.source == "csi":
        cap = cv2.VideoCapture(
            gst_csi(args.csi_id, args.csi_w, args.csi_h, args.csi_fps, args.flip),
            cv2.CAP_GSTREAMER,
        )
        return cap

    if args.source == "rtsp":
        cap = cv2.VideoCapture(
            gst_rtsp(args.rtsp, args.rtsp_w, args.rtsp_h, args.rtsp_latency, args.rtsp_tcp),
            cv2.CAP_GSTREAMER,
        )
        return cap

    # usb webcam
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_h)
    cap.set(cv2.CAP_PROP_FPS, args.webcam_fps)
    return cap


# ----------------------------
# Detector postprocess (YOLO-ish)
# ----------------------------
def decode_detector_output(out: np.ndarray, conf_thr=0.4):
    """
    Try to decode common YOLO export:
    out shape can be:
    - (1, N, 6) : [x, y, w, h, conf, cls] or [x1,y1,x2,y2,conf,cls]
    - (N, 6)
    - (1, N, 5+nc)
    This function returns list of (xyxy, score, cls).
    """
    x = out
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])

    if x.shape[1] < 6:
        return []

    res = []
    if x.shape[1] == 6:
        for row in x:
            a = row.astype(np.float32)
            conf = float(a[4])
            if conf < conf_thr:
                continue
            cls = int(a[5])
            # guess format
            x1, y1, x2, y2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
            # if looks like xywh (w/h positive, x2 smaller)
            if x2 <= x1 or y2 <= y1:
                cx, cy, w, h = x1, y1, x2, y2
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
            res.append(((x1, y1, x2, y2), conf, cls))
        return res

    # 5+nc (YOLO classic)
    # [cx,cy,w,h,obj, cls_probs...]
    nc = x.shape[1] - 5
    for row in x:
        a = row.astype(np.float32)
        obj = float(a[4])
        if obj < conf_thr:
            continue
        cls_scores = a[5:]
        cls = int(np.argmax(cls_scores))
        conf = float(obj * cls_scores[cls])
        if conf < conf_thr:
            continue
        cx, cy, w, h = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        res.append(((x1, y1, x2, y2), conf, cls))
    return res


# ----------------------------
# Main
# ----------------------------
def build_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="webcam",
                    help="input source")
    ap.add_argument("--show", type=int, default=1, help="show window (1/0)")

    # RTSP
    ap.add_argument("--rtsp", type=str, default="", help="rtsp url")
    ap.add_argument("--rtsp_w", type=int, default=1280)
    ap.add_argument("--rtsp_h", type=int, default=720)
    ap.add_argument("--rtsp_latency", type=int, default=200)
    ap.add_argument("--rtsp_tcp", type=int, default=1, help="1=tcp, 0=udp")

    # CSI
    ap.add_argument("--csi_id", type=int, default=0)
    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)

    # Webcam
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--webcam_w", type=int, default=1280)
    ap.add_argument("--webcam_h", type=int, default=720)
    ap.add_argument("--webcam_fps", type=int, default=30)

    # Models
    ap.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")

    # thresholds
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--nms", type=float, default=0.45)

    # OCR decode
    ap.add_argument("--ocr_blank", type=int, default=0)
    ap.add_argument("--ocr_charset", type=str, default="", help="optional charset file")
    ap.add_argument("--ocr_min_conf", type=float, default=0.35)

    return ap


def main(argv=None):
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()

    try:
        ap = build_argparser()
        args = ap.parse_args(argv)

        if args.source == "rtsp" and not args.rtsp:
            print("Bạn chọn --source rtsp nhưng chưa truyền --rtsp URL")
            return 2

        # Charset default: digits + A-Z
        charset = load_charset(args.ocr_charset)
        if not charset:
            charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Load engines
        det = TrtRunner(args.det_engine, logger_severity=trt.Logger.WARNING)
        ocr = TrtRunner(args.ocr_engine, logger_severity=trt.Logger.WARNING)

        # read model input shapes
        det_in_idx, det_in_shape, _ = det.get_input_shape()
        ocr_in_idx, ocr_in_shape, _ = ocr.get_input_shape()

        # expect NCHW
        det_h = int(det_in_shape[2]) if len(det_in_shape) == 4 else 640
        det_w = int(det_in_shape[3]) if len(det_in_shape) == 4 else 640

        # OCR shape may be dynamic; take abs if set
        if len(ocr_in_shape) == 4 and all(s > 0 for s in ocr_in_shape):
            ocr_h = int(ocr_in_shape[2])
            ocr_w = int(ocr_in_shape[3])
        else:
            # fallback commonly used
            ocr_h, ocr_w = 40, 160

        decoder = CTCDecoder(charset=charset, blank_index=args.ocr_blank)

        cap = open_capture(args)
        if not cap.isOpened():
            print("❌ Không mở được nguồn video. Check lại CSI/RTSP/USB, và docker run có --runtime nvidia + X11 chưa.")
            return 1

        win = "ALPR"
        if args.show:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        last = time.time()
        fps = 0.0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # RTSP đôi khi rớt frame, chờ nhẹ
                time.sleep(0.01)
                continue

            # FPS
            now = time.time()
            dt = now - last
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last = now

            # detector preprocess
            img_lb, r, (dw, dh) = letterbox(frame, new_shape=(det_w, det_h))
            img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
            inp = img_rgb.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))  # HWC -> CHW
            inp = np.expand_dims(inp, axis=0)   # NCHW

            det_outs = det.infer(inp)
            det_out = det_outs[0]

            cand = decode_detector_output(det_out, conf_thr=args.conf)

            # scale back to original frame
            boxes = []
            scores = []
            clses = []
            for (x1, y1, x2, y2), conf, cls in cand:
                # coords are in letterbox image space (det_w/det_h)
                x1 = (x1 - dw) / r
                y1 = (y1 - dh) / r
                x2 = (x2 - dw) / r
                y2 = (y2 - dh) / r

                x1 = max(0, min(frame.shape[1] - 1, x1))
                y1 = max(0, min(frame.shape[0] - 1, y1))
                x2 = max(0, min(frame.shape[1] - 1, x2))
                y2 = max(0, min(frame.shape[0] - 1, y2))

                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                clses.append(cls)

            keep = nms_boxes(boxes, scores, iou_thr=args.nms)

            plates = 0
            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                conf = scores[i]

                # crop plate
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue

                # OCR preprocess based on engine input
                # resize to (ocr_w, ocr_h)
                crop_resized = cv2.resize(crop, (ocr_w, ocr_h), interpolation=cv2.INTER_LINEAR)
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                oinp = crop_rgb.astype(np.float32) / 255.0
                oinp = np.transpose(oinp, (2, 0, 1))
                oinp = np.expand_dims(oinp, 0)

                ocr_outs = ocr.infer(oinp)
                ocr_logits = ocr_outs[0]

                text_raw, tconf = decoder.decode(ocr_logits)
                text = format_vn_plate(text_raw)
                if tconf < args.ocr_min_conf:
                    text = ""  # confidence thấp -> khỏi show bậy

                # draw
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                label = text if text else "..."
                # put label above box
                ty = max(0, int(y1) - 10)
                cv2.putText(frame, label, (int(x1), ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

                plates += 1

            # HUD
            cv2.putText(frame, f"FPS {fps:.1f} plates={plates}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q'):
                    break

        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        return 0

    finally:
        # IMPORTANT: release CUDA context cleanly
        try:
            ctx.pop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
