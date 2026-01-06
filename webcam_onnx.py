#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
from dataclasses import dataclass
import numpy as np
import cv2

# =========================
# Utils
# =========================
def log(*a):
    print("[INFO]", *a, flush=True)

def warn(*a):
    print("[WARN]", *a, flush=True)

def err(*a):
    print("[ERROR]", *a, flush=True)

def clamp(x, a, b):
    return max(a, min(b, x))

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    Resize + pad like YOLOv5 letterbox. Return: img, ratio, (dw, dh)
    """
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x):
    # x: [..., 4] => (cx, cy, w, h) -> (x1, y1, x2, y2)
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms_numpy(boxes, scores, iou_thres=0.45):
    """
    boxes: Nx4 (x1,y1,x2,y2)
    scores: N
    return indices kept
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

# =========================
# TensorRT runner (preferred)
# =========================
class TRTInfer:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.ok = False
        self.trt = None
        self.cuda = None
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = None
        self.host_mem = {}
        self.device_mem = {}
        self.binding_names = []
        self.input_bindings = []
        self.output_bindings = []

        try:
            import tensorrt as trt  # type: ignore
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # noqa: F401
            self.trt = trt
            self.cuda = cuda
        except Exception as e:
            warn("TensorRT/PyCUDA not available:", e)
            return

        if not os.path.exists(engine_path):
            warn("Engine not found:", engine_path)
            return

        logger = self.trt.Logger(self.trt.Logger.WARNING)
        with open(engine_path, "rb") as f, self.trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            warn("Failed to load engine:", engine_path)
            return

        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.Stream()

        self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_bindings.append(i)
            else:
                self.output_bindings.append(i)

        self.bindings = [None] * self.engine.num_bindings
        self.ok = True
        log("TensorRT engine loaded:", engine_path)
        log("Bindings:", self.binding_names)

    def _alloc_if_needed(self, idx, shape, dtype):
        # allocate host/device buffers based on shape/dtype
        vol = int(np.prod(shape))
        nbytes = vol * np.dtype(dtype).itemsize

        if idx in self.host_mem and self.host_mem[idx].nbytes == nbytes:
            return

        self.host_mem[idx] = self.cuda.pagelocked_empty(vol, dtype)
        self.device_mem[idx] = self.cuda.mem_alloc(nbytes)
        self.bindings[idx] = int(self.device_mem[idx])

    def infer(self, input_array: np.ndarray):
        """
        input_array: np.ndarray float32 with shape (1,3,H,W)
        returns list of output arrays (np.ndarray)
        """
        assert self.ok
        assert input_array.ndim == 4

        # set input binding shape (dynamic-safe)
        inp_idx = self.input_bindings[0]
        self.context.set_binding_shape(inp_idx, input_array.shape)

        # allocate input
        self._alloc_if_needed(inp_idx, input_array.shape, np.float32)
        np.copyto(self.host_mem[inp_idx], input_array.ravel())

        # allocate outputs
        outputs = []
        for out_idx in self.output_bindings:
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            out_dtype = np.float32  # most YOLO exports are fp32 output even if fp16 engine
            self._alloc_if_needed(out_idx, out_shape, out_dtype)
            outputs.append((out_idx, out_shape, out_dtype))

        # H2D input
        self.cuda.memcpy_htod_async(self.device_mem[inp_idx], self.host_mem[inp_idx], self.stream)

        # execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # D2H outputs
        out_arrays = []
        for out_idx, out_shape, out_dtype in outputs:
            self.cuda.memcpy_dtoh_async(self.host_mem[out_idx], self.device_mem[out_idx], self.stream)

        self.stream.synchronize()

        for out_idx, out_shape, out_dtype in outputs:
            out = np.array(self.host_mem[out_idx], dtype=out_dtype).reshape(out_shape)
            out_arrays.append(out)

        return out_arrays

# =========================
# ONNXRuntime fallback (optional)
# =========================
class ORTInfer:
    def __init__(self, onnx_path: str):
        self.ok = False
        self.sess = None
        self.input_name = None
        self.input_hw = None
        self.onnx_path = onnx_path
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            warn("onnxruntime not available:", e)
            return

        if not os.path.exists(onnx_path):
            warn("ONNX not found:", onnx_path)
            return

        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        ish = self.sess.get_inputs()[0].shape  # [1,3,H,W] maybe dynamic
        if len(ish) == 4 and isinstance(ish[2], int) and isinstance(ish[3], int):
            self.input_hw = (ish[2], ish[3])
        self.ok = True
        log("ONNXRuntime loaded:", onnx_path)

    def infer(self, input_array: np.ndarray):
        assert self.ok
        out = self.sess.run(None, {self.input_name: input_array})
        return out

# =========================
# Pipelines
# =========================
def gst_csi(sensor_id=0, sensor_mode=3, capture_w=1280, capture_h=720, framerate=30, flip=0):
    # appsink BGR for OpenCV
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={capture_w}, height={capture_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def gst_rtsp(rtsp_url, latency=200, codec="h264"):
    # decode using nvv4l2decoder for speed
    # NOTE: many RTSP cameras deliver H264
    depay = "rtph264depay" if codec.lower() == "h264" else "rtph265depay"
    parse = "h264parse" if codec.lower() == "h264" else "h265parse"
    return (
        f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp ! "
        f"{depay} ! {parse} ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def open_capture(src: str, rtsp_url=None):
    if src == "csi":
        sensor_id = int(os.getenv("CSI_SENSOR_ID", "0"))
        sensor_mode = int(os.getenv("CSI_SENSOR_MODE", "3"))  # 3 = 1640x1232@30 (thường ổn), 5=1280x720@120
        w = int(os.getenv("CSI_W", "1280"))
        h = int(os.getenv("CSI_H", "720"))
        fps = int(os.getenv("CSI_FPS", "30"))  # ép fps xuống cho mượt
        flip = int(os.getenv("CSI_FLIP", "0"))
        pipe = gst_csi(sensor_id, sensor_mode, w, h, fps, flip)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    if src == "rtsp":
        if not rtsp_url:
            rtsp_url = os.getenv("RTSP_URL", "")
        if not rtsp_url:
            raise ValueError("RTSP_URL is empty")
        latency = int(os.getenv("RTSP_LATENCY", "200"))
        codec = os.getenv("RTSP_CODEC", "h264")
        pipe = gst_rtsp(rtsp_url, latency=latency, codec=codec)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    # default webcam (USB)
    cam_index = int(os.getenv("CAM_INDEX", "0"))
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.getenv("WEBCAM_W", "1280")))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.getenv("WEBCAM_H", "720")))
    cap.set(cv2.CAP_PROP_FPS, int(os.getenv("WEBCAM_FPS", "30")))
    return cap

# =========================
# Decode YOLOv5 export
# - Supports output: (1,25200,6) for 1 class => [x,y,w,h,obj,cls]
# - Supports output: (1,25200,5+ncls) => [x,y,w,h,obj,cls...]
# =========================
def decode_yolov5(pred, conf_thres=0.25, iou_thres=0.45):
    """
    pred: np.ndarray (1, N, C)
    returns: boxes_xyxy (M,4), scores (M,), class_ids (M,)
    """
    if pred is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    if pred.ndim == 3:
        pred = pred[0]  # (N,C)
    if pred.ndim != 2:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    n, c = pred.shape
    if c < 6:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    xywh = pred[:, 0:4]
    obj = pred[:, 4:5]

    if c == 6:
        # 1-class (common for plate detector)
        cls_prob = pred[:, 5:6]
        conf = (obj * cls_prob).reshape(-1)
        cls_ids = np.zeros((n,), dtype=np.int32)
    else:
        cls_probs = pred[:, 5:]
        cls_ids = np.argmax(cls_probs, axis=1).astype(np.int32)
        cls_conf = cls_probs[np.arange(n), cls_ids]
        conf = (obj.reshape(-1) * cls_conf.reshape(-1)).astype(np.float32)

    mask = conf >= conf_thres
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    xywh = xywh[mask]
    conf = conf[mask]
    cls_ids = cls_ids[mask]

    boxes = xywh2xyxy(xywh).astype(np.float32)
    keep = nms_numpy(boxes, conf, iou_thres=iou_thres)
    boxes = boxes[keep]
    conf = conf[keep]
    cls_ids = cls_ids[keep]
    return boxes, conf, cls_ids

def scale_boxes_from_letterbox(boxes_xyxy, r, dwdh):
    """
    boxes in letterbox image coords -> original image coords
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    dw, dh = dwdh
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= r
    return boxes

def preprocess_bgr_to_nchw_float(im_bgr, size_hw):
    """
    size_hw: (H,W)
    returns: (1,3,H,W) float32 normalized 0..1
    """
    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # CHW
    im = np.expand_dims(im, 0)  # NCHW
    return im

# =========================
# OCR post-process
# =========================
CHARS_36 = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def build_text_from_char_boxes(boxes, confs, cls_ids, img_h, split_two_lines=True):
    """
    boxes: (M,4) xyxy in OCR image coords
    returns text string
    """
    if len(boxes) == 0:
        return ""

    # center points
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    # map classes -> chars
    ncls = int(cls_ids.max()) + 1 if len(cls_ids) else 0
    charset = CHARS_36.copy()
    if ncls > len(charset):
        charset += ["?"] * (ncls - len(charset))

    chars = [charset[i] if i < len(charset) else "?" for i in cls_ids.tolist()]

    # split into 2 lines if needed (VN plate often 2 lines)
    if split_two_lines and len(boxes) >= 6:
        # heuristic: if y-range big enough -> split
        y_range = float(cy.max() - cy.min())
        if y_range > 0.18 * img_h:
            y_mid = float(np.median(cy))
            top_idx = np.where(cy <= y_mid)[0]
            bot_idx = np.where(cy > y_mid)[0]

            def line_text(idxs):
                if len(idxs) == 0:
                    return ""
                order = idxs[np.argsort(cx[idxs])]
                return "".join([chars[i] for i in order])

            t1 = line_text(top_idx)
            t2 = line_text(bot_idx)
            # format a bit
            if t1 and t2:
                return f"{t1}-{t2}"
            return (t1 + t2).strip()

    # one line
    order = np.argsort(cx)
    text = "".join([chars[i] for i in order])
    return text.strip()

# =========================
# Main loop
# =========================
@dataclass
class Config:
    src: str
    show: bool
    img_size: int
    ocr_size: int
    conf: float
    iou: float
    ocr_conf: float
    ocr_iou: float
    skip: int
    ocr_every: int
    det_engine: str
    ocr_engine: str
    det_onnx: str
    ocr_onnx: str
    window_name: str

def build_config_from_env_and_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.getenv("SRC", "csi"), choices=["csi", "rtsp", "webcam"])
    ap.add_argument("--show", default=int(os.getenv("SHOW", "1")))
    ap.add_argument("--img", default=int(os.getenv("IMG_SIZE", "640")))
    ap.add_argument("--ocr", default=int(os.getenv("OCR_SIZE", "320")))
    ap.add_argument("--conf", default=float(os.getenv("CONF", "0.25")))
    ap.add_argument("--iou", default=float(os.getenv("IOU", "0.45")))
    ap.add_argument("--ocr-conf", default=float(os.getenv("OCR_CONF", "0.25")))
    ap.add_argument("--ocr-iou", default=float(os.getenv("OCR_IOU", "0.45")))
    ap.add_argument("--skip", default=int(os.getenv("SKIP", "0")))
    ap.add_argument("--ocr-every", default=int(os.getenv("OCR_EVERY", "3")))
    args = ap.parse_args()

    cfg = Config(
        src=args.src,
        show=bool(args.show),
        img_size=int(args.img),
        ocr_size=int(args.ocr),
        conf=float(args.conf),
        iou=float(args.iou),
        ocr_conf=float(args.ocr_conf),
        ocr_iou=float(args.ocr_iou),
        skip=int(args.skip),
        ocr_every=int(args.ocr_every),
        det_engine=os.getenv("DET_ENGINE", "./model/LP_detector_nano_61_fp16.engine"),
        ocr_engine=os.getenv("OCR_ENGINE", "./model/LP_ocr_nano_62_fp16.engine"),
        det_onnx=os.getenv("DET_ONNX", "./model/LP_detector_nano_61.onnx"),
        ocr_onnx=os.getenv("OCR_ONNX", "./model/LP_ocr_nano_62.onnx"),
        window_name=os.getenv("WINDOW_NAME", f"{args.src.upper()}-LPR"),
    )
    return cfg

def main():
    cfg = build_config_from_env_and_args()

    log(f"SRC={cfg.src} SHOW={cfg.show} IMG_SIZE={cfg.img_size} OCR_SIZE={cfg.ocr_size} CONF={cfg.conf} IOU={cfg.iou} SKIP={cfg.skip} OCR_EVERY={cfg.ocr_every}")
    log("DET_ENGINE=", cfg.det_engine)
    log("OCR_ENGINE=", cfg.ocr_engine)
    log("DET_ONNX=", cfg.det_onnx)
    log("OCR_ONNX=", cfg.ocr_onnx)

    # load DET backend
    det_backend = None
    if os.path.exists(cfg.det_engine):
        det_backend = TRTInfer(cfg.det_engine)
        if not det_backend.ok:
            det_backend = None
    if det_backend is None and os.path.exists(cfg.det_onnx):
        det_backend = ORTInfer(cfg.det_onnx)
        if not det_backend.ok:
            det_backend = None

    # load OCR backend
    ocr_backend = None
    if os.path.exists(cfg.ocr_engine):
        ocr_backend = TRTInfer(cfg.ocr_engine)
        if not ocr_backend.ok:
            ocr_backend = None
    if ocr_backend is None and os.path.exists(cfg.ocr_onnx):
        ocr_backend = ORTInfer(cfg.ocr_onnx)
        if not ocr_backend.ok:
            ocr_backend = None

    if det_backend is None:
        raise RuntimeError("No DET backend available. Make sure DET_ENGINE or DET_ONNX exists.")
    if ocr_backend is None:
        warn("No OCR backend available -> will only draw plate boxes (no text).")

    # open capture
    rtsp_url = os.getenv("RTSP_URL", "")
    cap = open_capture(cfg.src, rtsp_url=rtsp_url)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source. Check CSI/RTSP pipeline and docker X11 mounts.")

    # warmup (optional)
    last_plate_text = ""
    last_ocr_time = 0.0
    frame_i = 0

    # FPS smoothing
    t_prev = time.time()
    fps = 0.0

    log("Start loop. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            warn("Frame read failed. Re-opening capture in 1s ...")
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(1.0)
            cap = open_capture(cfg.src, rtsp_url=rtsp_url)
            continue

        frame_i += 1
        if cfg.skip > 0 and (frame_i % (cfg.skip + 1) != 0):
            continue

        h0, w0 = frame.shape[:2]

        # ---- DET preprocess
        det_in, r, dwdh = letterbox(frame, new_shape=(cfg.img_size, cfg.img_size), auto=False, scaleFill=False, scaleup=True, stride=32)
        inp = preprocess_bgr_to_nchw_float(det_in, (cfg.img_size, cfg.img_size))

        # ---- DET infer
        if isinstance(det_backend, TRTInfer):
            det_outs = det_backend.infer(inp)
        else:
            det_outs = det_backend.infer(inp)

        # take first output
        det_pred = det_outs[0]
        det_boxes, det_scores, det_cls = decode_yolov5(det_pred, conf_thres=cfg.conf, iou_thres=cfg.iou)
        det_boxes = scale_boxes_from_letterbox(det_boxes, r, dwdh)

        # clamp boxes
        det_boxes[:, [0, 2]] = np.clip(det_boxes[:, [0, 2]], 0, w0 - 1)
        det_boxes[:, [1, 3]] = np.clip(det_boxes[:, [1, 3]], 0, h0 - 1)

        # ---- OCR (only on best plate) every N frames
        plate_text = last_plate_text
        best_idx = int(np.argmax(det_scores)) if len(det_scores) else -1

        if ocr_backend is not None and best_idx >= 0:
            do_ocr = (frame_i % max(1, cfg.ocr_every) == 0)
            if do_ocr:
                x1, y1, x2, y2 = det_boxes[best_idx].astype(int).tolist()
                # margin
                mx = int(0.03 * (x2 - x1 + 1))
                my = int(0.08 * (y2 - y1 + 1))
                x1m = clamp(x1 - mx, 0, w0 - 1)
                y1m = clamp(y1 - my, 0, h0 - 1)
                x2m = clamp(x2 + mx, 0, w0 - 1)
                y2m = clamp(y2 + my, 0, h0 - 1)
                crop = frame[y1m:y2m, x1m:x2m].copy()
                if crop.size > 0:
                    ocr_in, rr, dd = letterbox(crop, new_shape=(cfg.ocr_size, cfg.ocr_size), auto=False)
                    ocr_inp = preprocess_bgr_to_nchw_float(ocr_in, (cfg.ocr_size, cfg.ocr_size))

                    if isinstance(ocr_backend, TRTInfer):
                        ocr_outs = ocr_backend.infer(ocr_inp)
                    else:
                        ocr_outs = ocr_backend.infer(ocr_inp)

                    ocr_pred = ocr_outs[0]
                    ocr_boxes, ocr_scores, ocr_cls = decode_yolov5(ocr_pred, conf_thres=cfg.ocr_conf, iou_thres=cfg.ocr_iou)
                    # boxes are in ocr letterbox coords -> keep in that space for sorting
                    plate_text = build_text_from_char_boxes(ocr_boxes, ocr_scores, ocr_cls, img_h=cfg.ocr_size, split_two_lines=True)

                    last_plate_text = plate_text
                    last_ocr_time = time.time()

        # ---- Draw
        overlay = frame.copy()

        # show all plate boxes
        for i in range(len(det_boxes)):
            x1, y1, x2, y2 = det_boxes[i].astype(int).tolist()

            # red fill (tô đỏ) + green thin border
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # fill red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)     # green border thin

            score = float(det_scores[i]) if i < len(det_scores) else 0.0
            # write red text only on best box
            if i == best_idx:
                txt = plate_text if plate_text else f"plate {score:.2f}"
                cv2.putText(frame, txt, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # blend overlay (transparent red)
        alpha = float(os.getenv("PLATE_ALPHA", "0.22"))
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # FPS
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        inst_fps = 1.0 / max(dt, 1e-6)
        fps = fps * 0.9 + inst_fps * 0.1

        cv2.putText(frame, f"FPS {fps:.1f} plates={len(det_boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if cfg.show:
            cv2.imshow(cfg.window_name, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
        else:
            # headless mode
            if frame_i % 30 == 0:
                log(f"FPS~{fps:.1f} plates={len(det_boxes)} text='{plate_text}'")

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    log("Exit.")

if __name__ == "__main__":
    main()
