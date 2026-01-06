#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson Nano ALPR (CSI / RTSP / Webcam) with TensorRT engines.

- Uses TensorRT .engine if present, else falls back to OpenCV DNN CPU.
- Fixes OCR input channel bug (model expects 3-ch, not 1-ch).
- Adds red plate highlight, thin green border, red text.

Run:
  python3 csi.py
  RTSP_URL='rtsp://...' python3 rtsp.py
  SRC=webcam python3 webcam_onnx.py
"""

import os
import time
import cv2
import numpy as np

# -------------------------
# ENV CONFIG
# -------------------------
SRC = os.environ.get("SRC", "csi")  # csi | rtsp | webcam
SHOW = os.environ.get("SHOW", "1") == "1"

IMG_SIZE = int(os.environ.get("IMG_SIZE", "640"))
CONF_THRES = float(os.environ.get("CONF", "0.25"))
IOU_THRES = float(os.environ.get("IOU", "0.45"))
SKIP = int(os.environ.get("SKIP", "0"))  # 0=detect every frame, 2=detect each 3 frames

# CSI config
CSI_WIDTH = int(os.environ.get("CSI_WIDTH", "1280"))
CSI_HEIGHT = int(os.environ.get("CSI_HEIGHT", "720"))
CSI_FPS = int(os.environ.get("CSI_FPS", "30"))
CSI_SENSOR_MODE = os.environ.get("CSI_SENSOR_MODE", "")  # "" to omit, else e.g. "3"

# RTSP config
RTSP_URL = os.environ.get("RTSP_URL", "")
RTSP_LATENCY = int(os.environ.get("RTSP_LATENCY", "100"))
RTSP_CODEC = os.environ.get("RTSP_CODEC", "h264")  # h264|h265

# Model paths
DET_ONNX = os.environ.get("DET_ONNX", "model/LP_detector_nano_61.onnx")
OCR_ONNX = os.environ.get("OCR_ONNX", "model/LP_ocr_nano_62.onnx")

DET_ENGINE = os.environ.get("DET_ENGINE", "model/LP_detector_nano_61_fp16.engine")
OCR_ENGINE = os.environ.get("OCR_ENGINE", "model/LP_ocr_nano_62_fp16.engine")

# OCR input size
OCR_W = int(os.environ.get("OCR_W", "160"))
OCR_H = int(os.environ.get("OCR_H", "40"))

# Draw style
BOX_THICKNESS = int(os.environ.get("BOX_THICKNESS", "2"))  # green border
FILL_ALPHA = float(os.environ.get("FILL_ALPHA", "0.25"))  # red fill alpha
TEXT_SCALE = float(os.environ.get("TEXT_SCALE", "0.8"))
TEXT_THICKNESS = int(os.environ.get("TEXT_THICKNESS", "2"))

# Charset (CTC-style, blank=0)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_ID = 0

# -------------------------
# Utils
# -------------------------
def logi(msg):
    print("[INFO]", msg, flush=True)

def logw(msg):
    print("[WARN]", msg, flush=True)

def has_file(p):
    try:
        return os.path.isfile(p) and os.path.getsize(p) > 0
    except Exception:
        return False

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    # YOLO-like letterbox
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

def nms_boxes(boxes, scores, iou_thres=0.45):
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
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def draw_plate(frame, box, text):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # red fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    frame[:] = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1.0 - FILL_ALPHA, 0)

    # green border (thin)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)

    # red text
    if text:
        ty = y1 - 10 if y1 - 10 > 10 else y2 + 25
        cv2.putText(
            frame,
            str(text),
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (0, 0, 255),
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

# -------------------------
# Video sources
# -------------------------
def gst_csi_pipeline(width, height, fps, sensor_mode=""):
    sm = f" sensor-mode={sensor_mode}" if str(sensor_mode).strip() != "" else ""
    # appsink: drop old frames to reduce lag
    return (
        f"nvarguscamerasrc{sm} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def gst_rtsp_pipeline(url, codec="h264", latency=100):
    codec = codec.lower().strip()
    depay = "rtph264depay" if codec == "h264" else "rtph265depay"
    parse = "h264parse" if codec == "h264" else "h265parse"
    # Use HW decoder nvv4l2decoder for Jetson
    return (
        f"rtspsrc location={url} latency={latency} ! "
        f"{depay} ! {parse} ! "
        f"nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

def open_source():
    if SRC == "csi":
        pipe = gst_csi_pipeline(CSI_WIDTH, CSI_HEIGHT, CSI_FPS, CSI_SENSOR_MODE)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap
    if SRC == "rtsp":
        if not RTSP_URL:
            raise RuntimeError("SRC=rtsp nhưng chưa set RTSP_URL")
        pipe = gst_rtsp_pipeline(RTSP_URL, RTSP_CODEC, RTSP_LATENCY)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logw("GStreamer RTSP mở không được, fallback sang cv2.VideoCapture(url)")
            cap = cv2.VideoCapture(RTSP_URL)
        return cap
    # webcam
    cam_index = int(os.environ.get("WEBCAM_INDEX", "0"))
    return cv2.VideoCapture(cam_index)

# -------------------------
# Backends (TensorRT + OpenCV DNN fallback)
# -------------------------
class TRTInfer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.ok = False
        self.trt = None
        self.cuda = None
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = None
        self.host_inputs = {}
        self.dev_inputs = {}
        self.host_outputs = {}
        self.dev_outputs = {}
        self.binding_names = []
        self.input_names = []
        self.output_names = []

        self._load()

    def _load(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            self.trt = trt
            self.cuda = cuda

            logger = trt.Logger(trt.Logger.INFO)
            runtime = trt.Runtime(logger)
            with open(self.engine_path, "rb") as f:
                engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError("deserialize_cuda_engine failed")

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

            # allocate once using current binding shapes (fixed-shape engines)
            self._allocate()
            self.ok = True
            logi(f"TensorRT loaded: {self.engine_path}")
        except Exception as e:
            logw(f"TensorRT load failed ({self.engine_path}): {e}")
            self.ok = False

    def _allocate(self):
        trt = self.trt
        cuda = self.cuda

        self.bindings = [None] * self.engine.num_bindings

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))

            # dynamic shape: must set it before allocate (we keep standard shapes)
            if any([d < 0 for d in shape]):
                # fallback: set known shapes for our models
                if self.engine.binding_is_input(i):
                    if "det" in name.lower() or "images" in name.lower() or "input" in name.lower():
                        shape = (1, 3, IMG_SIZE, IMG_SIZE)
                        self.context.set_binding_shape(i, shape)
                    else:
                        shape = (1, 3, OCR_H, OCR_W)
                        self.context.set_binding_shape(i, shape)
                shape = tuple(self.context.get_binding_shape(i))

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings[i] = int(dev_mem)

            if self.engine.binding_is_input(i):
                self.host_inputs[name] = host_mem
                self.dev_inputs[name] = dev_mem
            else:
                self.host_outputs[name] = host_mem
                self.dev_outputs[name] = dev_mem

    def infer(self, feed_dict):
        """feed_dict: {input_name: np.ndarray (NCHW)} -> outputs dict"""
        if not self.ok:
            raise RuntimeError("TRT not ready")

        cuda = self.cuda
        trt = self.trt

        # copy inputs
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if not self.engine.binding_is_input(i):
                continue
            if name not in feed_dict:
                # allow single-input engine: take first provided
                if len(feed_dict) == 1:
                    arr = list(feed_dict.values())[0]
                else:
                    raise KeyError(f"Missing input: {name}")
            else:
                arr = feed_dict[name]

            # ensure dtype matches binding
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)

            # set dynamic shape if needed
            if self.context.get_binding_shape(i) != tuple(arr.shape):
                self.context.set_binding_shape(i, tuple(arr.shape))

            host_mem = self.host_inputs[name]
            np.copyto(host_mem, arr.ravel())
            cuda.memcpy_htod_async(self.dev_inputs[name], host_mem, self.stream)

        # execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=int(self.stream.handle))

        # copy outputs back
        outputs = {}
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                continue
            name = self.engine.get_binding_name(i)
            cuda.memcpy_dtoh_async(self.host_outputs[name], self.dev_outputs[name], self.stream)

        self.stream.synchronize()

        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                continue
            name = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            out = np.array(self.host_outputs[name], dtype=dtype).reshape(shape)
            outputs[name] = out
        return outputs

class CvDnnONNX:
    """OpenCV DNN CPU fallback (slower)."""
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.net = None
        self.ok = False
        try:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.ok = True
            logw(f"Using OpenCV DNN CPU for {onnx_path} (slow)")
        except Exception as e:
            logw(f"OpenCV DNN load failed: {onnx_path}: {e}")
            self.ok = False

    def infer(self, blob):
        self.net.setInput(blob)
        out = self.net.forward()
        return out

# -------------------------
# Models: Detector + OCR
# -------------------------
class PlateDetector:
    def __init__(self):
        self.trt = TRTInfer(DET_ENGINE) if has_file(DET_ENGINE) else None
        self.cv = CvDnnONNX(DET_ONNX) if has_file(DET_ONNX) else None
        if self.trt and self.trt.ok:
            self.backend = "trt"
        elif self.cv and self.cv.ok:
            self.backend = "cv"
        else:
            raise RuntimeError("No detector backend available (need DET_ENGINE or DET_ONNX)")

    def preprocess(self, frame):
        img, r, (padw, padh) = letterbox(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_norm, (2, 0, 1))[None, ...]  # 1,3,H,W
        return chw, r, padw, padh

    def postprocess(self, pred, r, padw, padh, orig_shape):
        # pred: (1, N, 6) => xywh + conf + cls (cls ignored)
        if pred is None:
            return []
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[-1] < 5:
            return []

        xywh = pred[:, :4].astype(np.float32)
        conf = pred[:, 4].astype(np.float32)

        keep = conf >= CONF_THRES
        xywh = xywh[keep]
        conf = conf[keep]
        if xywh.shape[0] == 0:
            return []

        # xywh are in letterbox image coords
        x = xywh[:, 0]
        y = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # undo letterbox
        x1 = (x1 - padw) / r
        y1 = (y1 - padh) / r
        x2 = (x2 - padw) / r
        y2 = (y2 - padh) / r

        oh, ow = orig_shape[:2]
        boxes = np.stack([
            np.clip(x1, 0, ow - 1),
            np.clip(y1, 0, oh - 1),
            np.clip(x2, 0, ow - 1),
            np.clip(y2, 0, oh - 1),
        ], axis=1)

        keep_idx = nms_boxes(boxes, conf, IOU_THRES)
        boxes = boxes[keep_idx]
        conf2 = conf[keep_idx]
        # sort by confidence desc
        order = np.argsort(-conf2)
        boxes = boxes[order]
        conf2 = conf2[order]
        return [(boxes[i], float(conf2[i])) for i in range(len(boxes))]

    def detect(self, frame):
        inp, r, padw, padh = self.preprocess(frame)

        if self.backend == "trt":
            # engine input name might be unknown => pass first
            outs = self.trt.infer({self.trt.input_names[0]: inp})
            pred = outs[self.trt.output_names[0]]
        else:
            blob = cv2.dnn.blobFromImage(
                cv2.cvtColor(letterbox(frame, IMG_SIZE)[0], cv2.COLOR_BGR2RGB),
                scalefactor=1.0 / 255.0,
                size=(IMG_SIZE, IMG_SIZE),
                swapRB=False,
                crop=False
            )
            pred = self.cv.infer(blob)

        return self.postprocess(pred, r, padw, padh, frame.shape)

class PlateOCR:
    def __init__(self):
        self.trt = TRTInfer(OCR_ENGINE) if has_file(OCR_ENGINE) else None
        self.cv = CvDnnONNX(OCR_ONNX) if has_file(OCR_ONNX) else None
        if self.trt and self.trt.ok:
            self.backend = "trt"
        elif self.cv and self.cv.ok:
            self.backend = "cv"
        else:
            raise RuntimeError("No OCR backend available (need OCR_ENGINE or OCR_ONNX)")

    def preprocess(self, crop_bgr):
        # FIX: OCR model expects 3 channels (not grayscale)
        crop = cv2.resize(crop_bgr, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        chw = np.transpose(x, (2, 0, 1))[None, ...]  # 1,3,H,W
        return chw

    def ctc_decode(self, logits):
        # logits expected shape: (1, T, C) or (T, C) or (1, C, T)
        arr = logits
        arr = np.array(arr)

        if arr.ndim == 3 and arr.shape[0] == 1:
            # could be (1,T,C) or (1,C,T)
            if arr.shape[1] > 10 and arr.shape[2] <= 128:
                # (1,T,C)
                seq = arr[0]
            else:
                # (1,C,T) -> (T,C)
                seq = np.transpose(arr[0], (1, 0))
        elif arr.ndim == 2:
            seq = arr
        else:
            seq = arr.reshape(-1, arr.shape[-1])

        ids = np.argmax(seq, axis=1).tolist()
        # collapse repeats, remove blank
        out = []
        prev = None
        for i in ids:
            if i == prev:
                continue
            prev = i
            if i == BLANK_ID:
                continue
            # map id->char (blank=0, chars start at 1)
            j = i - 1
            if 0 <= j < len(CHARS):
                out.append(CHARS[j])
        return "".join(out)

    def recognize(self, frame, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        # add padding a bit
        pad = 6
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        inp = self.preprocess(crop)

        if self.backend == "trt":
            outs = self.trt.infer({self.trt.input_names[0]: inp})
            logits = outs[self.trt.output_names[0]]
        else:
            # OpenCV blob must be 3-ch
            blob = cv2.dnn.blobFromImage(
                cv2.cvtColor(cv2.resize(crop, (OCR_W, OCR_H)), cv2.COLOR_BGR2RGB),
                scalefactor=1.0 / 255.0,
                size=(OCR_W, OCR_H),
                swapRB=False,
                crop=False
            )
            logits = self.cv.infer(blob)

        return self.ctc_decode(logits)

# -------------------------
# Main
# -------------------------
def main():
    logi(f"SRC={SRC} SHOW={SHOW} IMG_SIZE={IMG_SIZE} CONF={CONF_THRES} IOU={IOU_THRES} SKIP={SKIP}")
    logi(f"DET_ENGINE={DET_ENGINE} OCR_ENGINE={OCR_ENGINE}")

    cap = open_source()
    if not cap or not cap.isOpened():
        raise RuntimeError("Không mở được nguồn video. Kiểm tra CSI/RTSP/webcam.")

    detector = PlateDetector()
    ocr = PlateOCR()

    last_boxes = []
    frame_id = 0
    t0 = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logw("Không đọc được frame (stream end / camera lỗi).")
                break

            frame_id += 1

            do_detect = True
            if SKIP > 0:
                do_detect = (frame_id % (SKIP + 1) == 1)

            if do_detect:
                try:
                    dets = detector.detect(frame)
                    last_boxes = dets
                except Exception as e:
                    # If TRT blows up and CUDA context dead => usually need restart container
                    logw(f"Detect error: {e}")
                    last_boxes = []

            plates = 0
            # OCR only on top few boxes to keep FPS
            for i, (box, score) in enumerate(last_boxes[:3]):
                txt = ""
                try:
                    txt = ocr.recognize(frame, box)
                except Exception as e:
                    logw(f"OCR error: {e}")
                    txt = ""
                if txt:
                    plates += 1
                draw_plate(frame, box, txt)

            # FPS
            dt = time.time() - t0
            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = fps_smooth * 0.9 + fps * 0.1 if fps_smooth > 0 else fps
            t0 = time.time()

            cv2.putText(
                frame,
                f"FPS {fps_smooth:.1f} plates={plates}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if SHOW:
                cv2.imshow("ALPR", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord("q"):
                    break
            else:
                # headless mode: small sleep to avoid 100% CPU
                time.sleep(0.001)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
