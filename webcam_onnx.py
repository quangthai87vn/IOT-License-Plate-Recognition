#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
from collections import deque

import cv2
import numpy as np

# -----------------------------
# Utils
# -----------------------------
def now_ms():
    return int(time.time() * 1000)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms_xyxy(boxes, scores, iou_th=0.45):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        rest = rest[ious < iou_th]
        idxs = rest
    return keep

def letterbox(im, new_shape=640, color=(114,114,114)):
    # Resize + pad to square (YOLO style)
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    nh, nw = new_shape

    r = min(nw / w, nh / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = nw - new_unpad[0]
    dh = nh - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def draw_transparent_fill(img, x1, y1, x2, y2, color_bgr=(0,0,255), alpha=0.25):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_text_red(img, text, x, y):
    # Red text with black outline for readability
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

# -----------------------------
# Vietnamese plate heuristics (for choosing OCR decode)
# -----------------------------
PLATE_REGEXES = [
    re.compile(r"^\d{2}-[A-Z0-9]{1,2}\d{3}\.\d{2}$"),     # 63-B9658.14 (no space)
    re.compile(r"^\d{2}-[A-Z0-9]{1,2}\s?\d{3}\.\d{2}$"),  # 63-B9 658.14
    re.compile(r"^\d{2}[A-Z]-\d{3}\.\d{2}$"),            # 51G-502.21
    re.compile(r"^\d{2}[A-Z]\d-\d{3}\.\d{2}$"),          # 51G1-234.56 (rare)
]

def plate_score(s):
    if not s:
        return 0
    ss = s.replace(" ", "")
    score = 0
    for rgx in PLATE_REGEXES:
        if rgx.match(ss):
            score += 10
    # prefer plausible length
    if 7 <= len(ss) <= 12:
        score += 2
    # penalize weird chars
    if re.search(r"[^0-9A-Z\-\.\s]", ss):
        score -= 5
    return score

# -----------------------------
# TensorRT runner (safe)
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
            print("[TRT] import failed:", e)
            return

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            print("[TRT] Cannot load engine:", engine_path)
            return

        self.context = self.engine.create_execution_context()
        if self.context is None:
            print("[TRT] Cannot create context")
            return

        self.bindings = [None] * self.engine.num_bindings
        self.host_mem = {}
        self.dev_mem = {}
        self.inp_names = []
        self.out_names = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)
            if is_input:
                self.inp_names.append(name)
            else:
                self.out_names.append(name)
            self.host_mem[name] = None
            self.dev_mem[name] = None

        self.stream = cuda.Stream()
        self.ok = True

    def allocate_for_shape(self, input_name: str, input_shape: tuple):
        # set binding shape if dynamic
        idx = self.engine.get_binding_index(input_name)
        if -1 in tuple(self.engine.get_binding_shape(idx)):
            self.context.set_binding_shape(idx, input_shape)

        # allocate input
        import numpy as np
        inp_dtype = self.trt.nptype(self.engine.get_binding_dtype(idx))
        inp_size = int(np.prod(input_shape))
        self.host_mem[input_name] = self.cuda.pagelocked_empty(inp_size, inp_dtype)
        self.dev_mem[input_name] = self.cuda.mem_alloc(self.host_mem[input_name].nbytes)
        self.bindings[idx] = int(self.dev_mem[input_name])

        # allocate outputs based on context shapes
        for out_name in self.out_names:
            out_idx = self.engine.get_binding_index(out_name)
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            out_dtype = self.trt.nptype(self.engine.get_binding_dtype(out_idx))
            out_size = int(np.prod(out_shape))
            self.host_mem[out_name] = self.cuda.pagelocked_empty(out_size, out_dtype)
            self.dev_mem[out_name] = self.cuda.mem_alloc(self.host_mem[out_name].nbytes)
            self.bindings[out_idx] = int(self.dev_mem[out_name])

    def infer(self, input_name: str, x: np.ndarray):
        # x must be contiguous float32/float16 etc with shape = binding
        if not self.ok:
            raise RuntimeError("TRT not ready")

        if self.host_mem[input_name] is None:
            self.allocate_for_shape(input_name, x.shape)

        # if dynamic and shape changes => re-alloc
        idx = self.engine.get_binding_index(input_name)
        cur_shape = tuple(self.context.get_binding_shape(idx))
        if cur_shape != tuple(x.shape):
            self.allocate_for_shape(input_name, x.shape)

        np.copyto(self.host_mem[input_name], x.ravel())

        # H2D
        self.cuda.memcpy_htod_async(self.dev_mem[input_name], self.host_mem[input_name], self.stream)
        # execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=int(self.stream.handle))
        # D2H
        outputs = {}
        for out_name in self.out_names:
            self.cuda.memcpy_dtoh_async(self.host_mem[out_name], self.dev_mem[out_name], self.stream)
        self.stream.synchronize()

        for out_name in self.out_names:
            out_idx = self.engine.get_binding_index(out_name)
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            outputs[out_name] = np.array(self.host_mem[out_name]).reshape(out_shape)

        return outputs

# -----------------------------
# Model wrappers
# -----------------------------
def load_detector(det_engine, det_onnx, prefer_trt=True):
    trt_runner = None
    cv_net = None
    backend = "cpu"

    if prefer_trt and det_engine and os.path.exists(det_engine):
        trt_runner = TRTRunner(det_engine)
        if trt_runner.ok:
            backend = "trt"
            print("[DET] Using TensorRT engine:", det_engine)
            return ("trt", trt_runner, None)

    if det_onnx and os.path.exists(det_onnx):
        cv_net = cv2.dnn.readNetFromONNX(det_onnx)
        # DO NOT force CUDA in your case (cuDNN mismatch + crash)
        cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = "cpu"
        print("[DET] Using OpenCV DNN CPU ONNX:", det_onnx)
        return ("cv", None, cv_net)

    raise FileNotFoundError("Detector model not found (engine/onnx).")

def load_ocr(ocr_engine, ocr_onnx, prefer_trt=True):
    trt_runner = None
    cv_net = None
    backend = "cpu"

    if prefer_trt and ocr_engine and os.path.exists(ocr_engine):
        trt_runner = TRTRunner(ocr_engine)
        if trt_runner.ok:
            backend = "trt"
            print("[OCR] Using TensorRT engine:", ocr_engine)
            return ("trt", trt_runner, None)

    if ocr_onnx and os.path.exists(ocr_onnx):
        cv_net = cv2.dnn.readNetFromONNX(ocr_onnx)
        cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = "cpu"
        print("[OCR] Using OpenCV DNN CPU ONNX:", ocr_onnx)
        return ("cv", None, cv_net)

    raise FileNotFoundError("OCR model not found (engine/onnx).")

def yolo_postprocess(raw, conf_th=0.25, iou_th=0.45):
    """
    Accept common YOLO outputs:
    - (1, N, 6) or (1, N, 5+nc)  (YOLOv5)
    - (1, 84, 8400) or (1, 85, 8400) (YOLOv8-ish) -> transpose
    Return boxes xyxy in input-space (640x640), scores.
    """
    out = raw
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    # YOLOv8 style
    if out.ndim == 2 and out.shape[0] in (84, 85) and out.shape[1] > 1000:
        out = out.T  # (8400, 84)

    if out.ndim != 2 or out.shape[1] < 6:
        return [], []

    boxes = []
    scores = []

    # YOLOv5 format: [cx, cy, w, h, obj, cls...]
    for row in out:
        cx, cy, w, h = row[0:4]
        obj = row[4]
        if out.shape[1] > 6:
            cls_scores = row[5:]
            cls = int(np.argmax(cls_scores))
            conf = float(obj * cls_scores[cls])
        else:
            # one-class
            conf = float(obj * row[5])

        if conf < conf_th:
            continue

        x1 = float(cx - w/2)
        y1 = float(cy - h/2)
        x2 = float(cx + w/2)
        y2 = float(cy + h/2)

        boxes.append([x1,y1,x2,y2])
        scores.append(conf)

    if not boxes:
        return [], []

    keep = nms_xyxy(boxes, scores, iou_th=iou_th)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    return boxes, scores

def ocr_preprocess(plate_bgr, ocr_w=160, ocr_h=40):
    # keep BGR -> RGB
    img = cv2.resize(plate_bgr, (ocr_w, ocr_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))  # CHW
    img = np.expand_dims(img, 0)      # NCHW
    return np.ascontiguousarray(img)

def ctc_greedy_decode(logits, charset, blank_index):
    # logits: (T,C) or (C,T)
    if logits.ndim != 2:
        return ""
    if logits.shape[0] < logits.shape[1]:
        # guess (T,C)
        T, C = logits.shape
        probs = logits
    else:
        # maybe (C,T)
        probs = logits.T
        T, C = probs.shape

    ids = np.argmax(probs, axis=1).tolist()
    # collapse repeats + remove blank
    res = []
    prev = None
    for i in ids:
        if i == prev:
            continue
        prev = i
        if i == blank_index:
            continue
        if 0 <= i < len(charset):
            res.append(charset[i])
    return "".join(res)

def ocr_decode_best(logits):
    # Vietnamese plate charset (no I/O/Q to reduce confusion)
    charset = list("0123456789ABCDEFGHJKLMNPRSTUVXYZ-.")
    # try blank=0 and blank=last
    cand1 = ctc_greedy_decode(logits, charset, blank_index=0)
    cand2 = ctc_greedy_decode(logits, charset, blank_index=len(charset)-1)

    # normalize
    def norm(s):
        return s.replace(" ", "").replace("..", ".").strip()

    c1 = norm(cand1)
    c2 = norm(cand2)

    s1 = plate_score(c1)
    s2 = plate_score(c2)
    if s2 > s1:
        return c2
    if s1 > s2:
        return c1
    # tie -> longer wins
    return c1 if len(c1) >= len(c2) else c2

# -----------------------------
# Video sources
# -----------------------------
def gst_csi(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

def gst_rtsp(url, latency=200):
    # hardware decode
    return (
        f"rtspsrc location={url} latency={latency} ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

def open_capture(args):
    if args.src == "csi":
        pipe = gst_csi(args.sensor_id, args.cam_w, args.cam_h, args.cam_fps, args.flip)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap
    if args.src == "rtsp":
        if not args.rtsp_url:
            raise RuntimeError("Thiếu --rtsp-url")
        pipe = gst_rtsp(args.rtsp_url, args.rtsp_latency)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap
    # webcam / usb cam
    cap = cv2.VideoCapture(args.cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    return cap

# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["csi","rtsp","webcam"], default=os.getenv("SRC", "csi"))
    ap.add_argument("--rtsp-url", default=os.getenv("RTSP_URL",""))
    ap.add_argument("--rtsp-latency", type=int, default=int(os.getenv("RTSP_LATENCY","200")))

    ap.add_argument("--sensor-id", type=int, default=int(os.getenv("SENSOR_ID","0")))
    ap.add_argument("--cam-index", type=int, default=int(os.getenv("CAM_INDEX","0")))
    ap.add_argument("--cam-w", type=int, default=int(os.getenv("CAM_W","1280")))
    ap.add_argument("--cam-h", type=int, default=int(os.getenv("CAM_H","720")))
    ap.add_argument("--cam-fps", type=int, default=int(os.getenv("CAM_FPS","30")))
    ap.add_argument("--flip", type=int, default=int(os.getenv("FLIP","0")))

    ap.add_argument("--img-size", type=int, default=int(os.getenv("IMG_SIZE","640")))
    ap.add_argument("--conf", type=float, default=float(os.getenv("CONF","0.25")))
    ap.add_argument("--iou", type=float, default=float(os.getenv("IOU","0.45")))
    ap.add_argument("--show", type=int, default=int(os.getenv("SHOW","1")))
    ap.add_argument("--skip-ocr", type=int, default=int(os.getenv("SKIP_OCR","2")))  # run OCR each N frames/plate

    ap.add_argument("--det-onnx", default=os.getenv("DET_ONNX","./model/LP_detector_nano_61.onnx"))
    ap.add_argument("--ocr-onnx", default=os.getenv("OCR_ONNX","./model/LP_ocr_nano_62.onnx"))
    ap.add_argument("--det-engine", default=os.getenv("DET_ENGINE","./model/LP_detector_nano_61_fp16.engine"))
    ap.add_argument("--ocr-engine", default=os.getenv("OCR_ENGINE","./model/LP_ocr_nano_62_fp16.engine"))
    ap.add_argument("--prefer-trt", type=int, default=int(os.getenv("PREFER_TRT","1")))

    args = ap.parse_args()

    print(f"[INFO] SRC={args.src} SHOW={bool(args.show)} IMG_SIZE={args.img_size} CONF={args.conf} IOU={args.iou} SKIP_OCR={args.skip_ocr}")
    print(f"[INFO] DET_ENGINE={args.det_engine}")
    print(f"[INFO] OCR_ENGINE={args.ocr_engine}")

    det_mode, det_trt, det_net = load_detector(args.det_engine, args.det_onnx, prefer_trt=bool(args.prefer_trt))
    ocr_mode, ocr_trt, ocr_net = load_ocr(args.ocr_engine, args.ocr_onnx, prefer_trt=bool(args.prefer_trt))

    cap = open_capture(args)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video source (CSI/RTSP/WEBCAM).")

    win = "ALPR"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # simple IOU tracking for smoothing OCR text
    tracks = []  # list of dict {bbox, texts(deque), last_ms, last_ocr_ms}
    track_ttl_ms = 1200

    fps_t0 = time.time()
    fps_cnt = 0
    fps_val = 0.0

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] cap.read() failed -> exit")
            break

        frame_id += 1
        H, W = frame.shape[:2]

        # DET preprocess
        det_in, r, (padx, pady) = letterbox(frame, args.img_size)
        det_rgb = cv2.cvtColor(det_in, cv2.COLOR_BGR2RGB)
        det_x = det_rgb.astype(np.float32) / 255.0
        det_x = np.transpose(det_x, (2,0,1))
        det_x = np.expand_dims(det_x, 0)
        det_x = np.ascontiguousarray(det_x)

        # DET infer
        if det_mode == "trt":
            outs = det_trt.infer(det_trt.inp_names[0], det_x)
            # pick first output
            raw = list(outs.values())[0]
        else:
            blob = cv2.dnn.blobFromImage(det_in, 1/255.0, (args.img_size,args.img_size), swapRB=True, crop=False)
            det_net.setInput(blob)
            raw = det_net.forward()

        boxes_640, scores = yolo_postprocess(raw, conf_th=args.conf, iou_th=args.iou)

        # map boxes back to original frame
        dets = []
        for (x1,y1,x2,y2), sc in zip(boxes_640, scores):
            x1 = (x1 - padx) / (r + 1e-9)
            y1 = (y1 - pady) / (r + 1e-9)
            x2 = (x2 - padx) / (r + 1e-9)
            y2 = (y2 - pady) / (r + 1e-9)

            x1 = int(clamp(round(x1), 0, W-1))
            y1 = int(clamp(round(y1), 0, H-1))
            x2 = int(clamp(round(x2), 0, W-1))
            y2 = int(clamp(round(y2), 0, H-1))
            if (x2-x1) < 30 or (y2-y1) < 20:
                continue
            dets.append((x1,y1,x2,y2,float(sc)))

        # cleanup old tracks
        tnow = now_ms()
        tracks = [t for t in tracks if (tnow - t["last_ms"]) <= track_ttl_ms]

        # match detections to tracks by IOU
        used = set()
        for (x1,y1,x2,y2,sc) in dets:
            best_iou, best_j = 0.0, -1
            for j,t in enumerate(tracks):
                if j in used: 
                    continue
                i = iou_xyxy([x1,y1,x2,y2], t["bbox"])
                if i > best_iou:
                    best_iou, best_j = i, j

            if best_iou > 0.3 and best_j >= 0:
                tr = tracks[best_j]
                tr["bbox"] = [x1,y1,x2,y2]
                tr["last_ms"] = tnow
                used.add(best_j)
            else:
                tracks.append({
                    "bbox":[x1,y1,x2,y2],
                    "texts":deque(maxlen=7),
                    "last_ms":tnow,
                    "last_ocr_ms":0
                })

        # OCR for tracks (skip to save FPS)
        for tr in tracks:
            x1,y1,x2,y2 = tr["bbox"]

            do_ocr = (args.skip_ocr <= 1) or (frame_id % args.skip_ocr == 0)
            if not do_ocr:
                continue

            # crop with margin
            mx = int(0.08 * (x2-x1))
            my = int(0.10 * (y2-y1))
            xx1 = clamp(x1 - mx, 0, W-1)
            yy1 = clamp(y1 - my, 0, H-1)
            xx2 = clamp(x2 + mx, 0, W-1)
            yy2 = clamp(y2 + my, 0, H-1)

            crop = frame[yy1:yy2, xx1:xx2]
            if crop.size == 0:
                continue

            x = ocr_preprocess(crop, ocr_w=160, ocr_h=40)

            try:
                if ocr_mode == "trt":
                    o = ocr_trt.infer(ocr_trt.inp_names[0], x)
                    logits = list(o.values())[0]
                else:
                    blob = cv2.dnn.blobFromImage(cv2.resize(crop, (160,40)), 1/255.0, (160,40), swapRB=True, crop=False)
                    ocr_net.setInput(blob)
                    logits = ocr_net.forward()
            except Exception as e:
                # avoid killing camera by exception storm
                print("[OCR] error:", e)
                continue

            # squeeze to 2D
            lg = logits
            while lg.ndim > 2:
                lg = np.squeeze(lg, axis=0)
            text = ocr_decode_best(lg)
            if text:
                tr["texts"].append(text)
                tr["last_ocr_ms"] = tnow

        # choose stable text for each track
        def stable_text(q: deque):
            if not q:
                return ""
            # majority vote
            freq = {}
            for s in q:
                key = s.replace(" ", "")
                freq[key] = freq.get(key, 0) + 1
            best = max(freq.items(), key=lambda x: x[1])[0]
            # restore with formatting (optional)
            return best

        # overlay
        if args.show:
            # HUD
            fps_cnt += 1
            if (time.time() - fps_t0) >= 1.0:
                fps_val = fps_cnt / (time.time() - fps_t0)
                fps_cnt = 0
                fps_t0 = time.time()

            cv2.putText(frame, f"FPS {fps_val:.1f} plates={len(tracks)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            for tr in tracks:
                x1,y1,x2,y2 = tr["bbox"]
                # fill red
                draw_transparent_fill(frame, x1,y1,x2,y2, color_bgr=(0,0,255), alpha=0.22)
                # thin green border
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

                txt = stable_text(tr["texts"])
                if txt:
                    put_text_red(frame, txt, x1, max(35, y1-8))

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
