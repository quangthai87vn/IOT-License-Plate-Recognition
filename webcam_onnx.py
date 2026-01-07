#!/usr/bin/env python3
import os
import time
import argparse
import re
import numpy as np
import cv2

# ==========================
# Utils
# ==========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad to meet stride-multiple (YOLO style). Return: img, ratio, (dw, dh)."""
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

def nms_boxes_xyxy(boxes, scores, score_th=0.25, nms_th=0.45):
    """boxes: list of [x1,y1,x2,y2] in pixels."""
    if len(boxes) == 0:
        return []
    b = []
    for (x1,y1,x2,y2) in boxes:
        b.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
    idxs = cv2.dnn.NMSBoxes(b, scores, score_th, nms_th)
    if idxs is None or len(idxs) == 0:
        return []
    # idxs can be [[0],[1]] or [0,1]
    idxs = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]
    return idxs

def draw_plate_overlay(frame, x1, y1, x2, y2, alpha=0.25):
    """Tô đỏ vùng biển số nhẹ + viền xanh mỏng."""
    x1 = clamp(x1, 0, frame.shape[1]-1)
    x2 = clamp(x2, 0, frame.shape[1]-1)
    y1 = clamp(y1, 0, frame.shape[0]-1)
    y2 = clamp(y2, 0, frame.shape[0]-1)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # đỏ fill
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)     # xanh viền

def put_text_red(frame, text, x, y):
    if not text:
        return
    y = clamp(y, 20, frame.shape[0]-10)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

def normalize_plate_text(s):
    s = (s or "").strip().upper().replace(" ", "")
    s = re.sub(r"[^0-9A-Z\-\.\:]", "", s)
    return s

def format_vn_plate_guess(raw):
    """
    Heuristic format:
    - Prefer pattern like: 63-B9 658.14
    - If string is continuous like 63B965814 -> insert '-' and '.'
    """
    s = normalize_plate_text(raw)
    if not s:
        return ""

    # already has separators
    if "-" in s or "." in s:
        return s

    # common lengths: 8-10 chars excluding separators
    # Example: 63B965814  (9)
    if len(s) >= 8:
        # try: 2 digits + 1-2 letters+digits for series, remaining digits
        # simplest: 2 + 2 + rest
        a = s[:2]
        b = s[2:4]
        c = s[4:]
        # put dot before last 2 digits if long
        if len(c) >= 5:
            c = c[:-2] + "." + c[-2:]
        return f"{a}-{b}{c}"
    return s

# ==========================
# Backend: TensorRT (preferred)
# ==========================
class TRTInfer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.ok = False
        self.trt = None
        self.cuda = None
        self.engine = None
        self.context = None
        self.bindings = None
        self.h_inputs = {}
        self.d_inputs = {}
        self.h_outputs = {}
        self.d_outputs = {}
        self.output_names = []
        self.input_name = None
        self.stream = None
        self.profile_index = 0

        self._load()

    def _load(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except Exception as e:
            print(f"[WARN] TensorRT/PyCUDA not available: {e}")
            return

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            print(f"[ERR] Failed to load engine: {self.engine_path}")
            return

        self.context = self.engine.create_execution_context()
        if self.context is None:
            print(f"[ERR] Failed to create context: {self.engine_path}")
            return

        self.stream = cuda.Stream()

        # bindings
        self.bindings = [None] * self.engine.num_bindings

        # find input
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_name = name
                break

        # set input shape (if dynamic) to OPT shape from profile 0
        in_idx = self.engine.get_binding_index(self.input_name)
        shape = self.engine.get_binding_shape(in_idx)

        # explicit batch + dynamic dims => set from profile
        if -1 in shape:
            try:
                mn, opt, mx = self.engine.get_profile_shape(self.profile_index, in_idx)
                self.context.set_binding_shape(in_idx, opt)
                shape = opt
            except Exception as e:
                print(f"[ERR] Cannot set dynamic shape from profile: {e}")
                return

        # allocate input buffers
        in_shape = tuple(self.context.get_binding_shape(in_idx))
        in_size = int(np.prod(in_shape))
        self.h_inputs[self.input_name] = cuda.pagelocked_empty(in_size, np.float32)
        self.d_inputs[self.input_name] = cuda.mem_alloc(self.h_inputs[self.input_name].nbytes)
        self.bindings[in_idx] = int(self.d_inputs[self.input_name])

        # allocate outputs buffers
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                continue
            name = self.engine.get_binding_name(i)
            out_shape = tuple(self.context.get_binding_shape(i))
            out_size = int(np.prod(out_shape))
            dtype = np.float32
            self.h_outputs[name] = cuda.pagelocked_empty(out_size, dtype)
            self.d_outputs[name] = cuda.mem_alloc(self.h_outputs[name].nbytes)
            self.bindings[i] = int(self.d_outputs[name])
            self.output_names.append(name)

        self.ok = True
        print(f"[INFO] TensorRT engine loaded: {self.engine_path}")
        print(f"[INFO]  input={self.input_name} shape={in_shape}, outputs={self.output_names}")

    def input_hw(self):
        """Return (H,W) expected by engine input."""
        if not self.ok:
            return None
        in_idx = self.engine.get_binding_index(self.input_name)
        s = tuple(self.context.get_binding_shape(in_idx))
        # expect NCHW
        return (int(s[2]), int(s[3]))

    def infer(self, x_nchw_float32):
        """x: np.float32 NCHW contiguous"""
        if not self.ok:
            raise RuntimeError("TRT engine not ready")

        in_idx = self.engine.get_binding_index(self.input_name)
        expected = tuple(self.context.get_binding_shape(in_idx))
        if tuple(x_nchw_float32.shape) != expected:
            # allow dynamic: set binding shape then re-alloc output (rare)
            try:
                self.context.set_binding_shape(in_idx, x_nchw_float32.shape)
            except Exception as e:
                raise RuntimeError(f"Binding shape mismatch. expected={expected}, got={x_nchw_float32.shape}, err={e}")

        # copy in
        np.copyto(self.h_inputs[self.input_name], x_nchw_float32.ravel())
        self.cuda.memcpy_htod_async(self.d_inputs[self.input_name], self.h_inputs[self.input_name], self.stream)

        # execute
        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        # copy out
        outs = []
        for name in self.output_names:
            self.cuda.memcpy_dtoh_async(self.h_outputs[name], self.d_outputs[name], self.stream)
        self.stream.synchronize()

        for name in self.output_names:
            # reshape using current binding shape
            out_idx = self.engine.get_binding_index(name)
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            outs.append(self.h_outputs[name].reshape(out_shape).copy())

        return outs

# ==========================
# Backend: OpenCV DNN fallback
# ==========================
class OpenCVDNN:
    def __init__(self, onnx_path, use_cuda=False):
        self.onnx_path = onnx_path
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.use_cuda = use_cuda

        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                print(f"[INFO] OpenCV DNN CUDA FP16: {onnx_path}")
            except Exception as e:
                print(f"[WARN] OpenCV DNN CUDA not available, fallback CPU: {e}")
                self.use_cuda = False

        if not self.use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[INFO] OpenCV DNN CPU: {onnx_path}")

    def infer(self, blob_nchw):
        self.net.setInput(blob_nchw)
        out = self.net.forward()
        return [out]

# ==========================
# YOLO Postprocess (generic)
# ==========================
def yolo_decode_single_output(out, conf_th, num_classes=None):
    """
    out can be:
      - (1, N, 5+nc) or (N, 5+nc)
    returns list: (cx,cy,w,h,score,cls)
    """
    if out is None:
        return []
    out = np.asarray(out)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6:
        return []

    obj = out[:, 4]
    cls_scores = out[:, 5:]
    if cls_scores.size == 0:
        # 1-class case might be (cx,cy,w,h,conf,cls_id) but uncommon
        score = obj
        cls = np.zeros_like(score, dtype=np.int32)
    else:
        cls = np.argmax(cls_scores, axis=1).astype(np.int32)
        score = obj * cls_scores[np.arange(len(cls_scores)), cls]

    keep = score >= conf_th
    out = out[keep]
    score = score[keep]
    cls = cls[keep]
    if len(out) == 0:
        return []

    dets = []
    for i in range(len(out)):
        cx, cy, w, h = out[i, 0], out[i, 1], out[i, 2], out[i, 3]
        dets.append((float(cx), float(cy), float(w), float(h), float(score[i]), int(cls[i])))
    return dets

def scale_boxes_back(dets_cxcywh, r, pad_left, pad_top, orig_w, orig_h):
    """Convert from model-space (after letterbox) back to original image xyxy."""
    boxes = []
    scores = []
    clses = []
    for (cx,cy,w,h,sc,cl) in dets_cxcywh:
        x1 = (cx - w/2 - pad_left) / r
        y1 = (cy - h/2 - pad_top) / r
        x2 = (cx + w/2 - pad_left) / r
        y2 = (cy + h/2 - pad_top) / r
        x1 = clamp(int(round(x1)), 0, orig_w-1)
        y1 = clamp(int(round(y1)), 0, orig_h-1)
        x2 = clamp(int(round(x2)), 0, orig_w-1)
        y2 = clamp(int(round(y2)), 0, orig_h-1)
        boxes.append([x1,y1,x2,y2])
        scores.append(float(sc))
        clses.append(int(cl))
    return boxes, scores, clses

# ==========================
# OCR decode (2 modes)
#  - Mode A: YOLO char boxes
#  - Mode B: CTC sequence
# auto-detect by output shape
# ==========================
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def ocr_decode_auto(outputs, conf_th=0.25):
    """
    outputs: list of np arrays from OCR model
    Return best guess string
    """
    if not outputs:
        return ""

    out = outputs[0]
    out = np.asarray(out)

    # Heuristic: YOLO-like output -> last dim >= 6 and middle dim is large
    if (out.ndim in (2,3)) and ((out.ndim == 3 and out.shape[-1] >= 6) or (out.ndim == 2 and out.shape[1] >= 6)):
        dets = yolo_decode_single_output(out, conf_th)
        # char boxes => sort into 2 lines
        # each det: (cx,cy,w,h,score,cls)
        chars = []
        for cx,cy,w,h,sc,cl in dets:
            ch = ALPHABET[cl] if 0 <= cl < len(ALPHABET) else ""
            chars.append((cx, cy, ch, sc))
        if not chars:
            return ""

        # group by y (2 lines)
        ys = np.array([c[1] for c in chars], dtype=np.float32)
        y_med = float(np.median(ys))
        top = [c for c in chars if c[1] <= y_med]
        bot = [c for c in chars if c[1] > y_med]

        top.sort(key=lambda x: x[0])
        bot.sort(key=lambda x: x[0])

        s_top = "".join([c[2] for c in top if c[2]])
        s_bot = "".join([c[2] for c in bot if c[2]])

        if s_top and s_bot:
            return f"{s_top}-{s_bot}"
        return s_top or s_bot

    # Heuristic: CTC-like (T,C)
    # handle (1,T,C) or (T,C)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim == 2:
        # decode by argmax per timestep
        idx = np.argmax(out, axis=1).tolist()
        # try blank=0 and blank=last, choose better by VN plate regex score
        def decode(blank_index):
            s = []
            prev = None
            for i in idx:
                if i == blank_index:
                    prev = i
                    continue
                if prev == i:
                    continue
                ch_i = i
                # map index -> char (skip blank)
                if blank_index == 0:
                    ch_i = i - 1
                if 0 <= ch_i < len(ALPHABET):
                    s.append(ALPHABET[ch_i])
                prev = i
            return "".join(s)

        cand1 = decode(0)
        cand2 = decode(out.shape[1]-1)

        def score_plate(x):
            x = normalize_plate_text(x)
            score = 0
            # strong patterns
            if re.search(r"^\d{2}[A-Z]\d", x): score += 2
            if re.search(r"\d{4,6}$", x): score += 1
            if 7 <= len(x) <= 10: score += 1
            return score

        best = cand1 if score_plate(cand1) >= score_plate(cand2) else cand2
        return best

    return ""

# ==========================
# Video sources
# ==========================
def open_capture(source, rtsp_url=None, cam_index=0, csi_w=1280, csi_h=720, csi_fps=30):
    if source == "csi":
        gst = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width={csi_w}, height={csi_h}, framerate={csi_fps}/1 ! "
            f"nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        return cap

    if source == "rtsp":
        if not rtsp_url:
            raise ValueError("RTSP URL missing")
        # try gstreamer first
        gst = (
            f"rtspsrc location={rtsp_url} latency=150 ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        # fallback ffmpeg
        cap = cv2.VideoCapture(rtsp_url)
        return cap

    # webcam
    cap = cv2.VideoCapture(cam_index)
    return cap

# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.environ.get("ALPR_SOURCE", "csi"), choices=["csi","rtsp","webcam"])
    ap.add_argument("--rtsp", default=os.environ.get("RTSP_URL", ""))
    ap.add_argument("--cam", type=int, default=0)

    ap.add_argument("--det_engine", default="./model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", default="./model/LP_ocr_nano_62_fp16.engine")

    ap.add_argument("--det_onnx", default="./model/LP_detector_nano_61.onnx")
    ap.add_argument("--ocr_onnx", default="./model/LP_ocr_nano_62.onnx")

    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--ocr_conf", type=float, default=0.25)

    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)

    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()

    # --------------------------
    # Load DET + OCR backends
    # --------------------------
    det = None
    ocr = None

    if os.path.exists(args.det_engine):
        det = TRTInfer(args.det_engine)
    if os.path.exists(args.ocr_engine):
        ocr = TRTInfer(args.ocr_engine)

    # fallback OpenCV DNN if TRT not ready
    if det is None or not det.ok:
        if os.path.exists(args.det_onnx):
            det = OpenCVDNN(args.det_onnx, use_cuda=False)
        else:
            raise FileNotFoundError("Detector model not found (engine/onnx).")

    if ocr is None or not ocr.ok:
        if os.path.exists(args.ocr_onnx):
            ocr = OpenCVDNN(args.ocr_onnx, use_cuda=False)
        else:
            print("[WARN] OCR model not found. Will only detect plates.")
            ocr = None

    # --------------------------
    # Open video
    # --------------------------
    cap = open_capture(args.source, args.rtsp, args.cam, args.csi_w, args.csi_h, args.csi_fps)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source. Check docker args / nvargus / rtsp url.")

    # detector input size
    det_hw = None
    if isinstance(det, TRTInfer) and det.ok:
        det_hw = det.input_hw()
    if det_hw is None:
        det_hw = (640, 640)  # default YOLO
    det_h, det_w = det_hw

    # ocr input size
    ocr_hw = None
    if ocr and isinstance(ocr, TRTInfer) and ocr.ok:
        ocr_hw = ocr.input_hw()

    fps_t0 = time.time()
    fps_cnt = 0
    fps_val = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Frame not received. Stream ended or camera busy.")
            break

        H, W = frame.shape[:2]

        # --------------------------
        # DET preprocess
        # --------------------------
        img_lb, r, (pad_left, pad_top) = letterbox(frame, (det_h, det_w))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW

        # --------------------------
        # DET infer
        # --------------------------
        try:
            if isinstance(det, TRTInfer):
                det_outs = det.infer(x)
            else:
                blob = x  # already NCHW float
                det_outs = det.infer(blob)
        except Exception as e:
            print(f"[ERR] DET inference failed: {e}")
            break

        out0 = det_outs[0] if det_outs else None
        dets = yolo_decode_single_output(out0, args.conf)
        boxes, scores, clses = scale_boxes_back(dets, r, pad_left, pad_top, W, H)

        keep = nms_boxes_xyxy(boxes, scores, args.conf, args.nms)
        plates = []
        for i in keep:
            x1,y1,x2,y2 = boxes[i]
            plates.append((x1,y1,x2,y2,float(scores[i])))

        # --------------------------
        # OCR per plate
        # --------------------------
        for (x1,y1,x2,y2,sc) in plates:
            draw_plate_overlay(frame, x1,y1,x2,y2, alpha=0.22)

            text = ""
            if ocr is not None:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    # determine OCR input
                    if ocr_hw is None:
                        # fallback guess (common): H=40, W=160
                        oh, ow = 40, 160
                    else:
                        oh, ow = ocr_hw

                    crop_rs = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    crop_rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB)
                    xo = crop_rgb.astype(np.float32) / 255.0
                    xo = np.transpose(xo, (2,0,1))[None, ...]  # NCHW

                    try:
                        if isinstance(ocr, TRTInfer):
                            ocr_outs = ocr.infer(xo)
                        else:
                            ocr_outs = ocr.infer(xo)
                        raw = ocr_decode_auto(ocr_outs, conf_th=args.ocr_conf)
                        text = format_vn_plate_guess(raw)
                    except Exception as e:
                        # OCR fail shouldn't kill whole app
                        text = ""
                        # print once per frame would spam; keep minimal
                        # print(f"[WARN] OCR failed: {e}")

            if text:
                put_text_red(frame, text, x1, y1 - 8)

        # --------------------------
        # FPS
        # --------------------------
        fps_cnt += 1
        if fps_cnt >= 10:
            dt = time.time() - fps_t0
            fps_val = fps_cnt / dt if dt > 0 else 0.0
            fps_cnt = 0
            fps_t0 = time.time()

        cv2.putText(frame, f"FPS {fps_val:.1f} plates={len(plates)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        if args.show:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
