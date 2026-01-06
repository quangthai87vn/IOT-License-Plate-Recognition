import os
import sys
import time
import cv2
import numpy as np
import subprocess

# =========================
# Config (env override)
# =========================
SHOW = os.getenv("SHOW", "1") == "1"
SRC = os.getenv("SRC", "csi")  # csi | rtsp | webcam
RTSP_URL = os.getenv("RTSP_URL", "")
WEBCAM_ID = int(os.getenv("WEBCAM_ID", "0"))

IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))     # detector input size
CONF_THRES = float(os.getenv("CONF", "0.25"))
IOU_THRES = float(os.getenv("IOU", "0.45"))
SKIP = int(os.getenv("SKIP", "0"))               # skip inference frames
OCR_EVERY = int(os.getenv("OCR_EVERY", "1"))     # OCR every N plates

# OCR input size (model fixed)
OCR_H = int(os.getenv("OCR_H", "40"))
OCR_W = int(os.getenv("OCR_W", "160"))

# model paths
DET_ONNX = os.getenv("DET_ONNX", "./model/LP_detector_nano_61.onnx")
OCR_ONNX = os.getenv("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")
DET_ENGINE = os.getenv("DET_ENGINE", "./model/LP_detector_nano_61_fp16.engine")
OCR_ENGINE = os.getenv("OCR_ENGINE", "./model/LP_ocr_nano_62_fp16.engine")

# CSI tuning
CSI_W = int(os.getenv("CSI_W", "1280"))
CSI_H = int(os.getenv("CSI_H", "720"))
CSI_FPS = int(os.getenv("CSI_FPS", "30"))
CSI_MODE = int(os.getenv("CSI_MODE", "3"))  # depends IMX219 mode list

# RTSP tuning
CODEC = os.getenv("CODEC", "h264")          # h264 | h265
RTSP_LATENCY = int(os.getenv("RTSP_LATENCY", "200"))

# draw tuning
FILL_ALPHA = float(os.getenv("FILL_ALPHA", "0.35"))  # red fill overlay


# =========================
# Helpers
# =========================
def file_exists(p: str) -> bool:
    return p and os.path.exists(p) and os.path.getsize(p) > 0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_imshow(win, img):
    try:
        cv2.imshow(win, img)
        return True
    except Exception:
        return False

def draw_plate(frame, box, text="", score=None):
    """Red fill, thin green border, red text."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = clamp(x1, 0, frame.shape[1] - 1)
    x2 = clamp(x2, 0, frame.shape[1] - 1)
    y1 = clamp(y1, 0, frame.shape[0] - 1)
    y2 = clamp(y2, 0, frame.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return

    # red fill overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness=-1)
    cv2.addWeighted(overlay, FILL_ALPHA, frame, 1.0 - FILL_ALPHA, 0, frame)

    # thin green border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    # red text
    if text is None:
        text = ""
    label = text.strip()
    if score is not None:
        label = f"{label}" if label else f"{score:.2f}"

    if label:
        ty = y1 - 10 if y1 - 10 > 20 else y2 + 25
        cv2.putText(
            frame, label, (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2, cv2.LINE_AA
        )


# =========================
# Video Reader
# =========================
class VideoReader:
    def __init__(self, src: str):
        self.src = src
        self.cap = None

    def _gst_csi(self):
        # appsink drop to reduce lag
        return (
            f"nvarguscamerasrc sensor-id=0 sensor-mode={CSI_MODE} ! "
            f"video/x-raw(memory:NVMM), width=(int){CSI_W}, height=(int){CSI_H}, "
            f"framerate=(fraction){CSI_FPS}/1 ! "
            f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def _gst_rtsp(self, url: str):
        # HW decode path (best on Jetson). If fails, fallback later.
        depay = "rtph264depay" if CODEC.lower() == "h264" else "rtph265depay"
        parse = "h264parse" if CODEC.lower() == "h264" else "h265parse"
        return (
            f"rtspsrc location={url} latency={RTSP_LATENCY} protocols=tcp drop-on-latency=true ! "
            f"{depay} ! {parse} ! "
            f"nvv4l2decoder ! nvvidconv ! "
            f"video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def open(self):
        if self.src == "csi":
            pipe = self._gst_csi()
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        elif self.src == "rtsp":
            if not RTSP_URL:
                raise ValueError("RTSP_URL is empty. Example: RTSP_URL=rtsp://ip:port/stream")
            pipe = self._gst_rtsp(RTSP_URL)
            self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)

            # fallback if gstreamer open failed
            if not self.cap.isOpened():
                print("[WARN] GStreamer RTSP open failed, fallback to cv2.VideoCapture(url) (CPU, may lag)")
                self.cap = cv2.VideoCapture(RTSP_URL)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
        else:
            self.cap = cv2.VideoCapture(WEBCAM_ID)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.src}")
        return self

    def read(self):
        return self.cap.read()

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


# =========================
# Backends
# =========================
class OpenCVDNNInfer:
    """OpenCV DNN fallback (CPU or CUDA if OpenCV built with CUDA)."""
    def __init__(self, onnx_path: str, prefer_cuda=True):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.cuda_ok = False

        if prefer_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                self.cuda_ok = True
            except Exception:
                self.cuda_ok = False

        if not self.cuda_ok:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, blob):
        self.net.setInput(blob)
        return self.net.forward()


class TRTInfer:
    """
    TensorRT engine inference with pycuda.
    IMPORTANT: must set correct input shape:
      - detector: (1,3,IMG_SIZE,IMG_SIZE)
      - ocr:      (1,3,OCR_H,OCR_W)
    """
    def __init__(self, engine_path: str, default_input_shape=None):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # create context

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Deserialize TensorRT engine failed: " + engine_path)

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Create TRT execution context failed")

        # bindings
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.host_mem = {}
        self.dev_mem = {}

        # pick first input binding name
        self.input_name = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_name = self.engine.get_binding_name(i)
                break

        # set default shape for dynamic engine
        if default_input_shape is not None and self.engine.binding_is_input(0):
            try:
                if any(d <= 0 for d in self.context.get_binding_shape(0)):
                    self.context.set_binding_shape(0, tuple(default_input_shape))
            except Exception:
                pass

        # allocate buffers
        self._allocate()

    def _allocate(self):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)
            if any(d <= 0 for d in shape):
                raise RuntimeError(f"Unresolved dynamic shape for binding {name}: {shape}")

            size = int(np.prod(shape))
            host = self.cuda.pagelocked_empty(size, dtype)
            dev = self.cuda.mem_alloc(host.nbytes)

            self.host_mem[name] = host
            self.dev_mem[name] = dev
            self.bindings.append(int(dev))

            if self.engine.binding_is_input(i):
                self.inputs.append(name)
            else:
                self.outputs.append(name)

        self.stream = self.cuda.Stream()

    def infer(self, feed_dict: dict):
        # assume single input
        inp_name = self.inputs[0]
        x = feed_dict[inp_name]

        x = np.ascontiguousarray(x)
        np.copyto(self.host_mem[inp_name], x.ravel())

        # H2D
        self.cuda.memcpy_htod_async(self.dev_mem[inp_name], self.host_mem[inp_name], self.stream)

        # exec
        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        # D2H
        outs = {}
        for out_name in self.outputs:
            self.cuda.memcpy_dtoh_async(self.host_mem[out_name], self.dev_mem[out_name], self.stream)

        self.stream.synchronize()

        # reshape outputs
        for out_name in self.outputs:
            idx = None
            for i in range(self.engine.num_bindings):
                if self.engine.get_binding_name(i) == out_name:
                    idx = i
                    break
            shape = tuple(self.context.get_binding_shape(idx))
            outs[out_name] = np.array(self.host_mem[out_name]).reshape(shape)

        return outs


# =========================
# Detector + OCR
# =========================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def nms(boxes, scores, iou_thres):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = []
        for j in idxs[1:]:
            ious.append(iou(boxes[i], boxes[j]))
        ious = np.array(ious)
        idxs = idxs[1:][ious < iou_thres]
    return keep

def iou(a, b):
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
    union = area_a + area_b - inter + 1e-6
    return inter / union

class PlateDetector:
    def __init__(self):
        self.kind = None
        self.backend = None

        # prefer TRT engine
        if file_exists(DET_ENGINE):
            try:
                self.backend = TRTInfer(DET_ENGINE, default_input_shape=(1, 3, IMG_SIZE, IMG_SIZE))
                self.kind = "trt"
                print("[INFO] DET backend = TensorRT engine:", DET_ENGINE)
                return
            except Exception as e:
                print("[WARN] TensorRT DET failed -> fallback. Err:", e)

        if file_exists(DET_ONNX):
            self.backend = OpenCVDNNInfer(DET_ONNX, prefer_cuda=True)
            self.kind = "cv"
            print("[INFO] DET backend = OpenCV DNN:", DET_ONNX)

    def _preprocess(self, frame_bgr):
        img, r, (padw, padh) = letterbox(frame_bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
        return x, r, padw, padh

    def _postprocess(self, pred, r, padw, padh, orig_shape):
        # pred expected: (1, N, 6) : x1,y1,x2,y2,score,cls
        p = np.squeeze(pred)
        if p.ndim == 1:
            p = p[None, :]
        if p.size == 0:
            return []

        # ensure 2D
        if p.shape[-1] < 6:
            return []

        boxes = p[:, 0:4].copy()
        scores = p[:, 4].copy()

        # threshold
        m = scores >= CONF_THRES
        boxes = boxes[m]
        scores = scores[m]
        if boxes.shape[0] == 0:
            return []

        # de-letterbox to original
        boxes[:, [0, 2]] -= padw
        boxes[:, [1, 3]] -= padh
        boxes /= r

        # clip
        h, w = orig_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

        # NMS
        keep = nms(boxes, scores, IOU_THRES)
        out = []
        for i in keep:
            out.append((boxes[i].tolist(), float(scores[i])))
        return out

    def detect(self, frame_bgr):
        x, r, padw, padh = self._preprocess(frame_bgr)

        if self.kind == "trt":
            outs = self.backend.infer({self.backend.input_name: x})
            pred = list(outs.values())[0]
        else:
            pred = self.backend.infer(x)

        return self._postprocess(pred, r, padw, padh, frame_bgr.shape)


class PlateOCR:
    # IMPORTANT: charset must match your OCR model training
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self):
        self.kind = None
        self.backend = None

        if file_exists(OCR_ENGINE):
            try:
                # FIX: OCR engine must use (1,3,40,160) not 640!
                self.backend = TRTInfer(OCR_ENGINE, default_input_shape=(1, 3, OCR_H, OCR_W))
                self.kind = "trt"
                print("[INFO] OCR backend = TensorRT engine:", OCR_ENGINE)
                return
            except Exception as e:
                print("[WARN] TensorRT OCR failed -> fallback. Err:", e)

        # fallback OpenCV DNN (CPU/CUDA)
        if file_exists(OCR_ONNX):
            self.backend = OpenCVDNNInfer(OCR_ONNX, prefer_cuda=True)
            self.kind = "cv"
            print("[INFO] OCR backend = OpenCV DNN:", OCR_ONNX)
        else:
            raise RuntimeError("OCR model not found")

    def _preprocess(self, plate_bgr):
        img = cv2.resize(plate_bgr, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    @staticmethod
    def ctc_decode(pred, chars):
        x = np.array(pred)
        if x.ndim == 3:
            x = x[0]

        # try to fix orientation (T,C) vs (C,T)
        # blank assumed 0, chars start from 1
        if x.shape[0] > x.shape[1] and x.shape[1] <= 80:
            x = x.T

        ids = np.argmax(x, axis=1).tolist()
        out = []
        prev = -1
        for i in ids:
            if i != prev and i != 0:
                j = i - 1
                if 0 <= j < len(chars):
                    out.append(chars[j])
            prev = i
        return "".join(out)

    def recognize(self, plate_bgr):
        x = self._preprocess(plate_bgr)

        if self.kind == "trt":
            outs = self.backend.infer({self.backend.input_name: x})
            pred = list(outs.values())[0]
        else:
            pred = self.backend.infer(x)

        return self.ctc_decode(pred, self.CHARS)


# =========================
# Main
# =========================
def main():
    print(f"[INFO] SRC={SRC} SHOW={SHOW} IMG_SIZE={IMG_SIZE} CONF={CONF_THRES} IOU={IOU_THRES} SKIP={SKIP} OCR_EVERY={OCR_EVERY}")
    if SRC == "rtsp":
        print(f"[INFO] RTSP_URL={RTSP_URL} CODEC={CODEC} LATENCY={RTSP_LATENCY}")
    if SRC == "csi":
        print(f"[INFO] CSI {CSI_W}x{CSI_H}@{CSI_FPS} mode={CSI_MODE}")

    vr = VideoReader(SRC).open()
    det = PlateDetector()
    ocr = PlateOCR()

    last_boxes = []
    frame_id = 0
    fps_t0 = time.time()
    fps_counter = 0
    fps = 0.0

    win_name = f"ALPR-{SRC.upper()}"

    try:
        while True:
            ok, frame = vr.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            frame_id += 1
            do_infer = (SKIP <= 0) or (frame_id % (SKIP + 1) == 1)

            ocr_results = []
            if do_infer:
                try:
                    boxes = det.detect(frame)
                    last_boxes = boxes
                except Exception as e:
                    # detector failed (TRT crash etc) -> keep last boxes
                    print("[WARN] DET failed, keep last boxes. Err:", e)
                    boxes = last_boxes

                for idx, (b, s) in enumerate(boxes):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    pad = int(0.03 * max(x2 - x1, y2 - y1))
                    x1p = clamp(x1 - pad, 0, frame.shape[1] - 1)
                    y1p = clamp(y1 - pad, 0, frame.shape[0] - 1)
                    x2p = clamp(x2 + pad, 0, frame.shape[1] - 1)
                    y2p = clamp(y2 + pad, 0, frame.shape[0] - 1)

                    plate = frame[y1p:y2p, x1p:x2p]
                    txt = ""

                    if plate.size != 0 and (idx % OCR_EVERY) == 0:
                        try:
                            txt = ocr.recognize(plate)
                        except Exception as e:
                            # OCR crash -> skip text, don't kill camera
                            print("[WARN] OCR failed:", e)
                            txt = ""

                    ocr_results.append((b, s, txt))
            else:
                ocr_results = [(b, s, "") for (b, s) in last_boxes]

            # FPS
            fps_counter += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps = fps_counter / dt
                fps_counter = 0
                fps_t0 = time.time()

            if SHOW:
                for b, s, txt in ocr_results:
                    draw_plate(frame, b, txt, score=s)

                cv2.putText(
                    frame,
                    f"FPS {fps:.1f} plates={len(ocr_results)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                safe_imshow(win_name, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord("q")):
                    break
            else:
                if do_infer:
                    print(f"FPS~{fps:.1f} plates={len(ocr_results)}")

    finally:
        vr.release()
        if SHOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
