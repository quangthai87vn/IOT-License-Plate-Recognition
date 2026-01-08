#!/usr/bin/env python3
import os
import time
import argparse
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 (auto init CUDA context)


# ----------------------------
# Utils
# ----------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    YOLOv5-style letterbox.
    Returns: resized image, ratio, (dw, dh)
    """
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # w,h padding

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
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms_boxes(boxes, scores, iou_thres=0.45):
    """
    Pure numpy NMS.
    boxes: (N,4) xyxy
    scores: (N,)
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
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


def yolo_postprocess(pred, conf_thres=0.25, iou_thres=0.45):
    """
    pred: (1, N, 5+nc) or (N, 5+nc)
    Returns list of dets: [x1,y1,x2,y2,conf,cls]
    """
    if pred.ndim == 3:
        pred = pred[0]

    # obj conf
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]
    cls_id = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
    conf = obj * cls_conf

    mask = conf >= conf_thres
    pred = pred[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if pred.shape[0] == 0:
        return []

    boxes_xywh = pred[:, :4]
    boxes = xywh2xyxy(boxes_xywh)

    dets = []
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        b = boxes[idx]
        s = conf[idx]
        keep = nms_boxes(b, s, iou_thres=iou_thres)
        for k in keep:
            dets.append([float(b[k, 0]), float(b[k, 1]), float(b[k, 2]), float(b[k, 3]),
                         float(s[k]), int(c)])
    dets.sort(key=lambda x: x[4], reverse=True)
    return dets


def scale_coords(box, ratio, dwdh):
    """
    Reverse letterbox scaling: from 640x640 back to original image coords.
    """
    x1, y1, x2, y2 = box
    dw, dh = dwdh
    x1 = (x1 - dw) / ratio
    x2 = (x2 - dw) / ratio
    y1 = (y1 - dh) / ratio
    y2 = (y2 - dh) / ratio
    return [x1, y1, x2, y2]


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    return [x1, y1, x2, y2]


def load_labels_from_pt(pt_path):
    """
    Try extract YOLOv5 class names from .pt (no inference).
    """
    try:
        import torch
        ckpt = torch.load(pt_path, map_location="cpu")
        # YOLOv5 checkpoint formats vary
        if isinstance(ckpt, dict):
            if "model" in ckpt and hasattr(ckpt["model"], "names"):
                return ckpt["model"].names
            if "ema" in ckpt and hasattr(ckpt["ema"], "names"):
                return ckpt["ema"].names
        # Sometimes ckpt itself is a model
        if hasattr(ckpt, "names"):
            return ckpt.names
    except Exception:
        pass
    return None


def read_plate_from_char_dets(char_dets, labels):
    """
    char_dets: list of [x1,y1,x2,y2,conf,cls] in plate-image coords (640 space scaled back)
    labels: list mapping cls->string (e.g., '0','1',...,'A',...,'-','.')
    Heuristic for 1-line / 2-line VN plates by y-center clustering.
    """
    if not char_dets:
        return "unknown"

    # sort by x then y for stable clustering
    chars = []
    for x1, y1, x2, y2, conf, cls in char_dets:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        chars.append((cx, cy, conf, cls))
    chars.sort(key=lambda t: (t[1], t[0]))

    # decide 1-line vs 2-line by y spread
    ys = np.array([c[1] for c in chars], dtype=np.float32)
    y_spread = float(ys.max() - ys.min()) if len(ys) else 0.0

    # estimate height
    # if y spread large => 2 lines
    two_line = y_spread > 35  # tuned for 640 crop; you can adjust

    if not two_line:
        chars.sort(key=lambda t: t[0])  # by x
        s = "".join(labels[c[3]] if c[3] < len(labels) else "?" for c in chars)
        return s

    # 2 lines: split by median y
    y_med = float(np.median(ys))
    top = [c for c in chars if c[1] <= y_med]
    bot = [c for c in chars if c[1] > y_med]
    top.sort(key=lambda t: t[0])
    bot.sort(key=lambda t: t[0])

    s1 = "".join(labels[c[3]] if c[3] < len(labels) else "?" for c in top)
    s2 = "".join(labels[c[3]] if c[3] < len(labels) else "?" for c in bot)
    return f"{s1}-{s2}"


# ----------------------------
# TensorRT Runner
# ----------------------------
@dataclass
class TrtBindings:
    host_inputs: list
    host_outputs: list
    device_inputs: list
    device_outputs: list
    bindings: list
    input_names: list
    output_names: list


class TrtRunner:
    def __init__(self, engine_path: str, logger_level: str = "WARNING"):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        level_map = {
            "VERBOSE": trt.Logger.VERBOSE,
            "INFO": trt.Logger.INFO,
            "WARNING": trt.Logger.WARNING,
            "ERROR": trt.Logger.ERROR,
            "INTERNAL_ERROR": trt.Logger.INTERNAL_ERROR,
        }
        self.logger = trt.Logger(level_map.get(logger_level.upper(), trt.Logger.WARNING))

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self.bind = self._allocate()

    def _allocate(self) -> TrtBindings:
        host_inputs, host_outputs = [], []
        device_inputs, device_outputs = [], []
        bindings = []
        input_names, output_names = [], []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            # for explicit batch, shape is known; for dynamic, set before allocate
            if -1 in shape:
                raise RuntimeError(f"Dynamic shape not supported in this simple runner: {name} {shape}")

            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                input_names.append(name)
                host_inputs.append(host_mem)
                device_inputs.append(device_mem)
            else:
                output_names.append(name)
                host_outputs.append(host_mem)
                device_outputs.append(device_mem)

        return TrtBindings(
            host_inputs=host_inputs,
            host_outputs=host_outputs,
            device_inputs=device_inputs,
            device_outputs=device_outputs,
            bindings=bindings,
            input_names=input_names,
            output_names=output_names,
        )

    def infer(self, input_array: np.ndarray):
        """
        input_array must be contiguous float32 with shape (1,3,640,640)
        """
        np.copyto(self.bind.host_inputs[0], input_array.ravel())

        cuda.memcpy_htod_async(self.bind.device_inputs[0], self.bind.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bind.bindings, stream_handle=self.stream.handle)
        for h_out, d_out in zip(self.bind.host_outputs, self.bind.device_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
        self.stream.synchronize()

        outs = []
        for out_name, h_out in zip(self.bind.output_names, self.bind.host_outputs):
            # reshape by engine binding shape
            idx = self.engine.get_binding_index(out_name)
            shape = tuple(self.engine.get_binding_shape(idx))
            outs.append(np.array(h_out).reshape(shape))
        return outs


# ----------------------------
# Video Sources
# ----------------------------
def make_csi_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 sync=false"
    )


def make_rtsp_pipeline(url, latency=250, tcp=True, width=1280, height=720):
    proto = "tcp" if tcp else "udp"
    return (
        f"rtspsrc location={url} latency={latency} protocols={proto} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={width},height={height} ! "
        f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false max-buffers=1"
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    ap.add_argument("--rtsp", type=str, default="")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--show", type=int, default=1)

    ap.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")

    ap.add_argument("--det_conf", type=float, default=0.35)
    ap.add_argument("--det_nms", type=float, default=0.45)
    ap.add_argument("--ocr_conf", type=float, default=0.25)
    ap.add_argument("--ocr_nms", type=float, default=0.45)

    ap.add_argument("--rtsp_latency", type=int, default=250)
    ap.add_argument("--rtsp_tcp", type=int, default=1)

    ap.add_argument("--in_w", type=int, default=640)
    ap.add_argument("--in_h", type=int, default=640)

    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)

    ap.add_argument("--log", type=str, default="WARNING")
    args = ap.parse_args()

    # labels for OCR (character classes)
    labels = load_labels_from_pt("model/LP_ocr_nano_62.pt")
    if labels is None:
        # fallback (may not match your model if your class order differs)
        labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.")
    labels = [str(x) for x in labels]

    det = TrtRunner(args.det_engine, logger_level=args.log)
    ocr = TrtRunner(args.ocr_engine, logger_level=args.log)

    # capture
    if args.source == "csi":
        pipeline = make_csi_pipeline(sensor_id=args.cam, width=args.csi_w, height=args.csi_h, fps=args.csi_fps, flip=args.flip)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        if not args.rtsp:
            raise SystemExit("Missing --rtsp url")
        pipeline = make_rtsp_pipeline(args.rtsp, latency=args.rtsp_latency, tcp=bool(args.rtsp_tcp))
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise SystemExit("Cannot open video source (check pipeline/permissions)")

    prev_t = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # avoid busy loop
            time.sleep(0.01)
            continue

        h0, w0 = frame.shape[:2]

        # ---- DETECTOR (plate) ----
        img, r, dwdh = letterbox(frame, (args.in_h, args.in_w), auto=False)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0).copy(order="C")

        det_out = det.infer(blob)[0]  # assume one output
        plate_dets = yolo_postprocess(det_out, conf_thres=args.det_conf, iou_thres=args.det_nms)

        plates_count = 0
        for x1, y1, x2, y2, conf, cls in plate_dets[:10]:
            # scale back to original frame
            box = scale_coords([x1, y1, x2, y2], r, dwdh)
            box = clip_box(box, w0, h0)
            bx1, by1, bx2, by2 = map(int, box)
            if bx2 <= bx1 or by2 <= by1:
                continue

            plates_count += 1

            # crop plate with padding
            pad = int(0.08 * max(bx2 - bx1, by2 - by1))
            cx1 = max(0, bx1 - pad)
            cy1 = max(0, by1 - pad)
            cx2 = min(w0, bx2 + pad)
            cy2 = min(h0, by2 + pad)
            plate = frame[cy1:cy2, cx1:cx2].copy()
            if plate.size == 0:
                continue

            # ---- OCR (character detection YOLO) ----
            p_img, pr, pdwdh = letterbox(plate, (args.in_h, args.in_w), auto=False)
            p_rgb = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
            p_blob = p_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            p_blob = np.expand_dims(p_blob, 0).copy(order="C")

            ocr_out = ocr.infer(p_blob)[0]
            char_dets = yolo_postprocess(ocr_out, conf_thres=args.ocr_conf, iou_thres=args.ocr_nms)

            # scale char boxes back to plate coords (not needed for sorting much, but better)
            ph, pw = plate.shape[:2]
            scaled_chars = []
            for cx1_, cy1_, cx2_, cy2_, cconf, ccls in char_dets:
                cbox = scale_coords([cx1_, cy1_, cx2_, cy2_], pr, pdwdh)
                cbox = clip_box(cbox, pw, ph)
                scaled_chars.append([*cbox, cconf, ccls])

            plate_text = read_plate_from_char_dets(scaled_chars, labels)

            # draw on original frame
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (bx1, max(0, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS
        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        cv2.putText(frame, f"FPS {fps:.1f} plates={plates_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        if args.show:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
