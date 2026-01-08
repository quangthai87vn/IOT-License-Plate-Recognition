#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import cv2
import numpy as np

# TensorRT + CUDA (JetPack thường có sẵn)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# -------------------------
# GStreamer pipelines
# -------------------------
def gst_csi(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink max-buffers=1 drop=1 sync=false"
        % (sensor_id, width, height, fps, flip)
    )


def gst_rtsp(url, latency=200, tcp=True, width=1280, height=720):
    proto = "tcp" if tcp else "udp"
    return (
        "rtspsrc location=%s latency=%d protocols=%s drop-on-latency=true ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx,width=%d,height=%d ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=false max-buffers=1"
        % (url, latency, proto, width, height)
    )


# -------------------------
# Letterbox
# -------------------------
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)


def xywh2xyxy(x):
    # x: [N,4] (cx,cy,w,h)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes, scores, iou_thres=0.45):
    # boxes: [N,4] xyxy
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

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

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess_yolov5(pred, conf_thres, iou_thres, ratio, pad, orig_shape):
    """
    pred: (num, no) where no = 5 + nc, format [cx,cy,w,h,obj,cls...]
    returns: list [x1,y1,x2,y2,score,cls]
    """
    if pred is None or len(pred) == 0:
        return []

    if pred.ndim == 3:
        pred = pred[0]

    # obj conf
    obj = pred[:, 4]
    cls_probs = pred[:, 5:] if pred.shape[1] > 6 else None

    if cls_probs is None:
        # single-class model output could be [cx,cy,w,h,conf,cls]
        # but YOLOv5 export usually still (5+nc). If not, fallback:
        scores = obj
        cls = np.zeros_like(scores, dtype=np.int32)
    else:
        cls = np.argmax(cls_probs, axis=1).astype(np.int32)
        cls_score = cls_probs[np.arange(cls_probs.shape[0]), cls]
        scores = obj * cls_score

    keep = scores >= conf_thres
    pred = pred[keep]
    scores = scores[keep]
    if pred.shape[0] == 0:
        return []

    cls = cls[keep] if cls_probs is not None else np.zeros((pred.shape[0],), dtype=np.int32)

    boxes = xywh2xyxy(pred[:, 0:4].copy())

    # scale back to original
    left, top = pad
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= ratio

    h0, w0 = orig_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w0 - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h0 - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w0 - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h0 - 1)

    # NMS per-class (ổn hơn OCR)
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        b = boxes[idx]
        s = scores[idx]
        k = nms_boxes(b, s, iou_thres=iou_thres)
        for kk in k:
            x1, y1, x2, y2 = b[kk]
            out.append([float(x1), float(y1), float(x2), float(y2), float(s[kk]), int(c)])

    # sort by score desc
    out.sort(key=lambda x: x[4], reverse=True)
    return out


# -------------------------
# OCR decoding (2-line plate)
# -------------------------
def decode_plate(chars_boxes, chars_cls, labels):
    if len(chars_boxes) == 0:
        return "unknown"

    chars_boxes = np.asarray(chars_boxes, dtype=np.float32)
    chars_cls = np.asarray(chars_cls, dtype=np.int32)

    cx = (chars_boxes[:, 0] + chars_boxes[:, 2]) / 2.0
    cy = (chars_boxes[:, 1] + chars_boxes[:, 3]) / 2.0
    h = (chars_boxes[:, 3] - chars_boxes[:, 1])

    y_spread = float(np.max(cy) - np.min(cy))
    h_med = float(np.median(h)) if len(h) else 0.0
    idx = np.arange(len(chars_boxes))

    if h_med > 0 and y_spread > 0.6 * h_med:
        mid = float(np.median(cy))
        row1 = idx[cy <= mid]
        row2 = idx[cy > mid]
        row1 = row1[np.argsort(cx[row1])]
        row2 = row2[np.argsort(cx[row2])]
        order = np.concatenate([row1, row2], axis=0)
    else:
        order = idx[np.argsort(cx)]

    out = []
    for i in order:
        c = int(chars_cls[i])
        if 0 <= c < len(labels):
            out.append(labels[c])
    text = "".join(out).replace(" ", "").strip()
    return text if text else "unknown"


def load_labels(names_path):
    if not names_path or not os.path.exists(names_path):
        return None
    labels = []
    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                labels.append(t)
    return labels if labels else None


# -------------------------
# TensorRT wrapper
# -------------------------
class TRTInfer(object):
    def __init__(self, onnx_path, engine_path, fp16=True, verbose=False):
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.fp16 = fp16
        self.verbose = verbose

        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = None
        self.context = None

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.engine_path):
            with open(self.engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        else:
            self.engine = self._build_engine_from_onnx(self.onnx_path, self.engine_path, self.fp16)

        if self.engine is None:
            raise RuntimeError("Không load/build được TensorRT engine: %s" % self.engine_path)

        self.context = self.engine.create_execution_context()
        self._allocate_buffers()

    def _build_engine_from_onnx(self, onnx_path, engine_path, fp16):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError("Không thấy ONNX: %s" % onnx_path)

        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("[TRT][ONNXParser]", parser.get_error(i))
                return None

        config = builder.create_builder_config()
        config.max_workspace_size = 1024 * 1024 * 1024  # 1GB

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # profile (static shape 1x3x640x640 vẫn cần explicit profile trong TRT8)
        profile = builder.create_optimization_profile()
        inp = network.get_input(0)
        in_name = inp.name
        in_shape = tuple(inp.shape)  # e.g. (1,3,640,640)
        profile.set_shape(in_name, in_shape, in_shape, in_shape)
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        if engine is None:
            return None

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine

    def _allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            shape = self.context.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = int(trt.volume(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append((binding, host_mem, device_mem, shape, dtype))
            else:
                self.outputs.append((binding, host_mem, device_mem, shape, dtype))

    def infer(self, input_np):
        # input_np must be contiguous and match input shape flattened
        in_name, in_host, in_dev, in_shape, in_dtype = self.inputs[0]

        np.copyto(in_host, input_np.ravel())

        cuda.memcpy_htod_async(in_dev, in_host, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        outs = []
        for (name, host, dev, shape, dtype) in self.outputs:
            cuda.memcpy_dtoh_async(host, dev, self.stream)
        self.stream.synchronize()

        for (name, host, dev, shape, dtype) in self.outputs:
            out = np.array(host, copy=True).reshape(shape)
            outs.append(out)
        return outs


# -------------------------
# Capture open
# -------------------------
def open_capture(args):
    if args.source == "csi":
        cap = cv2.VideoCapture(
            gst_csi(args.cam, args.csi_w, args.csi_h, args.csi_fps, args.flip),
            cv2.CAP_GSTREAMER,
        )
    elif args.source == "rtsp":
        cap = cv2.VideoCapture(
            gst_rtsp(args.rtsp, args.rtsp_latency, bool(args.rtsp_tcp), args.in_w, args.in_h),
            cv2.CAP_GSTREAMER,
        )
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Không mở được nguồn video: %s" % args.source)
    return cap


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--show", type=int, default=1)

    p.add_argument("--rtsp", type=str, default="")
    p.add_argument("--rtsp_latency", type=int, default=200)
    p.add_argument("--rtsp_tcp", type=int, default=1)

    p.add_argument("--csi_w", type=int, default=1280)
    p.add_argument("--csi_h", type=int, default=720)
    p.add_argument("--csi_fps", type=int, default=30)
    p.add_argument("--flip", type=int, default=0)

    p.add_argument("--in_w", type=int, default=1280)
    p.add_argument("--in_h", type=int, default=720)

    p.add_argument("--img", type=int, default=640)

    p.add_argument("--det_onnx", type=str, default="model/LP_detector_nano_61.onnx")
    p.add_argument("--ocr_onnx", type=str, default="model/LP_ocr_nano_62.onnx")

    p.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    p.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")

    p.add_argument("--ocr_names", type=str, default="model/LP_ocr_nano_62.names")

    p.add_argument("--det_conf", type=float, default=0.35)
    p.add_argument("--det_iou", type=float, default=0.45)
    p.add_argument("--ocr_conf", type=float, default=0.25)
    p.add_argument("--ocr_iou", type=float, default=0.45)

    p.add_argument("--fp16", type=int, default=1)
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "rtsp" and not args.rtsp:
        raise ValueError("Bạn chọn --source rtsp thì phải truyền --rtsp URL")

    labels = load_labels(args.ocr_names)
    if not labels:
        raise RuntimeError("Không đọc được OCR labels từ file: %s" % args.ocr_names)

    # Load / Build TRT engines
    det_trt = TRTInfer(args.det_onnx, args.det_engine, fp16=bool(args.fp16), verbose=bool(args.verbose))
    ocr_trt = TRTInfer(args.ocr_onnx, args.ocr_engine, fp16=bool(args.fp16), verbose=bool(args.verbose))

    cap = open_capture(args)

    frames = 0
    t0 = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frames += 1
        if frames % 10 == 0:
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0.0

        orig = frame.copy()

        # ----- Detector preprocess
        img_det, r_det, pad_det = letterbox(orig, args.img)
        img_det = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        img_det = img_det.transpose(2, 0, 1)  # CHW
        img_det = np.ascontiguousarray(img_det, dtype=np.float32) / 255.0
        img_det = img_det.reshape(1, 3, args.img, args.img)

        det_outs = det_trt.infer(img_det)
        det_pred = det_outs[0]  # usually (1,25200, 5+nc)

        dets = postprocess_yolov5(det_pred, args.det_conf, args.det_iou, r_det, pad_det, orig.shape)

        texts = []
        for d in dets[:5]:
            x1, y1, x2, y2, score, clsid = d
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            pad = int(0.05 * max(x2 - x1, y2 - y1))
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(orig.shape[1] - 1, x2 + pad)
            y2p = min(orig.shape[0] - 1, y2 + pad)

            crop = orig[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            # ----- OCR preprocess (on crop)
            img_ocr, r_ocr, pad_ocr = letterbox(crop, args.img)
            img_ocr = cv2.cvtColor(img_ocr, cv2.COLOR_BGR2RGB)
            img_ocr = img_ocr.transpose(2, 0, 1)
            img_ocr = np.ascontiguousarray(img_ocr, dtype=np.float32) / 255.0
            img_ocr = img_ocr.reshape(1, 3, args.img, args.img)

            ocr_outs = ocr_trt.infer(img_ocr)
            ocr_pred = ocr_outs[0]

            chars = postprocess_yolov5(ocr_pred, args.ocr_conf, args.ocr_iou, r_ocr, pad_ocr, crop.shape)

            boxes = []
            clss = []
            for c in chars:
                ox1, oy1, ox2, oy2, s, cid = c
                boxes.append([ox1, oy1, ox2, oy2])
                clss.append(cid)

            text = decode_plate(np.array(boxes, dtype=np.float32), np.array(clss, dtype=np.int32), labels)
            texts.append(text)

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        hud = "FPS %.1f plates=%d" % (fps, len(texts))
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)

        if args.show == 1:
            cv2.imshow("ALPR_ONNX_TRT", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):
                break
        else:
            if frames % 30 == 0:
                print(hud, "->", texts)

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
