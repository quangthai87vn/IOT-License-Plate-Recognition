#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpr_trt.py - ALPR chạy CSI/USB webcam/RTSP trên Jetson Nano bằng TensorRT engine (.engine)

Run nhanh:
  CSI:
    python3 alpr_trt.py --source csi --show 1
  RTSP:
    python3 alpr_trt.py --source rtsp --rtsp "rtsp://192.168.50.2:8554/mac" --rtsp_tcp 1 --rtsp_latency 200 --show 1
  Webcam USB:
    python3 alpr_trt.py --source webcam --cam 0 --show 1
"""
import os
import time
import argparse

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


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


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(float(new_shape[0]) / float(shape[0]), float(new_shape[1]) / float(shape[1]))
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # w,h
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def to_blob(im_bgr, img_size):
    im, r, (dw, dh) = letterbox(im_bgr, img_size)
    im = im[:, :, ::-1]  # BGR->RGB
    im = im.astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # HWC->CHW
    im = np.expand_dims(im, 0)        # NCHW
    return im, r, (dw, dh)


def nms_boxes(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def yolo_decode(pred, conf_thres, iou_thres):
    pred = np.array(pred)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    if pred.shape[1] > 6:
        boxes = pred[:, :4]
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
        conf = obj * cls_conf
        m = conf > conf_thres
        boxes = boxes[m]
        conf = conf[m]
        cls_id = cls_id[m]
        xy = boxes[:, :2]
        wh = boxes[:, 2:4]
        boxes = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1)
    else:
        boxes = pred[:, :4]
        conf = pred[:, 4]
        cls_id = pred[:, 5].astype(np.int32)
        m = conf > conf_thres
        boxes = boxes[m]
        conf = conf[m]
        cls_id = cls_id[m]

    keep = nms_boxes(boxes, conf, iou_thres)
    return boxes[keep], conf[keep], cls_id[keep]


def scale_boxes(boxes, r, dwdh, orig_shape):
    dw, dh = dwdh
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r
    h, w = orig_shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


class TrtRunner(object):
    def __init__(self, engine_path, logger_severity="warning"):
        if not os.path.exists(engine_path):
            raise FileNotFoundError("Không thấy engine: %s" % engine_path)

        if logger_severity == "verbose":
            sev = trt.Logger.VERBOSE
        elif logger_severity == "info":
            sev = trt.Logger.INFO
        elif logger_severity == "error":
            sev = trt.Logger.ERROR
        else:
            sev = trt.Logger.WARNING

        self.logger = trt.Logger(sev)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)

            shape = tuple(self.context.get_binding_shape(i))
            if -1 in shape:
                shape = tuple(self.engine.get_binding_shape(i))

            size = int(trt.volume(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))

            item = {"index": i, "name": name, "dtype": dtype, "shape": shape, "host": host_mem, "device": dev_mem}
            if is_input:
                self.inputs.append(item)
            else:
                self.outputs.append(item)

    def infer(self, input_array):
        inp = self.inputs[0]
        try:
            self.context.set_binding_shape(inp["index"], tuple(input_array.shape))
        except Exception:
            pass

        np.copyto(inp["host"], input_array.ravel())
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        outs = []
        for out in self.outputs:
            try:
                shape = tuple(self.context.get_binding_shape(out["index"]))
            except Exception:
                shape = out["shape"]
            outs.append(np.array(out["host"]).reshape(shape))
        return outs


def load_names(names_path):
    if not os.path.exists(names_path):
        raise FileNotFoundError("Không thấy file .names: %s" % names_path)
    names = []
    with open(names_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                names.append(t)
    return names


def decode_plate(chars_boxes, chars_cls, labels):
    if len(chars_boxes) == 0:
        return "unknown"

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

    text = ""
    for i in order:
        c = int(chars_cls[i])
        if 0 <= c < len(labels):
            text += labels[c]

    text = text.replace(" ", "").strip()
    return text if text else "unknown"


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
    p.add_argument("--det_conf", type=float, default=0.35)
    p.add_argument("--det_nms", type=float, default=0.45)
    p.add_argument("--ocr_conf", type=float, default=0.25)
    p.add_argument("--ocr_nms", type=float, default=0.45)

    p.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    p.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")
    p.add_argument("--ocr_names", type=str, default="model/LP_ocr_nano_62.names")
    p.add_argument("--log", type=str, default="warning", choices=["warning", "info", "error", "verbose"])
    return p.parse_args()


def open_capture(args):
    if args.source == "csi":
        cap = cv2.VideoCapture(gst_csi(args.cam, args.csi_w, args.csi_h, args.csi_fps, args.flip), cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        if not args.rtsp:
            raise ValueError("Bạn chọn --source rtsp nhưng chưa truyền --rtsp URL")
        cap = cv2.VideoCapture(gst_rtsp(args.rtsp, args.rtsp_latency, bool(args.rtsp_tcp), args.in_w, args.in_h), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Không mở được video source (%s)" % args.source)
    return cap


def main():
    args = parse_args()

    labels = load_names(args.ocr_names)
    det = TrtRunner(args.det_engine, logger_severity=args.log)
    ocr = TrtRunner(args.ocr_engine, logger_severity=args.log)

    cap = open_capture(args)

    t0 = time.time()
    frames = 0
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

        blob, r, dwdh = to_blob(frame, args.img)
        det_pred = det.infer(blob)[0]
        det_boxes, det_scores, det_cls = yolo_decode(det_pred, args.det_conf, args.det_nms)
        det_boxes = scale_boxes(det_boxes, r, dwdh, frame.shape)

        plate_texts = []
        max_plates = min(5, len(det_boxes))

        for i in range(max_plates):
            x1, y1, x2, y2 = det_boxes[i].astype(np.int32).tolist()
            pad = int(0.05 * max(x2 - x1, y2 - y1))
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(frame.shape[1] - 1, x2 + pad)
            y2p = min(frame.shape[0] - 1, y2 + pad)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            p_blob, pr, pdwdh = to_blob(crop, args.img)
            ocr_pred = ocr.infer(p_blob)[0]
            c_boxes, c_scores, c_cls = yolo_decode(ocr_pred, args.ocr_conf, args.ocr_nms)
            c_boxes = scale_boxes(c_boxes, pr, pdwdh, crop.shape)

            text = decode_plate(c_boxes, c_cls, labels)
            plate_texts.append(text)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        hud = "FPS %.1f plates=%d" % (fps, len(plate_texts))
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        if args.show == 1:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):
                break
        else:
            if frames % 30 == 0:
                print(hud, "->", plate_texts)

    cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
