#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np

# TensorRT + PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# ----------------------------
# Utils: letterbox + NMS
# ----------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad giữ tỉ lệ (yolo style). Return: img, ratio, (dw, dh)"""
    shape = im.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im_padded, r, (dw, dh)


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_numpy(boxes, scores, iou_thres=0.45):
    """Pure numpy NMS. boxes: (N,4) xyxy"""
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


# ----------------------------
# TensorRT runner
# ----------------------------
class TrtRunner:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        # TRT 8.x: Logger phải dùng Severity enum
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Cannot load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create execution context")

        self.stream = cuda.Stream()

        # Bindings
        self.bindings = [None] * self.engine.num_bindings
        self.host_mem = {}
        self.dev_mem = {}

        # assume 1 input
        self.input_idx = None
        self.output_idxs = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_idx = i
            else:
                self.output_idxs.append(i)

        if self.input_idx is None:
            raise RuntimeError("No input binding found")

        # allocate later after set shape
        self.allocated = False

    def allocate(self, input_shape):
        """input_shape: (1,3,H,W)"""
        self.context.set_binding_shape(self.input_idx, input_shape)

        # Input alloc
        in_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        in_size = trt.volume(self.context.get_binding_shape(self.input_idx))
        self.host_mem[self.input_idx] = cuda.pagelocked_empty(in_size, dtype=in_dtype)
        self.dev_mem[self.input_idx] = cuda.mem_alloc(self.host_mem[self.input_idx].nbytes)
        self.bindings[self.input_idx] = int(self.dev_mem[self.input_idx])

        # Outputs alloc
        for oi in self.output_idxs:
            out_shape = tuple(self.context.get_binding_shape(oi))
            out_dtype = trt.nptype(self.engine.get_binding_dtype(oi))
            out_size = trt.volume(out_shape)
            self.host_mem[oi] = cuda.pagelocked_empty(out_size, dtype=out_dtype)
            self.dev_mem[oi] = cuda.mem_alloc(self.host_mem[oi].nbytes)
            self.bindings[oi] = int(self.dev_mem[oi])

        self.allocated = True

    def infer(self, input_tensor: np.ndarray):
        """input_tensor shape: (1,3,H,W) float16/float32 contiguous"""
        if not self.allocated:
            self.allocate(input_tensor.shape)

        # H2D
        np.copyto(self.host_mem[self.input_idx], input_tensor.ravel())
        cuda.memcpy_htod_async(self.dev_mem[self.input_idx], self.host_mem[self.input_idx], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # D2H outputs
        outputs = []
        for oi in self.output_idxs:
            cuda.memcpy_dtoh_async(self.host_mem[oi], self.dev_mem[oi], self.stream)
        self.stream.synchronize()

        for oi in self.output_idxs:
            out_shape = tuple(self.context.get_binding_shape(oi))
            outputs.append(self.host_mem[oi].reshape(out_shape))

        # Nếu model chỉ có 1 output thì trả luôn output đó
        if len(outputs) == 1:
            return outputs[0]
        return outputs


# ----------------------------
# YOLOv5 postprocess (ONNX/TRT)
# output expected: (1, N, 5+nc)
# ----------------------------
def yolo_postprocess(pred, conf_thres=0.25, iou_thres=0.45):
    pred = np.asarray(pred)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if pred.ndim != 3:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")

    p = pred[0]  # (N, 5+nc)
    if p.shape[1] < 6:
        raise ValueError(f"Output looks wrong: {p.shape} (need >=6)")

    obj = p[:, 4]
    cls_scores = p[:, 5:]
    cls_id = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(len(cls_scores)), cls_id]
    conf = obj * cls_conf

    keep = conf >= conf_thres
    p = p[keep]
    conf = conf[keep]
    cls_id = cls_id[keep]

    if len(p) == 0:
        return np.empty((0, 6), dtype=np.float32)

    boxes = xywh2xyxy(p[:, 0:4])
    keep_idx = nms_numpy(boxes, conf, iou_thres=iou_thres)

    dets = np.concatenate([boxes[keep_idx], conf[keep_idx, None], cls_id[keep_idx, None].astype(np.float32)], axis=1)
    return dets  # (M,6) => x1,y1,x2,y2,conf,cls


# ----------------------------
# OCR decode (char-detector): sort chars (1 dòng / 2 dòng)
# ----------------------------
DEFAULT_OCR_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["-", "."]

def decode_plate_from_char_dets(char_dets, ocr_classes):
    """
    char_dets: (K,6) xyxy conf cls
    Return string
    """
    if len(char_dets) == 0:
        return "unknown"

    # lấy center để gom dòng
    centers = np.column_stack([
        (char_dets[:, 0] + char_dets[:, 2]) / 2.0,
        (char_dets[:, 1] + char_dets[:, 3]) / 2.0
    ])
    ys = centers[:, 1]

    # Heuristic: nếu spread theo y lớn -> biển 2 dòng
    y_span = ys.max() - ys.min()
    h_mean = np.mean(char_dets[:, 3] - char_dets[:, 1])
    two_lines = y_span > 0.6 * h_mean

    def cls_to_char(c):
        c = int(c)
        if 0 <= c < len(ocr_classes):
            return ocr_classes[c]
        return "?"

    if not two_lines:
        # sort theo x
        order = np.argsort(centers[:, 0])
        chars = [cls_to_char(char_dets[i, 5]) for i in order]
        return "".join(chars)

    # 2 dòng: tách theo y median
    y_med = np.median(ys)
    top_idx = np.where(ys <= y_med)[0]
    bot_idx = np.where(ys > y_med)[0]

    top_order = top_idx[np.argsort(centers[top_idx, 0])]
    bot_order = bot_idx[np.argsort(centers[bot_idx, 0])]

    top = "".join(cls_to_char(char_dets[i, 5]) for i in top_order)
    bot = "".join(cls_to_char(char_dets[i, 5]) for i in bot_order)

    # format kiểu VN hay dùng: "63-B9\n951.64" -> mình nối bằng "-" nếu cần
    return f"{top}-{bot}" if top and bot else (top + bot)


# ----------------------------
# VideoCapture via GStreamer
# ----------------------------
def gst_csi(sensor_id=0, w=1280, h=720, fps=30, flip=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )

def gst_rtsp(url, w=1280, h=720, latency=200, tcp=True, codec="h264"):
    # codec: h264 / h265
    if codec.lower() == "h265":
        depay = "rtph265depay"
    else:
        depay = "rtph264depay"

    protocols = "tcp" if tcp else "udp"
    return (
        f"rtspsrc location={url} protocols={protocols} latency={latency} drop-on-latency=true ! "
        f"{depay} ! "
        f"h264parse ! "
        f"nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! video/x-raw,format=BGRx,width={w},height={h} ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


# ----------------------------
# Main
# ----------------------------
def load_class_names(path, fallback):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        return names if names else fallback
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    ap.add_argument("--rtsp", type=str, default="")
    ap.add_argument("--cam", type=int, default=0)

    ap.add_argument("--show", type=int, default=1)

    # capture params
    ap.add_argument("--csi_w", type=int, default=1280)
    ap.add_argument("--csi_h", type=int, default=720)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)

    ap.add_argument("--rtsp_w", type=int, default=1280)
    ap.add_argument("--rtsp_h", type=int, default=720)
    ap.add_argument("--rtsp_latency", type=int, default=200)
    ap.add_argument("--rtsp_tcp", type=int, default=1)
    ap.add_argument("--rtsp_codec", choices=["h264", "h265"], default="h264")

    # engines
    ap.add_argument("--det_engine", type=str, default="model/LP_detector_nano_61_fp16.engine")
    ap.add_argument("--ocr_engine", type=str, default="model/LP_ocr_nano_62_fp16.engine")

    # yolo input
    ap.add_argument("--imgsz", type=int, default=640)

    # thresholds
    ap.add_argument("--det_conf", type=float, default=0.35)
    ap.add_argument("--det_iou", type=float, default=0.45)
    ap.add_argument("--ocr_conf", type=float, default=0.50)
    ap.add_argument("--ocr_iou", type=float, default=0.45)

    # OCR classes
    ap.add_argument("--ocr_classes", type=str, default="")  # txt file, mỗi dòng 1 class

    # speed
    ap.add_argument("--ocr_every", type=int, default=2)  # OCR mỗi N frame để tăng FPS
    args = ap.parse_args()

    ocr_classes = load_class_names(args.ocr_classes, DEFAULT_OCR_CLASSES)

    # Open video
    if args.source == "csi":
        cap = cv2.VideoCapture(gst_csi(0, args.csi_w, args.csi_h, args.csi_fps, args.flip), cv2.CAP_GSTREAMER)
    elif args.source == "rtsp":
        if not args.rtsp:
            raise SystemExit("Missing --rtsp URL")
        cap = cv2.VideoCapture(
            gst_rtsp(args.rtsp, args.rtsp_w, args.rtsp_h, args.rtsp_latency, bool(args.rtsp_tcp), args.rtsp_codec),
            cv2.CAP_GSTREAMER
        )
    else:
        cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise SystemExit("❌ Cannot open camera/stream. Check GStreamer pipeline / permissions / URL.")

    # Load TRT engines
    det_trt = TrtRunner(args.det_engine)
    ocr_trt = TrtRunner(args.ocr_engine)

    # FPS
    t0 = time.time()
    frame_id = 0
    fps = 0.0

    last_plate_text = {}  # track by bbox key

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # RTSP hay drop frame => cứ continue nhẹ
            time.sleep(0.01)
            continue

        frame_id += 1

        # DET preprocess
        img, r, (dw, dh) = letterbox(frame, (args.imgsz, args.imgsz))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = img_rgb.transpose(2, 0, 1).astype(np.float16) / 255.0
        inp = np.expand_dims(inp, 0).copy(order="C")

        pred = det_trt.infer(inp)
        dets = yolo_postprocess(pred, conf_thres=args.det_conf, iou_thres=args.det_iou)

        plates_count = 0

        # draw dets + OCR
        for (x1, y1, x2, y2, conf, cls_id) in dets:
            # scale back to original frame
            # undo padding + ratio
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r

            x1 = int(max(0, x1)); y1 = int(max(0, y1))
            x2 = int(min(frame.shape[1] - 1, x2)); y2 = int(min(frame.shape[0] - 1, y2))

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            plates_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # OCR mỗi N frame để tăng FPS
            bbox_key = (x1//10, y1//10, x2//10, y2//10)
            do_ocr = (frame_id % args.ocr_every == 0) or (bbox_key not in last_plate_text)

            if do_ocr:
                crop = frame[y1:y2, x1:x2].copy()

                # OCR preprocess: IMPORTANT
                # Nếu OCR model là YOLO char-detector, thường cũng cần letterbox về 640x640
                oimg, orr, (odw, odh) = letterbox(crop, (args.imgsz, args.imgsz))
                oimg_rgb = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
                oinp = oimg_rgb.transpose(2, 0, 1).astype(np.float16) / 255.0
                oinp = np.expand_dims(oinp, 0).copy(order="C")

                opred = ocr_trt.infer(oinp)
                odets = yolo_postprocess(opred, conf_thres=args.ocr_conf, iou_thres=args.ocr_iou)

                # scale char boxes back to crop coords
                if len(odets) > 0:
                    # undo pad/ratio
                    odets[:, 0] = (odets[:, 0] - odw) / orr
                    odets[:, 1] = (odets[:, 1] - odh) / orr
                    odets[:, 2] = (odets[:, 2] - odw) / orr
                    odets[:, 3] = (odets[:, 3] - odh) / orr

                    # clamp
                    odets[:, 0] = np.clip(odets[:, 0], 0, crop.shape[1]-1)
                    odets[:, 2] = np.clip(odets[:, 2], 0, crop.shape[1]-1)
                    odets[:, 1] = np.clip(odets[:, 1], 0, crop.shape[0]-1)
                    odets[:, 3] = np.clip(odets[:, 3], 0, crop.shape[0]-1)

                text = decode_plate_from_char_dets(odets, ocr_classes)
                last_plate_text[bbox_key] = text
            else:
                text = last_plate_text.get(bbox_key, "unknown")

            cv2.putText(frame, text, (x1, max(30, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # FPS calc
        t1 = time.time()
        dt = t1 - t0
        if dt >= 1.0:
            fps = frame_id / dt
            frame_id = 0
            t0 = t1

        cv2.putText(frame, f"FPS {fps:.1f} plates={plates_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.show == 1:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
