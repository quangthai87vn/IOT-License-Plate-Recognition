#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ALPR chạy .pt trên Jetson Nano (PyTorch YOLOv5)
- Detector: model/LP_detector_nano_61.pt
- OCR:      model/LP_ocr_nano_62.pt

Run:
  CSI:
    python3 alpr_pt_jetson.py --source csi --show 1
  RTSP:
    python3 alpr_pt_jetson.py --source rtsp --rtsp "rtsp://192.168.50.2:8554/mac" --rtsp_tcp 1 --rtsp_latency 200 --show 1
  Webcam:
    python3 alpr_pt_jetson.py --source webcam --cam 0 --show 1
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np

import torch

# -------------------------
# GStreamer helpers
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
# OCR decode helpers
# -------------------------
def decode_plate(chars_boxes, chars_cls, labels):
    """Sắp xếp ký tự theo hàng + theo trục x rồi ghép text."""
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

    # 2 dòng (VN plate): nếu y_spread đủ lớn
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

def load_names_fallback(path):
    if not path or not os.path.exists(path):
        return None
    names = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                names.append(t)
    return names if names else None

# -------------------------
# YOLOv5 glue
# -------------------------
def setup_yolov5_path():
    """
    Ưu tiên ./yolov5, nếu không có thì ./yolov5_v5
    """
    base = os.path.abspath(os.path.dirname(__file__))
    cand = [os.path.join(base, "yolov5"), os.path.join(base, "yolov5_v5")]
    for p in cand:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "models")) and os.path.isdir(os.path.join(p, "utils")):
            sys.path.insert(0, p)
            return p
    raise RuntimeError("Không tìm thấy thư mục yolov5/ hoặc yolov5_v5/ (có models/ và utils/).")

def load_yolov5_pt(weights, device):
    """
    Load YOLOv5 .pt theo kiểu repo YOLOv5 (attempt_load).
    """
    from models.experimental import attempt_load  # type: ignore
    model = attempt_load(weights, map_location=device)
    model.eval()
    return model

def infer_yolov5(model, im_bgr, img_size, device, conf_thres, iou_thres, classes=None, half=False):
    """
    Chạy inference YOLOv5:
    - letterbox + tensor
    - model forward
    - non_max_suppression + scale_coords
    Trả về list dets: [x1,y1,x2,y2,conf,cls]
    """
    from utils.augmentations import letterbox  # type: ignore
    from utils.general import non_max_suppression, scale_coords  # type: ignore

    im0 = im_bgr
    im = letterbox(im0, new_shape=img_size, auto=False)[0]
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = model(im)[0]  # (bs, n, 5+nc)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)

    dets = []
    if pred and pred[0] is not None and len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        dets = det.detach().cpu().numpy().tolist()

    return dets

# -------------------------
# Args + Capture
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

    p.add_argument("--det_pt", type=str, default="model/LP_detector_nano_61.pt")
    p.add_argument("--ocr_pt", type=str, default="model/LP_ocr_nano_62.pt")
    p.add_argument("--ocr_names", type=str, default="model/LP_ocr_nano_62.names")  # fallback nếu model.names thiếu

    p.add_argument("--det_conf", type=float, default=0.35)
    p.add_argument("--det_iou", type=float, default=0.45)
    p.add_argument("--ocr_conf", type=float, default=0.25)
    p.add_argument("--ocr_iou", type=float, default=0.45)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--half", type=int, default=1)  # fp16 on cuda
    return p.parse_args()

def open_capture(args):
    if args.source == "csi":
        cap = cv2.VideoCapture(
            gst_csi(args.cam, args.csi_w, args.csi_h, args.csi_fps, args.flip),
            cv2.CAP_GSTREAMER,
        )
    elif args.source == "rtsp":
        if not args.rtsp:
            raise ValueError("Bạn chọn --source rtsp nhưng chưa truyền --rtsp URL")
        cap = cv2.VideoCapture(
            gst_rtsp(args.rtsp, args.rtsp_latency, bool(args.rtsp_tcp), args.in_w, args.in_h),
            cv2.CAP_GSTREAMER,
        )
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Không mở được video source: %s" % args.source)
    return cap

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # Setup YOLOv5 import path
    y5_path = setup_yolov5_path()
    print("[INFO] Using YOLOv5 repo at:", y5_path)

    # Device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA không available, fallback CPU")
        device = torch.device("cpu")
        half = False
    else:
        device = torch.device(args.device)
        half = bool(args.half) and (device.type == "cuda")

    # Load models
    if not os.path.exists(args.det_pt):
        raise FileNotFoundError("Không thấy det pt: %s" % args.det_pt)
    if not os.path.exists(args.ocr_pt):
        raise FileNotFoundError("Không thấy ocr pt: %s" % args.ocr_pt)

    det_model = load_yolov5_pt(args.det_pt, device)
    ocr_model = load_yolov5_pt(args.ocr_pt, device)

    # half (fp16) nếu CUDA
    if half:
        det_model.half()
        ocr_model.half()

    # labels OCR
    labels = None
    # YOLOv5 thường có model.names
    try:
        labels = getattr(ocr_model, "names", None)
    except Exception:
        labels = None
    if not labels:
        labels = load_names_fallback(args.ocr_names)

    if not labels:
        raise RuntimeError("Không lấy được labels OCR (model.names rỗng và không đọc được .names).")

    print("[INFO] OCR labels:", labels)

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

        # 1) Detect plate
        dets = infer_yolov5(
            det_model, frame, args.img, device,
            args.det_conf, args.det_iou,
            classes=None, half=half
        )

        plate_texts = []
        for d in dets[:5]:
            x1, y1, x2, y2, conf, cls = d
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # pad crop
            pad = int(0.05 * max(x2 - x1, y2 - y1))
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(frame.shape[1] - 1, x2 + pad)
            y2p = min(frame.shape[0] - 1, y2 + pad)

            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            # 2) OCR
            ocr_dets = infer_yolov5(
                ocr_model, crop, args.img, device,
                args.ocr_conf, args.ocr_iou,
                classes=None, half=half
            )

            # convert ocr dets -> boxes + cls
            boxes = []
            clss = []
            for od in ocr_dets:
                ox1, oy1, ox2, oy2, oconf, ocls = od
                boxes.append([ox1, oy1, ox2, oy2])
                clss.append(int(ocls))

            text = decode_plate(np.array(boxes, dtype=np.float32), np.array(clss, dtype=np.int32), labels)
            plate_texts.append(text)

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        hud = "FPS %.1f plates=%d" % (fps, len(plate_texts))
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2, cv2.LINE_AA)

        if args.show == 1:
            cv2.imshow("ALPR_PT", frame)
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
