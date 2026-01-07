#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse

import cv2
import torch

import function.utils_rotate as utils_rotate
import function.helper as helper


def pick_yolo_repo():
    # repo bạn có thể là ./yolov5 hoặc ./yolov5_v5
    if os.path.isdir("./yolov5"):
        return "./yolov5"
    if os.path.isdir("./yolov5_v5"):
        return "./yolov5_v5"
    return "./yolov5"  # fallback


def load_yolov5_custom(repo_dir: str, weights_path: str):
    """
    Load YOLOv5 custom model từ local repo (không download internet).
    """
    try:
        return torch.hub.load(repo_dir, "custom", path=weights_path, source="local")
    except TypeError:
        # Một số bản YOLOv5 dùng param "weights"
        return torch.hub.load(repo_dir, "custom", weights=weights_path, source="local")


def gst_csi_pipeline(sensor_id=0, csi_w=1640, csi_h=1232, fps=30, flip=0, out_w=1280, out_h=720):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={csi_w}, height={csi_h}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def gst_rtsp_pipeline(url, latency=200, tcp=True, out_w=1280, out_h=720):
    proto = "tcp" if tcp else "udp"
    # Jetson decode H264 tốt nhất bằng nvv4l2decoder
    return (
        f"rtspsrc location={url} latency={latency} protocols={proto} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
        f"nvvidconv ! video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )


def open_capture(args):
    if args.source == "csi":
        pipe = gst_csi_pipeline(
            sensor_id=args.cam,
            csi_w=args.csi_w,
            csi_h=args.csi_h,
            fps=args.csi_fps,
            flip=args.flip,
            out_w=args.out_w,
            out_h=args.out_h
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    if args.source == "rtsp":
        if not args.rtsp:
            raise SystemExit("Thiếu --rtsp URL")
        pipe = gst_rtsp_pipeline(
            args.rtsp,
            latency=args.rtsp_latency,
            tcp=not args.rtsp_udp,
            out_w=args.out_w,
            out_h=args.out_h
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        return cap

    # webcam USB / laptop cam
    cap = cv2.VideoCapture(args.webcam)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csi", "rtsp", "webcam"], default="csi")
    ap.add_argument("--show", type=int, default=1)

    # CSI
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--csi_w", type=int, default=1640)
    ap.add_argument("--csi_h", type=int, default=1232)
    ap.add_argument("--csi_fps", type=int, default=30)
    ap.add_argument("--flip", type=int, default=0)
    ap.add_argument("--out_w", type=int, default=1280)
    ap.add_argument("--out_h", type=int, default=720)

    # RTSP
    ap.add_argument("--rtsp", type=str, default="")
    ap.add_argument("--rtsp_latency", type=int, default=250)
    ap.add_argument("--rtsp_udp", action="store_true")  # mặc định TCP để đỡ nhoè

    # webcam
    ap.add_argument("--webcam", type=int, default=0)

    # model paths
    ap.add_argument("--det", type=str, default="model/LP_detector_nano_61.pt")
    ap.add_argument("--ocr", type=str, default="model/LP_ocr_nano_62.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--ocr_conf", type=float, default=0.60)

    args = ap.parse_args()

    # Torch optimize
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    repo_dir = pick_yolo_repo()

    print(f"[INFO] YOLO repo = {repo_dir}")
    print(f"[INFO] DET model = {args.det}")
    print(f"[INFO] OCR model = {args.ocr}")

    yolo_det = load_yolov5_custom(repo_dir, args.det)
    yolo_ocr = load_yolov5_custom(repo_dir, args.ocr)

    # set confidence
    yolo_det.conf = args.det_conf
    yolo_ocr.conf = args.ocr_conf

    cap = open_capture(args)
    if not cap.isOpened():
        raise SystemExit("[ERROR] Không mở được nguồn video. Kiểm tra GStreamer/URL/cam.")

    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # RTSP đôi khi hụt frame, cứ continue
            continue

        # DETECT plates
        plates = yolo_det(frame, size=args.imgsz)
        list_plates = plates.pandas().xyxy[0].values.tolist()

        plates_count = 0

        for plate in list_plates:
            x1, y1, x2, y2, conf, cls, name = plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # nới bbox chút cho OCR dễ hơn
            pad = 6
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1]-1, x2 + pad); y2 = min(frame.shape[0]-1, y2 + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # OCR y như code PC của bạn (deskew + helper.read_plate)
            lp_text = "unknown"
            found = False
            for cc in range(0, 2):
                for ct in range(0, 2):
                    lp_text = helper.read_plate(yolo_ocr, utils_rotate.deskew(crop, cc, ct))
                    if lp_text != "unknown":
                        found = True
                        break
                if found:
                    break

            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if found:
                plates_count += 1
                cv2.putText(frame, lp_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev_t))
        prev_t = now

        cv2.putText(frame, f"FPS {fps:.1f} plates={plates_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if args.show == 1:
            cv2.imshow("ALPR", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
