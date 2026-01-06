#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("SRC", "rtsp")
    # bạn set RTSP_URL ngoài lệnh cũng được, ví dụ:
    # RTSP_URL="rtsp://192.168.50.2:8554/mac" python3 rtsp.py
    os.environ.setdefault("RTSP_LATENCY", "200")
    os.environ.setdefault("RTSP_CODEC", "h264")
    os.environ.setdefault("OCR_EVERY", "3")
    os.environ.setdefault("IMG_SIZE", "640")
    os.environ.setdefault("OCR_SIZE", "320")
    os.environ.setdefault("SHOW", "1")

    from webcam_onnx import main
    sys.exit(main() or 0)
