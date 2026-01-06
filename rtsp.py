#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

if __name__ == "__main__":
    os.environ["SRC"] = "rtsp"

    # bắt buộc: RTSP_URL
    if not os.getenv("RTSP_URL", ""):
        print('Bạn cần set RTSP_URL. Ví dụ:')
        print('RTSP_URL="rtsp://192.168.50.2:8554/mac" python3 rtsp.py')
        sys.exit(1)

    os.environ.setdefault("RTSP_LATENCY", "120")
    os.environ.setdefault("RTSP_CODEC", "h264")

    os.environ.setdefault("DET_ONNX", "./model/LP_detector_nano_61.onnx")
    os.environ.setdefault("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "webcam_onnx.py")
    sys.exit(subprocess.call([sys.executable, script]))
