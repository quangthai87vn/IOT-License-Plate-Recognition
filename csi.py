#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

if __name__ == "__main__":
    os.environ["SRC"] = "csi"
    # ép CSI về 720p@30 cho mượt
    os.environ.setdefault("CSI_W", "1280")
    os.environ.setdefault("CSI_H", "720")
    os.environ.setdefault("CSI_FPS", "30")

    # model paths (bạn đổi nếu khác)
    os.environ.setdefault("DET_ONNX", "./model/LP_detector_nano_61.onnx")
    os.environ.setdefault("OCR_ONNX", "./model/LP_ocr_nano_62.onnx")

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "webcam_onnx.py")
    sys.exit(subprocess.call([sys.executable, script]))
