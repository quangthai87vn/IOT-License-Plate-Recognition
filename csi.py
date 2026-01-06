#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

def sh(cmd):
    return subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    # ÉP CSI mượt: mode 3 @30fps (đỡ giật hơn mode 5 @120fps)
    os.environ.setdefault("SRC", "csi")
    os.environ.setdefault("CSI_SENSOR_ID", "0")
    os.environ.setdefault("CSI_SENSOR_MODE", "3")
    os.environ.setdefault("CSI_W", "1280")
    os.environ.setdefault("CSI_H", "720")
    os.environ.setdefault("CSI_FPS", "30")
    os.environ.setdefault("OCR_EVERY", "3")     # giảm tải OCR
    os.environ.setdefault("IMG_SIZE", "640")    # giảm xuống 416 nếu muốn mượt hơn
    os.environ.setdefault("OCR_SIZE", "320")
    os.environ.setdefault("SHOW", "1")

    # optional: cleanup Argus nếu hay bị kẹt camera
    if os.getenv("RESET_ARGUS", "0") == "1":
        sh("pkill -f gst-launch || true")
        sh("sudo systemctl restart nvargus-daemon || true")

    from webcam_onnx import main
    sys.exit(main() or 0)
