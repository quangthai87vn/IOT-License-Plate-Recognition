#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ALPR with CSI camera.

Usage:
  python3 csi.py

Optional env:
  CSI_WIDTH=1280 CSI_HEIGHT=720 CSI_FPS=30 CSI_SENSOR_MODE=3
  SHOW=1 IMG_SIZE=640 CONF=0.25 IOU=0.45
"""
import os

os.environ["SRC"] = "csi"

from webcam_onnx import main

if __name__ == "__main__":
    main()
