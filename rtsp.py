#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ALPR with RTSP stream.

Usage:
  RTSP_URL='rtsp://user:pass@ip:554/...' python3 rtsp.py

Optional env:
  RTSP_LATENCY=100  (ms)
  RTSP_CODEC=h264|h265
  SHOW=1 IMG_SIZE=640 CONF=0.25 IOU=0.45
"""
import os

os.environ["SRC"] = "rtsp"

# Allow user to pass RTSP URL as first CLI arg too
import sys
if len(sys.argv) >= 2 and sys.argv[1].startswith("rtsp"):
    os.environ["RTSP_URL"] = sys.argv[1]

from webcam_onnx import main

if __name__ == "__main__":
    main()
