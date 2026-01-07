#!/usr/bin/env python3
import os
import sys
import runpy

# RTSP wrapper
os.environ["ALPR_SOURCE"] = "rtsp"

if len(sys.argv) < 2:
    print("Usage: python3 rtsp.py <rtsp_url>")
    sys.exit(1)

os.environ["RTSP_URL"] = sys.argv[1]
runpy.run_path("webcam_onnx.py", run_name="__main__")
