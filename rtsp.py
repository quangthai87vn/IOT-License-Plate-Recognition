#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from webcam_onnx import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Thiếu RTSP URL. Ví dụ: python3 rtsp.py \"rtsp://192.168.1.10:554/stream\"")
        sys.exit(2)
    url = sys.argv[1]
    raise SystemExit(main(["--source", "rtsp", "--rtsp", url, "--show", "1"]))
