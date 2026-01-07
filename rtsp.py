#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shortcut chạy RTSP.

Ví dụ:
  python3 rtsp.py rtsp://192.168.50.2:8554/mac --show
hoặc:
  python3 rtsp.py --rtsp rtsp://192.168.50.2:8554/mac --show

Nó gọi webcam_onnx.py với --source rtsp.
"""

import sys
import subprocess


def main():
    args = sys.argv[1:]

    # Nếu tham số đầu là rtsp://... thì chuyển thành --rtsp <url>
    if args and args[0].startswith("rtsp://") and ("--rtsp" not in args):
        args = ["--rtsp", args[0]] + args[1:]

    cmd = [sys.executable, "webcam_onnx.py", "--source", "rtsp"] + args
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
